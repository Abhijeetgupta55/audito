"""Production FastAPI application — Audito skin & hair diagnostic system."""
import logging
import asyncio
import time
from datetime import datetime
from typing import Optional, List, Dict, Any
import base64
import uuid

# ── In-process product cache (stage1 background prefetch → stage2 lookup) ────
# Key: session_key (UUID). Value: {products, kb_context, expires_at}
# TTL=300s. Cache misses just fall through to a fresh product search.
_reco_cache: Dict[str, Dict] = {}

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from backend.config import settings
from backend.logging_config import setup_logging, get_logger
from backend.agents import get_orchestrator
from backend.vector_store import get_pinecone_store
from backend.evaluations import EvaluationMetrics
from backend.monitoring import setup_langfuse
from backend import progress_store

# Setup logging
setup_logging(settings.DEBUG)
logger = get_logger(__name__)

# FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="Audito — multi-agent dermatology AI with real diagnosis workflows",
)

_origins_raw = settings.ALLOWED_ORIGINS.strip()
_allowed_origins = (
    [o.strip() for o in _origins_raw.split(",") if o.strip()]
    if _origins_raw
    else ["*"]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=len(_allowed_origins) > 1 or _allowed_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if settings.LANGFUSE_ENABLED:
    setup_langfuse(settings.LANGFUSE_PUBLIC_KEY, settings.LANGFUSE_SECRET_KEY)

evals = EvaluationMetrics()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every inbound request immediately — before any route logic runs."""
    logger.info(f"→ {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"← {response.status_code} {request.url.path}")
    return response


# ============================================================================
# MODELS
# ============================================================================

class HistoryMessage(BaseModel):
    role: str   # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    conversation_history: Optional[List[HistoryMessage]] = None


class ChatResponse(BaseModel):
    request_id: str
    intent: str               # "concern" | "chit_chat" | "follow_up" | "unknown"
    concern: str
    severity: str = "mild"
    diagnosis: str            # clinical assessment text (fallback)
    diagnosis_data: dict = {}        # structured diagnosis {concerns, severity, diagnosis_summary, cautions, requires_doctor}
    ingredient_rationale: str = ""   # text fallback for actives
    actives: list = []               # structured actives [{name, mechanism, target_concern}]
    recommendation_data: dict = {}   # structured product rationale {product_rationale[], caution}
    recommendation: str       # per-product rationale (text fallback)
    products: list
    show_products: bool
    agent_path: list
    safety_passed: bool
    warnings: list
    latency_ms: float
    tokens_used: int


class DoctorFeedback(BaseModel):
    recommendation_id: str
    quality_score: int
    product_relevance_score: int
    safety_score: int
    comments: Optional[str] = None
    improvements: Optional[str] = None


class RecommendProductsRequest(BaseModel):
    """Stage 2 request — sent when user clicks 'Get Product Recommendations'."""
    session_key: Optional[str] = None        # cache key from stage1 response
    identified_concern: str
    severity: str = "mild"
    skin_type: str = "unknown"
    user_message: str = ""
    kb_context: str = ""                     # KB chunks retrieved in stage1
    diagnosis: str = ""
    ingredient_rationale: str = ""           # already generated in stage1
    skin_analysis: Optional[Dict[str, Any]] = None


# ============================================================================
# STARTUP
# ============================================================================

@app.get("/test-gemini")
async def test_gemini():
    """Diagnostic endpoint — tests raw Gemini connectivity in isolation."""
    from backend.agents import _ensure_gemini, _agenerate
    key_set = bool(settings.GEMINI_API_KEY)
    if not key_set:
        return {"status": "error", "reason": "GEMINI_API_KEY not set on this server"}
    client_ok = _ensure_gemini()
    if not client_ok:
        return {"status": "error", "reason": "Gemini client failed to initialize (key may be invalid)"}
    try:
        result = await _agenerate(["Say hello in exactly 5 words."], max_tokens=30, retries=1)
        if result:
            return {"status": "ok", "model": settings.GEMINI_MODEL, "response": result}
        return {
            "status": "error",
            "reason": "Gemini returned empty response — check Render logs for finish_reason",
            "model": settings.GEMINI_MODEL,
        }
    except Exception as e:
        return {"status": "error", "reason": str(e), "model": settings.GEMINI_MODEL}


@app.get("/health")
async def health_check():
    store = await get_pinecone_store()
    gemini_configured = bool(settings.GEMINI_API_KEY)
    return {
        "status": "healthy" if gemini_configured else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "products_loaded": len(store.products),
        "model": settings.GEMINI_MODEL,
        "gemini_configured": gemini_configured,
        "warnings": [] if gemini_configured else ["GEMINI_API_KEY not set — LLM calls will fail"],
    }


@app.on_event("startup")
async def startup():
    logger.info("🚀 Audito API starting up")
    if settings.GEMINI_API_KEY:
        logger.info(f"✅ GEMINI_API_KEY set (model={settings.GEMINI_MODEL})")
    else:
        logger.error("❌ GEMINI_API_KEY is NOT set — all AI responses will use fallback text. Add this env var on Render.")
    store = await get_pinecone_store()
    logger.info(f"✅ Product store ready ({len(store.products)} products)")
    await get_orchestrator()
    logger.info("✅ Multi-agent orchestrator initialized")


# ============================================================================
# MAIN ENDPOINTS
# ============================================================================

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """Main chat endpoint.

    Agent pipeline:
      triage → [vision?] → [diagnosis | conversational] → [search → recommendation]? → safety
    """
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] Chat: {request.message[:100]}")

    try:
        orchestrator = await get_orchestrator()

        # Convert history to plain dicts
        history = [
            {"role": m.role, "content": m.content}
            for m in (request.conversation_history or [])
        ]

        state = await orchestrator.run(
            user_message=request.message,
            conversation_history=history,
        )

        # Build product list for response
        products_out = [
            {
                "id": p.get("id", ""),
                "name": p.get("name", ""),
                "brand": p.get("brand", ""),
                "description": p.get("description", ""),
                "key_ingredients": p.get("key_ingredients", []),
                "how_to_use": p.get("how_to_use", ""),
                "price_range": p.get("price_range", ""),
                "format": p.get("format", ""),
                "score": p.get("similarity_score", 0),
            }
            for p in state.recommended_products
        ]

        response = ChatResponse(
            request_id=request_id,
            intent=state.intent,
            concern=state.identified_concern or "none",
            severity=state.severity or "mild",
            diagnosis=state.diagnosis,
            diagnosis_data=state.diagnosis_data,
            ingredient_rationale=state.ingredient_rationale,
            actives=state.actives,
            recommendation_data=state.recommendation_data,
            recommendation=state.recommendation_text or state.conversational_reply,
            products=products_out,
            show_products=state.show_products,
            agent_path=state.agent_history,
            safety_passed=state.safety_checks_passed,
            warnings=state.safety_warnings,
            latency_ms=state.total_latency_ms,
            tokens_used=state.tokens_used,
        )

        background_tasks.add_task(
            evals.evaluate_response,
            request_id,
            request.message,
            response.recommendation,
            response.products,
            response.latency_ms,
        )

        logger.info(
            f"[{request_id}] ✅ intent={state.intent} concern={state.identified_concern} "
            f"path={' → '.join(state.agent_history)} {state.total_latency_ms:.0f}ms"
        )
        return response

    except Exception as e:
        logger.error(f"[{request_id}] ❌ Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze-image")
async def analyze_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: Optional[str] = Form(None),
    message: Optional[str] = Form(None),
):
    """Stage 1: Vision → Diagnosis → KB retrieval → Ingredient rationale.
    Products are deferred to /api/recommend-products (stage 2).
    """
    request_id = str(uuid.uuid4())
    effective_user = user_id or "anonymous"
    user_message = message or "Please analyze my skin or hair condition from this photo."
    logger.info(f"[{request_id}] Image upload: {file.filename} user={effective_user} has_context={bool(message)}")

    try:
        contents = await file.read()
        image_base64 = base64.b64encode(contents).decode("utf-8")
        image_type = file.content_type or "image/jpeg"

        orchestrator = await get_orchestrator()
        state = await orchestrator.run(
            user_message=user_message,
            image_data=image_base64,
            image_type=image_type,
            stage="stage1",
        )

        logger.info(f"[{request_id}] ✅ Stage1 done | {state.total_latency_ms:.0f}ms")

        # ── Background: precompute product search for stage2 ──────────────────
        session_key = str(uuid.uuid4())
        if state.identified_concern and state.identified_concern not in ("none", "unclear_image", "vision_error", ""):
            background_tasks.add_task(
                _prefetch_products,
                session_key=session_key,
                concern=state.identified_concern,
                severity=state.severity,
                skin_type=state.skin_analysis.get("skin_type", "unknown") if state.skin_analysis else "unknown",
                user_message=user_message,
                kb_context=state.kb_context,
            )

        # ── Save progress ─────────────────────────────────────────────────────
        record_id = None
        if state.skin_analysis and state.skin_analysis.get("is_clear", True):
            metrics = state.skin_analysis.get("metrics", {})
            try:
                record_id = progress_store.save_analysis(
                    user_id=effective_user,
                    metrics=metrics,
                    skin_analysis=state.skin_analysis,
                )
            except Exception as e:
                logger.error(f"[{request_id}] Progress save failed: {e}")

        progress_data = None
        if record_id:
            try:
                progress_data = progress_store.get_progress_report(effective_user)
            except Exception as e:
                logger.error(f"[{request_id}] Progress fetch failed: {e}")

        return {
            "request_id": request_id,
            "session_key": session_key,
            "kb_context": state.kb_context,
            "record_id": record_id,
            "progress_report": progress_data,
            "intent": state.intent,
            "skin_analysis": state.skin_analysis,
            "identified_concern": state.identified_concern,
            "severity": state.severity,
            "diagnosis": state.diagnosis,
            "diagnosis_data": state.diagnosis_data,
            "ingredient_rationale": state.ingredient_rationale,
            "actives": state.actives,
            "recommendation_data": state.recommendation_data,
            "recommendation": state.recommendation_text or state.conversational_reply,
            "products": state.recommended_products,
            "show_products": state.show_products,
            "warnings": state.safety_warnings,
            "agent_path": state.agent_history,
            "latency_ms": state.total_latency_ms,
        }

    except Exception as e:
        logger.error(f"[{request_id}] ❌ Image analysis failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


# ── Background product prefetch (fires after stage1 response is sent) ─────────

async def _prefetch_products(
    session_key: str,
    concern: str,
    severity: str,
    skin_type: str,
    user_message: str,
    kb_context: str,
) -> None:
    """Pre-run product search so stage2 can use cache instead of waiting."""
    try:
        from backend.rag_store import get_rag_store
        query = f"Concern: {concern}. Severity: {severity}. Skin type: {skin_type}. User query: {user_message}"
        store = await get_rag_store()
        if store is not None:
            products = store.search_products(query, top_k=5)
        else:
            local_store = await get_pinecone_store()
            products = local_store.search_products(query, top_k=5)
        _reco_cache[session_key] = {
            "products": products,
            "kb_context": kb_context,
            "expires_at": time.time() + 300,
        }
        logger.info(f"Prefetch done [{session_key[:8]}]: {len(products)} products cached")
    except Exception as e:
        logger.error(f"Prefetch failed [{session_key[:8]}]: {e}")


# ── Stage 2: product recommendation ──────────────────────────────────────────

@app.post("/api/recommend-products")
async def recommend_products(request: RecommendProductsRequest):
    """Stage 2: retrieve products + generate rationale. Called when user clicks the button."""
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] Stage2 recommend: concern={request.identified_concern} key={request.session_key and request.session_key[:8]}")

    try:
        # Check prefetch cache
        cached: Optional[Dict] = None
        if request.session_key:
            entry = _reco_cache.get(request.session_key)
            if entry and entry.get("expires_at", 0) > time.time():
                cached = entry
                logger.info(f"[{request_id}] Cache HIT — skipping product search")
            else:
                logger.info(f"[{request_id}] Cache MISS — running product search")

        orchestrator = await get_orchestrator()
        state = await orchestrator.run(
            user_message=request.user_message or f"Concern: {request.identified_concern}",
            stage="stage2",
            prefill={
                "current_agent": "search",
                "identified_concern": request.identified_concern,
                "severity": request.severity,
                "kb_context": request.kb_context,
                "diagnosis": request.diagnosis,
                "ingredient_rationale": request.ingredient_rationale,
                "skin_analysis": request.skin_analysis or {},
                # If cache hit, inject products so ProductSearchAgent is skipped implicitly
                # by having retrieved_products already populated (search still runs but is fast)
                "retrieved_products": cached["products"] if cached else [],
            },
        )

        # If cache hit, override with cached products to avoid re-running search
        if cached and not state.retrieved_products:
            state.retrieved_products = cached["products"]

        logger.info(f"[{request_id}] ✅ Stage2 done | {state.total_latency_ms:.0f}ms | {len(state.recommended_products)} products")

        return {
            "request_id": request_id,
            "products": state.recommended_products,
            "recommendation": state.recommendation_text,
            "recommendation_data": state.recommendation_data,
            "show_products": state.show_products,
            "agent_path": state.agent_history,
            "latency_ms": state.total_latency_ms,
        }

    except Exception as e:
        logger.error(f"[{request_id}] ❌ Stage2 failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Progress endpoints ────────────────────────────────────────────────────────

@app.get("/api/progress/{user_id}")
async def get_progress(user_id: str):
    """Return full progress report: trends, comparisons, insights."""
    try:
        report = progress_store.get_progress_report(user_id)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/progress/{user_id}/history")
async def get_history(user_id: str):
    """Return raw history records for a user."""
    try:
        records = progress_store.get_history(user_id)
        return {"user_id": user_id, "count": len(records), "records": records}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/progress/{user_id}")
async def delete_progress(user_id: str):
    """Delete all progress data for a user."""
    try:
        progress_store.delete_history(user_id)
        return {"status": "deleted", "user_id": user_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/search")
async def search_products(concern: str, top_k: int = 5):
    """Direct product search by concern keyword."""
    request_id = str(uuid.uuid4())
    try:
        store = await get_pinecone_store()
        products = store.search_products(concern, top_k)
        return {"request_id": request_id, "concern": concern, "products": products, "count": len(products)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/feedback")
async def submit_feedback(feedback: DoctorFeedback):
    """Endpoint for doctors to provide feedback on recommendation quality."""
    request_id = str(uuid.uuid4())
    try:
        await evals.store_feedback(
            feedback.recommendation_id,
            feedback.quality_score,
            feedback.product_relevance_score,
            feedback.safety_score,
            feedback.comments,
            feedback.improvements,
        )
        return {"request_id": request_id, "status": "feedback_received"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/agents")
async def list_agents():
    return {
        "agents": [
            {"id": "triage", "name": "Triage Agent", "role": "Classifies intent: concern vs chit-chat vs follow-up"},
            {"id": "vision", "name": "Vision Agent", "role": "Analyzes uploaded skin/scalp photos"},
            {"id": "diagnosis", "name": "Diagnosis Agent", "role": "Clinical assessment of the identified concern"},
            {"id": "conversational", "name": "Conversational Agent", "role": "Handles greetings and non-concern messages"},
            {"id": "search", "name": "Product Search Agent", "role": "Searches product DB for concern-matched items"},
            {"id": "recommendation", "name": "Recommendation Agent", "role": "Writes clinical product rationale"},
            {"id": "doctor_expert", "name": "Doctor Expert Agent", "role": "Guidance for severe cases needing professional care"},
            {"id": "safety", "name": "Safety Agent", "role": "Final safety check and response assembly"},
        ]
    }


@app.get("/api/metrics")
async def get_metrics():
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "requests_processed": evals.requests_processed,
        "average_latency_ms": evals.get_average_latency(),
        "hallucination_rate": evals.get_hallucination_rate(),
        "safety_success_rate": evals.get_safety_success_rate(),
        "average_products_recommended": evals.get_average_products(),
        "doctor_feedback_received": evals.feedback_count,
    }


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info", reload=settings.DEBUG)
