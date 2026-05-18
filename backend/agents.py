"""Multi-agent pipeline for Audito skin & hair diagnostic assistant.

Agent routing:
  triage → (chit-chat) → conversational  [END — no products]
  triage → (concern) → diagnosis
  triage → (image) → vision → diagnosis
  diagnosis → (mild/moderate, no Rx needed) → search → recommendation → safety [END]
  diagnosis → (severe / Rx needed) → doctor_expert → safety [END]
"""
import json
import logging
import re
from typing import Any, Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, field

from google import genai
from google.genai import types as genai_types

from backend.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configure Gemini once
# ---------------------------------------------------------------------------

_client: Optional["genai.Client"] = None


def _ensure_gemini() -> bool:
    global _client
    if _client is None and settings.GEMINI_API_KEY:
        _client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _client is not None


_SAFETY_OFF = [
    genai_types.SafetySetting(category="HARM_CATEGORY_HARASSMENT",        threshold="BLOCK_NONE"),
    genai_types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH",       threshold="BLOCK_NONE"),
    genai_types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
    genai_types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
]


def _safe_text(response) -> str:
    """Extract text from a Gemini response without crashing on blocked outputs."""
    try:
        return response.text or ""
    except Exception:
        pass
    try:
        for candidate in (response.candidates or []):
            for part in (candidate.content.parts or []):
                t = getattr(part, "text", None)
                if t:
                    return t
    except Exception:
        pass
    return ""


def _generate(prompt_parts: list, model_name: str = None, max_tokens: int = None) -> str:
    """Call Gemini and return the text response. Never raises."""
    if not _ensure_gemini():
        return ""
    name = model_name or settings.GEMINI_MODEL
    try:
        cfg = genai_types.GenerateContentConfig(
            safety_settings=_SAFETY_OFF,
            max_output_tokens=max_tokens,
            thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
        )
        response = _client.models.generate_content(
            model=name,
            contents=prompt_parts,
            config=cfg,
        )
        return _safe_text(response)
    except Exception as e:
        err = str(e)
        if "429" in err:
            logger.warning(f"Rate limited on {name} — quota exhausted, returning empty")
        else:
            logger.error(f"Gemini generate failed [{name}]: {err[:200]}")
        return ""


def _parse_json(text: str) -> Dict:
    """Strip markdown fences and parse JSON."""
    text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    # Handle potential trailing commas
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON object
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return {}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class AgentState:
    user_message: str
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    image_data: Optional[str] = None
    image_type: Optional[str] = None
    stage: str = "full"  # "full" | "stage1" | "stage2"

    current_agent: str = "triage"
    agent_history: List[str] = field(default_factory=list)

    # Triage output
    intent: str = "unknown"          # "concern" | "chit_chat" | "follow_up" | "unknown"
    identified_concern: str = ""
    severity: str = "mild"
    requires_doctor: bool = False

    # Diagnosis output
    diagnosis: str = ""              # structured clinical summary for user
    diagnosis_data: Dict[str, Any] = field(default_factory=dict)

    # Vision output
    skin_analysis: Dict[str, Any] = field(default_factory=dict)
    vision_feedback: str = ""        # if image was unclear

    # Search / recommendation
    search_query: str = ""
    kb_context: str = ""
    retrieved_products: List[Dict[str, Any]] = field(default_factory=list)
    recommended_products: List[Dict[str, Any]] = field(default_factory=list)
    actives: List[Dict[str, str]] = field(default_factory=list)  # [{name, mechanism, target_concern}]
    ingredient_rationale: str = ""   # text fallback for actives
    recommendation_text: str = ""    # per-product clinical rationale (text)
    recommendation_data: Dict[str, Any] = field(default_factory=dict)  # structured product rationale

    # Conversational (chit-chat / non-concern)
    conversational_reply: str = ""

    # Safety
    safety_checks_passed: bool = True
    safety_warnings: List[str] = field(default_factory=list)

    # Final assembled response
    final_response: str = ""
    show_products: bool = False

    # Metrics
    total_latency_ms: float = 0
    tokens_used: int = 0


# ---------------------------------------------------------------------------
# CONCERN keywords for fallback triage
# ---------------------------------------------------------------------------

CONCERN_KEYWORDS = {
    "acne": ["acne", "pimple", "breakout", "zit", "blemish", "blackhead", "whitehead", "comedone"],
    "dry_skin": ["dry skin", "dry", "flaky", "tight skin", "dehydrated", "peeling skin"],
    "oily_skin": ["oily", "greasy", "shiny skin", "excess oil", "sebum", "large pores"],
    "sensitive_skin": ["sensitive", "redness", "irritat", "rash", "itch", "stinging", "burning"],
    "dark_spots": ["dark spot", "pigmentation", "hyperpigmentation", "uneven tone", "melasma", "scar"],
    "anti_aging": ["wrinkle", "fine line", "aging", "sagging", "elasticity"],
    "hair_loss": ["hair loss", "hair fall", "thinning", "shedding", "bald", "alopecia"],
    "dandruff": ["dandruff", "flaking", "scalp itch", "itchy scalp", "flaky scalp"],
    "dull_skin": ["dull", "glow", "radiance", "brightening", "dark complexion"],
    "general_skincare": [
        "skin", "hair", "skincare", "haircare", "improve", "routine", "texture",
        "complexion", "what should i use", "what can i", "recommend", "advice",
        "tips", "regimen", "products for", "care for", "help with my", "any thoughts",
        "what to use", "how to", "should i", "ingredient", "moisturizer", "serum",
        "cleanser", "sunscreen", "spf", "toner", "exfoliat",
    ],
}

CHIT_CHAT_PATTERNS = [
    "hi", "hello", "hey", "hi there", "hey there", "how are you", "what can you do", "who are you",
    "thanks", "thank you", "ok", "okay", "bye", "good morning", "good evening", "good night",
    "what are you", "what are u", "what do you do", "what is audito",
]


def _fallback_triage(message: str) -> Dict[str, Any]:
    msg = message.lower().strip()

    # Check chit-chat first
    for pat in CHIT_CHAT_PATTERNS:
        if pat in msg and len(msg) < 80:
            return {"intent": "chit_chat", "concern": "none", "severity": "mild",
                    "requires_doctor": False}

    # Check concern keywords
    for concern, keywords in CONCERN_KEYWORDS.items():
        if any(kw in msg for kw in keywords):
            severity = "moderate" if any(w in msg for w in ["severe", "bad", "worse", "persistent", "weeks"]) else "mild"
            return {"intent": "concern", "concern": concern, "severity": severity,
                    "requires_doctor": severity == "severe"}

    # Any skin/hair word present but didn't match a specific concern → general skincare
    general_terms = ["skin", "hair", "face", "scalp", "complexion", "pore", "glow", "routine",
                     "product", "moistur", "serum", "cleanser", "sunscreen", "spf", "exfoliat"]
    if any(t in msg for t in general_terms):
        return {"intent": "concern", "concern": "general_skincare", "severity": "mild",
                "requires_doctor": False}

    # Unknown — ask user to elaborate
    return {"intent": "unknown", "concern": "none", "severity": "mild", "requires_doctor": False}


# ---------------------------------------------------------------------------
# Agent 1 — Triage
# Decides: chit_chat | concern | follow_up | unknown
# ---------------------------------------------------------------------------

class TriageAgent:
    SYSTEM = """You are a clinical triage AI for Audito — a skin and hair health tracker.

Classify the user message into ONE intent:
- "concern"    → ANY message about skin, hair, skincare, haircare, ingredients, routines, or improvement. Includes general questions like "how do I improve my skin", "any thoughts on my skin", "what should I use", "skincare tips", "what's good for oily skin". Use concern="general_skincare" when the question is non-specific.
- "chit_chat"  → ONLY pure greetings/farewells/thanks with zero skin or hair content (e.g. "hi", "thanks", "bye"). Nothing else.
- "follow_up"  → user is continuing a previous skin/hair conversation (uses "it", "that", "what about", "also")
- "unknown"    → truly ambiguous with no skin/hair context whatsoever

When in doubt between "concern" and "chit_chat", choose "concern".

Respond ONLY with valid JSON:
{"intent": "concern|chit_chat|follow_up|unknown", "concern": "concern_tag_or_none", "severity": "mild|moderate|severe", "requires_doctor": false, "reasoning": "1 sentence"}"""

    async def process(self, state: AgentState) -> Dict[str, Any]:
        logger.info("Triage Agent running")

        # Skip LLM for image uploads — always routes to vision regardless of intent
        if state.image_data:
            logger.info("Triage: image detected, skipping LLM and routing directly to vision")
            return {
                "intent": "concern",
                "identified_concern": "",
                "severity": "mild",
                "requires_doctor": False,
                "current_agent": "vision",
                "agent_history": ["triage"],
            }

        result = None
        if _ensure_gemini():
            try:
                text = _generate([self.SYSTEM, f"\nUser message: {state.user_message}"], max_tokens=200)
                result = _parse_json(text)
                logger.info(f"Triage LLM result: {result}")
            except Exception as e:
                logger.error(f"Triage LLM failed: {e}. Using fallback.")

        if not result:
            result = _fallback_triage(state.user_message)

        intent = result.get("intent", "unknown")
        concern = result.get("concern", "none")
        severity = result.get("severity") or "mild"
        requires_doctor = bool(result.get("requires_doctor", False))

        # Routing
        if state.image_data:
            next_agent = "vision"
        elif intent == "concern":
            next_agent = "diagnosis"
        elif intent == "follow_up":
            next_agent = "diagnosis"
        elif intent == "chit_chat":
            next_agent = "conversational"
        else:
            next_agent = "conversational"  # Ask for clarification

        return {
            "intent": intent,
            "identified_concern": concern if concern != "none" else "",
            "severity": severity,
            "requires_doctor": requires_doctor,
            "current_agent": next_agent,
            "agent_history": ["triage"],
        }


# ---------------------------------------------------------------------------
# Agent 2 — Vision (image quality check + photo analysis)
# ---------------------------------------------------------------------------

class VisionAgent:
    SYSTEM = """You are a clinical dermatology AI analyzing a skin or scalp photo.

STEP 1 — Image quality check:
Is the image clear, well-lit, in focus, and does it show the skin/scalp area of concern?

STEP 2 — If quality is adequate, analyze:
- Skin/scalp type (oily / dry / combination / normal / unknown)
- Visible conditions (state ONLY what is clearly visible — be conservative)
- Severity: mild / moderate / severe
- Primary concern tag: one of [acne, oily_skin, dry_skin, sensitive_skin, dark_spots, anti_aging, hair_loss, dandruff, dull_skin, large_pores]
- Clinical observation: 2-3 factual sentences about what you see
- Structured metrics (0-10 scale, 0 = none/perfect, 10 = severe/worst):
  * acne_severity: visible acne/pimples/breakouts
  * redness: visible redness/inflammation
  * pigmentation: dark spots/uneven tone/hyperpigmentation
  * hydration: dryness (0 = well hydrated, 10 = very dry/flaky)
  * texture: roughness/unevenness (0 = smooth, 10 = rough)
  * dark_circles: under-eye darkness (if visible, else 0)
  * wrinkles: visible fine lines/wrinkles (0 = none, 10 = deep)
  * hair_thinning: visible hair thinning/loss (if applicable, else 0)
  * scalp_condition: scalp issues like dandruff/irritation (0 = healthy, 10 = severe)
  * confidence_score: your confidence in this analysis (0-1 float)

Rules:
- Never diagnose or prescribe medication
- Only state what is clearly visible; if uncertain, say so
- Do NOT invent conditions not visible in the image
- Set metrics to 0 if not visible or not applicable

Respond ONLY in valid JSON:
{"is_clear": true, "clarity_feedback": "", "skin_type": "oily|dry|combination|normal|unknown", "conditions": ["..."], "severity": "mild|moderate|severe", "primary_concern": "concern_tag", "clinical_observation": "...", "metrics": {"acne_severity": 0, "redness": 0, "pigmentation": 0, "hydration": 0, "texture": 0, "dark_circles": 0, "wrinkles": 0, "hair_thinning": 0, "scalp_condition": 0, "confidence_score": 0.8}}"""

    async def process(self, state: AgentState) -> Dict[str, Any]:
        logger.info("Vision Agent running")

        if not state.image_data:
            return {"current_agent": "diagnosis"}

        if not _ensure_gemini():
            return {
                "skin_analysis": {"skin_type": "unknown", "conditions": [state.identified_concern or "general"]},
                "current_agent": "diagnosis",
                "agent_history": state.agent_history + ["vision"],
            }

        try:
            import io
            import base64

            img_bytes = base64.b64decode(state.image_data)

            # Use inline bytes via genai_types.Part
            img_part = genai_types.Part.from_bytes(
                data=img_bytes,
                mime_type=state.image_type or "image/jpeg",
            )
            text = _generate([self.SYSTEM, img_part], model_name=settings.GEMINI_VISION_MODEL, max_tokens=800)
            analysis = _parse_json(text)
            logger.info(f"Vision analysis: is_clear={analysis.get('is_clear')}")

            if not analysis or analysis.get("is_clear") is False:
                feedback = analysis.get("clarity_feedback", "The photo was not clear enough or could not be analyzed.") if analysis else "The photo was not clear enough or could not be analyzed."
                safe_analysis = analysis or {"is_clear": False, "clarity_feedback": feedback, "metrics": {}}
                return {
                    "skin_analysis": safe_analysis,
                    "vision_feedback": feedback,
                    "identified_concern": "unclear_image",
                    "current_agent": "conversational",
                    "agent_history": state.agent_history + ["vision"],
                }

            concern = analysis.get("primary_concern") or state.identified_concern or "general"
            metrics = analysis.get("metrics", {})
            # Attach metrics directly to analysis dict so they travel with skin_analysis
            analysis["metrics"] = metrics
            return {
                "skin_analysis": analysis,
                "identified_concern": concern,
                "severity": analysis.get("severity", state.severity),
                "current_agent": "diagnosis",
                "agent_history": state.agent_history + ["vision"],
            }

        except Exception as e:
            logger.error(f"Vision Agent failed: {e}")
            return {
                "current_agent": "diagnosis",
                "agent_history": state.agent_history + ["vision"],
            }


# ---------------------------------------------------------------------------
# Agent 3 — Clinical Intake + Differential Diagnosis
#
# Turn 1: User describes concern → system checks if enough info exists.
#   If NOT enough → asks up to 3 targeted clinical questions. [END turn]
#   If enough info → runs full differential diagnosis. → routes to search/doctor
#
# "Enough info" means we know at least:
#   - Duration (how long)
#   - Location / affected area
#   - Any associated symptoms (itch, pain, discharge, etc.)
# ---------------------------------------------------------------------------

class DiagnosisAgent:

    # --- Phase 1: Intake evaluator ---
    INTAKE_SYSTEM = """You are a clinical intake AI for a dermatology assistant.

Your job: decide if we have ENOUGH information to attempt a differential diagnosis.

We need at minimum:
1. Duration — how long has this been present?
2. Location — exactly where on the body/scalp?
3. Symptoms — any itch, pain, burning, discharge, spreading?

Also useful (not mandatory):
- Any recent changes (new products, food, medications, stress, travel)?
- Family history of skin/hair conditions?
- Previous treatments tried?

Review the full conversation history provided and decide:
- If MISSING 2 or more of the mandatory items → set "need_more_info": true
- If we have enough → set "need_more_info": false

If need_more_info is true, write 2-3 specific, natural-sounding follow-up questions targeting only what is MISSING.
Do NOT ask for things already answered.

Respond ONLY in valid JSON:
{
  "need_more_info": true,
  "missing": ["duration", "location", "symptoms"],
  "questions": "One conversational message asking the missing questions naturally.",
  "collected": {
    "duration": "extracted value or null",
    "location": "extracted value or null",
    "symptoms": "extracted value or null",
    "triggers": "extracted value or null",
    "treatments_tried": "extracted value or null"
  }
}"""

    # --- Phase 2: Differential diagnosis ---
    DIFFERENTIAL_SYSTEM = """You are a clinical dermatologist AI. Output ONLY valid JSON — no prose, no markdown fences.

{
  "concerns": ["primary finding in 4-6 words"],
  "severity": "mild|moderate|severe",
  "diagnosis_summary": [
    "consistent with X — differential if relevant",
    "cannot confirm without physical exam — specific limitation",
    "next step — product-based or dermatologist referral"
  ],
  "cautions": [],
  "requires_doctor": false
}

Rules:
- diagnosis_summary: exactly 3 strings, 8-15 words each
- concerns: 1-2 concise items
- cautions: 0-1 items only
- Conservative language — never a single definitive diagnosis
- requires_doctor: true only for severe presentations"""

    def _build_history_text(self, history: list, current_message: str) -> str:
        """Format conversation history into a readable clinical transcript."""
        lines = []
        for msg in history:
            role = "Patient" if msg.get("role") == "user" else "Assistant"
            lines.append(f"{role}: {msg.get('content', '')}")
        lines.append(f"Patient: {current_message}")
        return "\n".join(lines)

    async def process(self, state: AgentState) -> Dict[str, Any]:
        logger.info("Diagnosis Agent running")

        if not _ensure_gemini():
            return {
                "diagnosis": (
                    "I'm unable to provide an assessment right now due to a configuration issue. "
                    "Please consult a dermatologist directly."
                ),
                "show_products": False,
                "current_agent": None,
                "agent_history": state.agent_history + ["diagnosis"],
            }

        # General skincare inquiry — skip clinical intake, route straight to product search
        if state.identified_concern == "general_skincare" and not state.image_data:
            logger.info("General skincare inquiry — routing to search without intake")
            return {
                "diagnosis": "",
                "current_agent": "search",
                "agent_history": state.agent_history + ["diagnosis"],
            }

        history_text = self._build_history_text(
            state.conversation_history, state.user_message
        )

        # Vision context
        vision_context = ""
        if state.skin_analysis and state.skin_analysis.get("clinical_observation"):
            obs = state.skin_analysis.get("clinical_observation", "")
            skin_type = state.skin_analysis.get("skin_type", "unknown")
            conds = ", ".join(state.skin_analysis.get("conditions", []))
            vision_context = (
                f"\n\n[PHOTO ANALYSIS]\n"
                f"Skin type: {skin_type}\n"
                f"Conditions observed: {conds}\n"
                f"Clinical observation: {obs}"
            )

        # ── PHASE 1: Do we have enough info? ──────────────────────────────

        # When we have clear photo analysis, vision already provides location + symptoms.
        # Skip intake — go straight to differential diagnosis.
        if state.image_data and state.skin_analysis and state.skin_analysis.get("clinical_observation"):
            need_more = False
            logger.info("Intake skipped — photo analysis provides sufficient context")
        else:
            intake_prompt = (
                f"Full conversation so far:\n{history_text}"
                f"{vision_context}\n\n"
                f"Concern category: {state.identified_concern.replace('_', ' ')}"
            )
            try:
                intake_raw = _generate([self.INTAKE_SYSTEM, intake_prompt], max_tokens=150)
                intake = _parse_json(intake_raw)
            except Exception as e:
                logger.error(f"Intake evaluation failed: {e}")
                intake = {"need_more_info": False}

            need_more = intake.get("need_more_info", False)

        if need_more:
            questions = intake.get(
                "questions",
                "Could you tell me how long you've been experiencing this, where exactly it is, "
                "and whether you have any itching, pain, or other symptoms?"
            )
            logger.info("Intake: need more info, asking follow-up questions")
            # This ends the pipeline here — waiting for user reply
            return {
                "diagnosis": questions,
                "final_response": questions,
                "show_products": False,
                "current_agent": None,
                "agent_history": state.agent_history + ["diagnosis"],
            }

        # ── PHASE 2: Full differential diagnosis ──────────────────────────
        logger.info("Intake complete — running differential diagnosis")

        diff_prompt = (
            f"Full patient history:\n{history_text}"
            f"{vision_context}\n\n"
            f"Concern category: {state.identified_concern.replace('_', ' ')}\n"
            f"Reported severity: {state.severity}"
        )

        diagnosis_data_parsed: Dict[str, Any] = {}
        try:
            raw_diff = _generate([self.DIFFERENTIAL_SYSTEM, diff_prompt], max_tokens=350).strip()
            if raw_diff:
                diagnosis_data_parsed = _parse_json(raw_diff)
                if diagnosis_data_parsed and diagnosis_data_parsed.get("diagnosis_summary"):
                    diagnosis_text = "\n".join(
                        f"• {s}" for s in diagnosis_data_parsed["diagnosis_summary"]
                    )
                else:
                    # LLM returned plain text instead of JSON — use as-is
                    diagnosis_text = raw_diff
                    diagnosis_data_parsed = {}
            else:
                # Rate limited or empty — fall through to fallback below
                raise ValueError("LLM returned empty (rate limited)")

        except Exception as e:
            logger.error(f"Differential diagnosis LLM failed: {e}")
            concern_label = state.identified_concern.replace("_", " ")
            if state.skin_analysis and state.skin_analysis.get("clinical_observation"):
                conds = ", ".join(state.skin_analysis.get("conditions", []))
                diagnosis_text = (
                    f"• Findings consistent with {conds or concern_label}.\n"
                    "• Extent and severity cannot be confirmed without physical examination.\n"
                    "• Review matched products below; consult a dermatologist for a confirmed diagnosis."
                )
            else:
                diagnosis_text = (
                    f"• Description consistent with {concern_label}.\n"
                    "• Cannot assess severity or rule out differentials without examination.\n"
                    "• Provide more detail or consult a dermatologist for a confirmed diagnosis."
                )
            return {
                "diagnosis": diagnosis_text,
                "diagnosis_data": {},
                "show_products": False,
                "current_agent": "search" if state.identified_concern else "safety",
                "agent_history": state.agent_history + ["diagnosis"],
            }

        # ── Determine routing ─────────────────────────────────────────────
        parsed_severity = diagnosis_data_parsed.get("severity") or state.severity or "mild"
        parsed_requires_doctor = bool(diagnosis_data_parsed.get("requires_doctor", state.requires_doctor))

        if parsed_requires_doctor or parsed_severity == "severe":
            next_agent = "doctor_expert"
        else:
            next_agent = "search"

        return {
            "diagnosis": diagnosis_text,
            "diagnosis_data": diagnosis_data_parsed,
            "severity": parsed_severity,
            "requires_doctor": parsed_requires_doctor,
            "current_agent": next_agent,
            "agent_history": state.agent_history + ["diagnosis"],
        }


# ---------------------------------------------------------------------------
# Agent 4 — Conversational (chit-chat, unclear, image-unclear)
# ---------------------------------------------------------------------------

class ConversationalAgent:
    SYSTEM = """You are Audito — a knowledgeable assistant specialising in skin and hair health, built as a personal skin and hair tracker.

Answer any question the user asks — greetings, general questions, whatever they say — naturally and helpfully. You are not restricted to skin/hair topics: if someone says "hi" reply warmly, if they ask something off-topic answer it briefly.

Your core expertise is dermatology and trichology. After answering any off-topic question, gently bring the conversation back to skin or hair health with a single short follow-up question or offer.

When skin/hair topics come up:
- Questions about Audito itself: explain that it analyzes skin/hair photos and concerns through a multi-agent AI pipeline, tracks metrics over time, and recommends products from a curated dermatology database.
- Vague skin/hair questions ("tips for my skin", "what should I use"): ask 1 targeted clarifying question — primary concern, how long, skin type — before advising.
- Specific skin/hair concerns: give direct, accurate clinical advice drawing on dermatology knowledge.

Style: direct, conversational, no filler phrases. Match the energy of the message."""

    async def process(self, state: AgentState) -> Dict[str, Any]:
        logger.info("Conversational Agent running")

        # Unclear image — give retake guidance
        if state.vision_feedback:
            reply = (
                f"I wasn't able to get a clear enough reading from your photo. {state.vision_feedback}\n\n"
                "For an accurate analysis, please retake the photo:\n"
                "• In natural daylight or bright indoor light (no flash)\n"
                "• Hold steady — no blur\n"
                "• Affected area clearly centred in frame\n"
                "• No heavy filters or editing"
            )
            return {
                "conversational_reply": reply,
                "final_response": reply,
                "show_products": False,
                "current_agent": None,
                "agent_history": state.agent_history + ["conversational"],
            }

        # Build context from conversation history so the LLM has full context
        history_lines = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Audito'}: {m['content']}"
            for m in (state.conversation_history or [])
        )
        context = (
            f"{history_lines}\nUser: {state.user_message}" if history_lines
            else f"User: {state.user_message}"
        )

        reply = _generate([self.SYSTEM, context], max_tokens=500).strip()

        if not reply:
            logger.warning("Conversational Agent: Gemini returned empty — API may be unavailable")
            reply = "I'm having trouble reaching the AI right now. Please try again in a moment."

        return {
            "conversational_reply": reply,
            "final_response": reply,
            "show_products": False,
            "current_agent": None,
            "agent_history": state.agent_history + ["conversational"],
        }


# ---------------------------------------------------------------------------
# Agent 5 — Product Search
# Only runs when there IS a diagnosed concern
# ---------------------------------------------------------------------------

class ProductSearchAgent:
    async def process(self, state: AgentState) -> Dict[str, Any]:
        logger.info(f"Product Search Agent running (stage={state.stage})")

        # Guard: skip if no real concern
        if not state.identified_concern or state.identified_concern in ("none", "unclear_image", ""):
            logger.info("No concern identified — skipping product search")
            return {
                "retrieved_products": [],
                "current_agent": "recommendation",
                "agent_history": state.agent_history + ["search"],
            }

        try:
            from backend.rag_store import get_rag_store
            store = await get_rag_store()

            skin_type = state.skin_analysis.get("skin_type", "") if state.skin_analysis else ""
            query = (
                f"Concern: {state.identified_concern}. Severity: {state.severity}. "
                f"Skin type: {skin_type}. User query: {state.user_message}"
            )

            if state.stage == "stage2":
                # KB context already retrieved in stage1 — only run product search
                if store is not None:
                    products = store.search_products(query, top_k=5)
                    logger.info(f"Stage2 RAG: {len(products)} products for {state.identified_concern}")
                else:
                    logger.warning("RAG unavailable, falling back to keyword search")
                    from backend.vector_store import get_pinecone_store
                    local = await get_pinecone_store()
                    products = local.search_products(query, top_k=5)
                return {
                    "search_query": query,
                    "retrieved_products": products,
                    "current_agent": "recommendation",
                    "agent_history": state.agent_history + ["search"],
                }

            # stage1 or full: always retrieve KB context
            kb_context = ""
            products = []
            if store is not None:
                kb_context = store.search_knowledge(query, top_k=3)
                if state.stage == "full":
                    products = store.search_products(query, top_k=5)
                    logger.info(f"RAG full: {len(products)} products, {len(kb_context)} KB chars")
                else:
                    logger.info(f"Stage1 RAG: KB only, {len(kb_context)} KB chars")
            else:
                logger.warning("RAG unavailable, falling back to keyword search")
                from backend.vector_store import get_pinecone_store
                local = await get_pinecone_store()
                # Always retrieve products for full; for stage1 also try KB from local store
                if state.stage == "full":
                    products = local.search_products(query, top_k=5)
                else:
                    # stage1: try to get KB context from local store search results as text
                    raw_products = local.search_products(query, top_k=3)
                    if raw_products:
                        kb_context = "\n".join(
                            f"{p.get('name','')}: {p.get('description','')} "
                            f"[Ingredients: {', '.join(p.get('key_ingredients',[])[:4])}]"
                            for p in raw_products
                        )
                        logger.info(f"Stage1 fallback KB: built from {len(raw_products)} products")

            return {
                "search_query": query,
                "kb_context": kb_context,
                "retrieved_products": products,
                "current_agent": "recommendation",
                "agent_history": state.agent_history + ["search"],
            }
        except Exception as e:
            logger.error(f"Product Search Agent failed: {e}")
            return {
                "retrieved_products": [],
                "kb_context": "",
                "current_agent": "recommendation",
                "agent_history": state.agent_history + ["search"],
            }


# ---------------------------------------------------------------------------
# Agent 6 — Recommendation
# Writes a clinical rationale for the selected products
# ---------------------------------------------------------------------------

class RecommendationAgent:
    # Stage1: ingredient reasoning from KB — no products needed
    INGREDIENTS_SYSTEM = """You are a clinical dermatology assistant. Output ONLY valid JSON.

{
  "actives": [
    {
      "name": "Ingredient Name",
      "mechanism": "improve collagen turnover",
      "target_concern": "wrinkles + texture"
    }
  ]
}

Rules:
- 2-4 actives only, grounded in the Knowledge Base provided
- name: title case ingredient name
- mechanism: 3-6 words, active verb (improve / reduce / support / boost / inhibit)
- target_concern: 2-5 words specific to this patient's concern
- No prose, no markdown, no product names"""

    # Stage2: per-product rationale — ingredient reasoning already shown
    PRODUCTS_SYSTEM = """You are a clinical dermatology assistant. Output ONLY valid JSON.

{
  "product_rationale": [
    {
      "product_name": "exact name from list",
      "key_active": "main active ingredient",
      "benefit": "specific benefit for this concern"
    }
  ],
  "caution": "one-line caution about sun sensitivity, interactions, or patch testing"
}

Rules:
- Max 3 products from the provided list only — never invent names
- benefit: 4-8 words, specific to this patient's concern
- caution: one sentence, omit if not applicable
- Do NOT re-explain ingredient actives — already shown"""

    # Full mode: both sections in one pass (used by text/chat pipeline)
    SYSTEM = """You are a clinical dermatology assistant. Output ONLY valid JSON.

{
  "actives": [
    {
      "name": "Ingredient Name",
      "mechanism": "improve collagen turnover",
      "target_concern": "wrinkles + texture"
    }
  ],
  "product_rationale": [
    {
      "product_name": "exact name from list",
      "key_active": "main active",
      "benefit": "specific benefit for this patient"
    }
  ],
  "caution": "one-line caution"
}

Rules:
- actives: 2-4 items, grounded in Knowledge Base
- product_rationale: max 3, only from provided list — never invent names
- caution: one sentence, omit if not applicable
- No intro, no markdown, no prose paragraphs"""

    async def process(self, state: AgentState) -> Dict[str, Any]:
        logger.info(f"Recommendation Agent running (stage={state.stage})")
        if state.stage == "stage1":
            return await self._run_stage1(state)
        if state.stage == "stage2":
            return await self._run_stage2(state)
        return await self._run_full(state)

    # ── Stage 1: ingredient rationale only ───────────────────────────────────
    async def _run_stage1(self, state: AgentState) -> Dict[str, Any]:
        obs = state.skin_analysis.get("clinical_observation", "") if state.skin_analysis else ""
        actives: List[Dict[str, str]] = []
        ingredient_rationale = ""
        if _ensure_gemini():
            try:
                ctx = (
                    f"Concern: {state.identified_concern.replace('_', ' ')} ({state.severity})\n"
                    f"Clinical observation: {obs or 'N/A'}\n\n"
                )
                if state.kb_context:
                    ctx += f"Dermatology Knowledge Context:\n{state.kb_context}"
                else:
                    ctx += "Apply established clinical dermatology knowledge for this concern."
                raw = _generate([self.INGREDIENTS_SYSTEM, ctx], max_tokens=350).strip()
                parsed = _parse_json(raw)
                actives = parsed.get("actives", []) if parsed else []
                if actives:
                    ingredient_rationale = "\n".join(
                        f"• {a['name']} — {a.get('mechanism', '')} — {a.get('target_concern', '')}"
                        for a in actives
                    )
            except Exception as e:
                logger.error(f"Stage1 ingredient rationale failed: {e}")
        return {
            "actives": actives,
            "ingredient_rationale": ingredient_rationale,
            "recommendation_text": "",
            "recommended_products": [],
            "show_products": False,
            "current_agent": "safety",
            "agent_history": state.agent_history + ["recommendation"],
        }

    # ── Stage 2: product text only ────────────────────────────────────────────
    async def _run_stage2(self, state: AgentState) -> Dict[str, Any]:
        products = state.retrieved_products
        if not products:
            return {
                "actives": state.actives,
                "ingredient_rationale": state.ingredient_rationale,
                "recommendation_text": "",
                "recommendation_data": {},
                "recommended_products": [],
                "show_products": False,
                "current_agent": "safety",
                "agent_history": state.agent_history + ["recommendation"],
            }
        product_context = "\n".join([
            f"- {p['name']} by {p['brand']}: {p.get('description', '')} "
            f"[Key ingredients: {', '.join(p.get('key_ingredients', [])[:3])}]"
            for p in products[:5]
        ])
        obs = state.skin_analysis.get("clinical_observation", "") if state.skin_analysis else ""
        rec_data: Dict[str, Any] = {}
        rec_text = ""
        if _ensure_gemini():
            try:
                ctx = (
                    f"Concern: {state.identified_concern.replace('_', ' ')} ({state.severity})\n"
                    f"Clinical observation: {obs or 'N/A'}\n\n"
                    f"Dermatology Knowledge Context:\n{state.kb_context or ''}\n\n"
                    f"Available products:\n{product_context}"
                )
                raw = _generate([self.PRODUCTS_SYSTEM, ctx], max_tokens=350).strip()
                rec_data = _parse_json(raw) or {}
                if rec_data.get("product_rationale"):
                    lines = [
                        f"• {pr['product_name']} — {pr.get('key_active', '')} → {pr.get('benefit', '')}"
                        for pr in rec_data["product_rationale"]
                    ]
                    if rec_data.get("caution"):
                        lines.append(f"⚠ {rec_data['caution']}")
                    rec_text = "\n".join(lines)
            except Exception as e:
                logger.error(f"Stage2 product recommendation failed: {e}")
        return {
            "actives": state.actives,
            "ingredient_rationale": state.ingredient_rationale,
            "recommendation_text": rec_text,
            "recommendation_data": rec_data,
            "recommended_products": products[:3],
            "show_products": True,
            "current_agent": "safety",
            "agent_history": state.agent_history + ["recommendation"],
        }

    # ── Full mode: both sections in one pass (text/chat pipeline) ─────────────
    async def _run_full(self, state: AgentState) -> Dict[str, Any]:
        products = state.retrieved_products
        if not products:
            return {
                "actives": [],
                "ingredient_rationale": "",
                "recommendation_text": "I couldn't find specific products in our database matching your concern. Please describe your concern in more detail, or consult a dermatologist for personalised advice.",
                "recommendation_data": {},
                "recommended_products": [],
                "show_products": False,
                "current_agent": "safety",
                "agent_history": state.agent_history + ["recommendation"],
            }
        product_context = "\n".join([
            f"- {p['name']} by {p['brand']}: {p.get('description', '')} "
            f"[Key ingredients: {', '.join(p.get('key_ingredients', [])[:3])}]"
            for p in products[:5]
        ])
        obs = state.skin_analysis.get("clinical_observation", "") if state.skin_analysis else ""
        rec_data: Dict[str, Any] = {}
        actives: List[Dict[str, str]] = []
        ingredient_rationale = ""
        rec_text = ""
        if _ensure_gemini():
            try:
                user_ctx = (
                    f"Concern: {state.identified_concern.replace('_', ' ')} ({state.severity})\n"
                    f"Clinical observation: {obs or 'N/A'}\n\n"
                    f"Dermatology Knowledge Context:\n{state.kb_context}\n\n"
                    f"Available products:\n{product_context}"
                )
                raw = _generate([self.SYSTEM, user_ctx], max_tokens=450).strip()
                rec_data = _parse_json(raw) or {}
                actives = rec_data.get("actives", [])
                if actives:
                    ingredient_rationale = "\n".join(
                        f"• {a['name']} — {a.get('mechanism', '')} — {a.get('target_concern', '')}"
                        for a in actives
                    )
                if rec_data.get("product_rationale"):
                    lines = [
                        f"• {pr['product_name']} — {pr.get('key_active', '')} → {pr.get('benefit', '')}"
                        for pr in rec_data["product_rationale"]
                    ]
                    if rec_data.get("caution"):
                        lines.append(f"⚠ {rec_data['caution']}")
                    rec_text = "\n".join(lines)
            except Exception as e:
                logger.error(f"Full recommendation LLM failed: {e}")
        return {
            "actives": actives,
            "ingredient_rationale": ingredient_rationale,
            "recommendation_text": rec_text,
            "recommendation_data": rec_data,
            "recommended_products": products[:3],
            "show_products": True,
            "current_agent": "safety",
            "agent_history": state.agent_history + ["recommendation"],
        }


# ---------------------------------------------------------------------------
# Agent 7 — Doctor Expert (severe / requires professional care)
# ---------------------------------------------------------------------------

class DoctorExpertAgent:
    SYSTEM = """You are a dermatologist giving urgent clinical guidance. Output exactly 4 bullets — no prose.

• **Urgency**: [why this severity requires professional care — 1 clause]
• **Do now**: [1-2 safe interim steps, comma-separated]
• **See**: [specialist type] — [what to expect at the appointment]
• **Avoid**: [1-2 things to avoid before appointment]

Rules:
- Exactly 4 bullets, 1 sentence each, no paragraphs
- No OTC product recommendations for severe cases
- Total output under 70 words"""

    async def process(self, state: AgentState) -> Dict[str, Any]:
        logger.info("Doctor Expert Agent running")

        guidance = ""
        if _ensure_gemini():
            try:
                ctx = (
                    f"Concern: {state.identified_concern.replace('_', ' ')} — {state.severity}\n"
                    f"User description: {state.user_message}"
                )
                guidance = _generate([self.SYSTEM, ctx], max_tokens=220).strip()
            except Exception as e:
                logger.error(f"Doctor Expert Agent failed: {e}")

        if not guidance:
            concern_label = state.identified_concern.replace("_", " ")
            guidance = (
                f"• **Urgency**: Severity warrants in-person evaluation — {concern_label} at {state.severity} level.\n"
                "• **Do now**: Keep area clean, apply fragrance-free moisturiser if needed.\n"
                "• **See**: Certified dermatologist — expect clinical examination and possible patch testing.\n"
                "• **Avoid**: Self-medicating, harsh actives, or squeezing/picking before appointment."
            )

        return {
            "recommendation_text": guidance,
            "show_products": False,
            "current_agent": "safety",
            "agent_history": state.agent_history + ["doctor_expert"],
        }


# ---------------------------------------------------------------------------
# Agent 8 — Safety
# ---------------------------------------------------------------------------

class SafetyAgent:
    async def process(self, state: AgentState) -> Dict[str, Any]:
        logger.info("Safety Agent running")
        warnings = []
        if state.severity == "severe":
            warnings.append("⚠️ This appears to be a severe condition. Please consult a dermatologist.")
        if state.requires_doctor:
            warnings.append("A professional consultation is recommended for your concern.")

        # Assemble final response
        final = ""
        if state.diagnosis:
            final += state.diagnosis
        if state.recommendation_text:
            if final:
                final += "\n\n---\n\n"
            final += state.recommendation_text
        if state.conversational_reply and not final:
            final = state.conversational_reply

        return {
            "safety_checks_passed": True,
            "safety_warnings": warnings,
            "final_response": final,
            "current_agent": None,
            "agent_history": state.agent_history + ["safety"],
        }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class MultiAgentOrchestrator:
    def __init__(self):
        self.agents = {
            "triage": TriageAgent(),
            "vision": VisionAgent(),
            "diagnosis": DiagnosisAgent(),
            "conversational": ConversationalAgent(),
            "search": ProductSearchAgent(),
            "recommendation": RecommendationAgent(),
            "doctor_expert": DoctorExpertAgent(),
            "safety": SafetyAgent(),
        }

    async def run(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict]] = None,
        image_data: Optional[str] = None,
        image_type: Optional[str] = None,
        stage: str = "full",
        prefill: Optional[Dict] = None,
    ) -> AgentState:
        t0 = datetime.utcnow()
        state = AgentState(
            user_message=user_message,
            conversation_history=conversation_history or [],
            image_data=image_data,
            image_type=image_type,
            stage=stage,
        )
        # Pre-populate state for stage2 (skips vision/diagnosis/triage)
        if prefill:
            for key, value in prefill.items():
                if hasattr(state, key):
                    setattr(state, key, value)

        max_steps = 10
        for step in range(max_steps):
            if not state.current_agent:
                break
            agent = self.agents.get(state.current_agent)
            if not agent:
                logger.warning(f"Unknown agent: {state.current_agent}")
                break

            logger.info(f"[Step {step+1}] Running agent: {state.current_agent}")
            result = await agent.process(state)

            for key, value in result.items():
                if hasattr(state, key):
                    setattr(state, key, value)

        state.total_latency_ms = (datetime.utcnow() - t0).total_seconds() * 1000
        path = " → ".join(state.agent_history)
        logger.info(f"Pipeline done: {path} | {state.total_latency_ms:.0f}ms")
        return state


# Singleton
_orchestrator: Optional[MultiAgentOrchestrator] = None


async def get_orchestrator() -> MultiAgentOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = MultiAgentOrchestrator()
    return _orchestrator
