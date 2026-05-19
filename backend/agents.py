"""Multi-agent pipeline for Audito skin & hair diagnostic assistant.

Agent routing:
  triage → (chit-chat) → conversational  [END — no products]
  triage → (concern) → diagnosis
  triage → (image) → vision → diagnosis
  diagnosis → (mild/moderate, no Rx needed) → search → recommendation → safety [END]
  diagnosis → (severe / Rx needed) → doctor_expert → safety [END]
"""
import asyncio
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field

from google import genai
from google.genai import types as genai_types

from backend.config import settings

logger = logging.getLogger(__name__)

# When we hit 429, set a cooldown so retries don't burn more quota uselessly.
# Daily-quota 429s won't recover within seconds — give up fast on the current request
# instead of hammering the API with retries.
_quota_cooldown_until: float = 0.0

# ---------------------------------------------------------------------------
# Configure Gemini once
# ---------------------------------------------------------------------------

_client: Optional["genai.Client"] = None


def _ensure_gemini() -> bool:
    global _client
    if _client is None:
        if not settings.GEMINI_API_KEY:
            logger.error("GEMINI_API_KEY is not set — all LLM calls will be skipped. Set this env var on Render.")
            return False
        _client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _client is not None


_SAFETY_OFF = [
    genai_types.SafetySetting(category="HARM_CATEGORY_HARASSMENT",        threshold="BLOCK_NONE"),
    genai_types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH",       threshold="BLOCK_NONE"),
    genai_types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
    genai_types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
]


def _extract_response(response, model_name: str) -> Tuple[str, str, str]:
    """
    Parse a Gemini response into (text, finish_reason, safety_info).
    Uses multiple extraction strategies in order of reliability:
      1. response.text shortcut (canonical SDK method — handles all cases)
      2. Walk candidates[0].content.parts (manual fallback)
      3. response.parsed (for structured output)
    Comprehensive logging on empty responses so we can debug what Gemini returned.
    """
    text = ""
    finish_reason = "UNKNOWN"
    safety_info = ""

    try:
        # Prompt-level block (fires before any candidate is generated)
        prompt_feedback = getattr(response, "prompt_feedback", None)
        if prompt_feedback:
            block_reason = getattr(prompt_feedback, "block_reason", None)
            if block_reason and str(block_reason) not in ("None", "BLOCK_REASON_UNSPECIFIED", "0"):
                logger.warning(f"[{model_name}] Prompt blocked — block_reason={block_reason}")
                return "", f"PROMPT_BLOCKED:{block_reason}", ""

        # Strategy 1: response.text — the canonical SDK shortcut.
        # This is what every basic wrapper uses; it handles parts walking internally.
        try:
            direct = getattr(response, "text", None)
            if direct and isinstance(direct, str) and direct.strip():
                text = direct
        except Exception as e:
            logger.debug(f"[{model_name}] response.text raised: {e}")

        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            logger.error(
                f"[{model_name}] NO_CANDIDATES — response has zero candidates. "
                f"Full response: {repr(response)[:800]}"
            )
            return text, "NO_CANDIDATES", ""

        candidate = candidates[0]

        # Finish reason
        raw_fr = getattr(candidate, "finish_reason", None)
        finish_reason = str(raw_fr) if raw_fr is not None else "None"

        # Safety ratings
        safety_ratings = getattr(candidate, "safety_ratings", None) or []
        if safety_ratings:
            parts_sr = []
            for r in safety_ratings:
                cat = getattr(r, "category", "?")
                cat_str = cat.name if hasattr(cat, "name") else str(cat)
                prob = getattr(r, "probability", "?")
                prob_str = prob.name if hasattr(prob, "name") else str(prob)
                parts_sr.append(f"{cat_str}={prob_str}")
            safety_info = " | ".join(parts_sr)

        # Log unexpected finish reasons
        stop_values = {"FinishReason.STOP", "STOP", "1", "None", "FINISH_REASON_STOP"}
        if finish_reason not in stop_values:
            logger.warning(
                f"[{model_name}] Non-STOP finish: finish_reason={finish_reason} "
                f"safety=[{safety_info}]"
            )

        # Strategy 2: walk parts manually if response.text didn't give us anything
        if not text:
            content = getattr(candidate, "content", None)
            if content:
                for part in (getattr(content, "parts", None) or []):
                    if getattr(part, "thought", False):
                        continue
                    part_text = getattr(part, "text", None)
                    if part_text:
                        text += part_text

        # Token usage
        usage = getattr(response, "usage_metadata", None)
        if usage:
            in_tok = getattr(usage, "prompt_token_count", "?")
            out_tok = getattr(usage, "candidates_token_count", "?")
            thought_tok = getattr(usage, "thoughts_token_count", None) or 0
            logger.info(
                f"[{model_name}] tokens in={in_tok} out={out_tok} thought={thought_tok}"
            )

        # If still empty after both strategies, dump the candidate for diagnosis
        if not text:
            logger.error(
                f"[{model_name}] EXTRACTED EMPTY TEXT after both strategies. "
                f"finish={finish_reason} safety=[{safety_info}] "
                f"candidate={repr(candidate)[:600]}"
            )

    except Exception as exc:
        logger.error(f"_extract_response [{model_name}]: parse error — {exc}")

    return text, finish_reason, safety_info


def _generate(prompt_parts: list, model_name: str = None, max_tokens: int = None) -> str:
    """
    Single synchronous Gemini call. Never raises, never retries.
    For retry logic use _agenerate() (async).

    Respects a cooldown after recent 429s — when daily quota is exhausted, retrying
    immediately burns more requests for nothing. The cooldown lets the pipeline
    fail fast and fall back to templates instead of waiting on doomed retries.
    """
    global _quota_cooldown_until  # declared once for the whole function
    if time.time() < _quota_cooldown_until:
        remaining = _quota_cooldown_until - time.time()
        logger.warning(f"_generate: quota cooldown active ({remaining:.0f}s remaining) — skipping call")
        return ""

    if not _ensure_gemini():
        logger.error("_generate: Gemini client not ready — GEMINI_API_KEY missing")
        return ""

    name = model_name or settings.GEMINI_MODEL
    preview = str(prompt_parts[0])[:100] if prompt_parts else ""
    logger.info(f"_generate → {name} | {preview!r}")

    try:
        cfg_kwargs: Dict[str, Any] = {
            "safety_settings": _SAFETY_OFF,
        }
        # Only set max_output_tokens if a real value is provided.
        # Passing None to GenerateContentConfig can cause SDK validation issues
        # in newer google-genai versions.
        if max_tokens is not None and max_tokens > 0:
            cfg_kwargs["max_output_tokens"] = max_tokens

        cfg = genai_types.GenerateContentConfig(**cfg_kwargs)
        response = _client.models.generate_content(
            model=name,
            contents=prompt_parts,
            config=cfg,
        )

        text, finish_reason, safety_info = _extract_response(response, name)

        if text:
            logger.info(f"_generate ← {len(text)} chars | finish={finish_reason}")
        else:
            logger.warning(
                f"_generate: EMPTY response | model={name} finish={finish_reason} "
                f"safety=[{safety_info}]"
            )
        return text

    except Exception as e:
        err = str(e)
        # Log the FULL exception type and message so we can see what's actually failing
        logger.error(
            f"_generate: exception caught | model={name} | "
            f"type={type(e).__name__} | message={err[:500]}"
        )
        if "429" in err or "RESOURCE_EXHAUSTED" in err or "quota" in err.lower():
            # Set a 60-second cooldown so the retry loop in _agenerate and any
            # parallel agent calls don't immediately burn more requests on a
            # known-exhausted quota.
            _quota_cooldown_until = time.time() + 60
            logger.warning(f"_generate: 429/quota hit — set 60s cooldown to stop retry storm")
        elif "404" in err or "not found" in err.lower():
            logger.error(f"_generate: 404 model '{name}' not found — verify model name")
        elif "403" in err or "PERMISSION" in err or "API key" in err.lower():
            logger.error(f"_generate: auth error — verify GEMINI_API_KEY is valid")
        elif "INVALID_ARGUMENT" in err or "400" in err:
            logger.error(f"_generate: bad request — check prompt/config format")
        return ""


async def _agenerate(
    prompt_parts: list,
    model_name: str = None,
    max_tokens: int = None,
    retries: int = 1,
) -> str:
    """
    Async wrapper around _generate with non-blocking retry on transient empty.
    retries=0 → single attempt (use for fast conversational paths).
    retries=1 → one retry after 1.5s (default).
    retries=2 → two retries for critical paths (vision).
    """
    for attempt in range(retries + 1):
        text = _generate(prompt_parts, model_name, max_tokens)
        if text:
            return text
        if attempt < retries:
            wait = 1.5 * (attempt + 1)
            logger.warning(
                f"_agenerate: empty on attempt {attempt + 1}/{retries + 1} — "
                f"retrying in {wait:.1f}s"
            )
            await asyncio.sleep(wait)

    name = model_name or settings.GEMINI_MODEL
    logger.error(
        f"_agenerate: all {retries + 1} attempt(s) returned empty "
        f"[model={name}] — check Render logs for finish_reason"
    )
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

        # Exact-match greetings — skip LLM for messages that are LITERALLY just
        # "hi", "hello", "thanks", etc. with no other content. Anything ambiguous
        # ("hi I have acne") still goes through the LLM for proper classification.
        # Triage is just internal routing logic; the user-facing response is still
        # fully generated by the Conversational agent's LLM call.
        msg_normalized = state.user_message.lower().strip().rstrip("!.?,")
        is_pure_greeting = msg_normalized in {p.lower() for p in CHIT_CHAT_PATTERNS}

        if is_pure_greeting:
            logger.info(f"Triage: pure greeting '{msg_normalized}' — keyword routing, 0 LLM calls")
            result = {"intent": "chit_chat", "concern": "none", "severity": "mild", "requires_doctor": False}
        else:
            result = None
            if _ensure_gemini():
                try:
                    text = await _agenerate(
                        [self.SYSTEM, f"\nUser message: {state.user_message}"],
                        max_tokens=200,
                        retries=1,
                    )
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
            text = await _agenerate(
                [self.SYSTEM, img_part],
                model_name=settings.GEMINI_VISION_MODEL,
                max_tokens=800,
                retries=2,
            )
            logger.info(f"Vision raw response: {len(text)} chars — preview: {text[:120]!r}")

            # Step 1: check if model returned anything at all
            if not text:
                logger.error("Vision: _generate returned empty — model/API failure")
                return {
                    "skin_analysis": {"is_clear": False, "metrics": {}},
                    "vision_feedback": "The AI service is currently unavailable (Gemini API quota or connection issue). Please try again later.",
                    "identified_concern": "vision_error",
                    "current_agent": "conversational",
                    "agent_history": state.agent_history + ["vision"],
                }

            # Step 2: parse JSON
            analysis = _parse_json(text)
            logger.info(f"Vision parsed JSON keys: {list(analysis.keys()) if analysis else '(empty)'}")

            # Step 3: check if parsing yielded anything useful
            if not analysis:
                logger.error(f"Vision: JSON parse failed — raw response: {text[:200]!r}")
                return {
                    "skin_analysis": {"is_clear": False, "metrics": {}},
                    "vision_feedback": "The AI service returned an unexpected response format. This usually clears up on retry — please upload the image again.",
                    "identified_concern": "vision_error",
                    "current_agent": "conversational",
                    "agent_history": state.agent_history + ["vision"],
                }

            confidence = float((analysis.get("metrics") or {}).get("confidence_score", 0.5) or 0.5)
            logger.info(f"Vision analysis: is_clear={analysis.get('is_clear')} confidence={confidence:.2f}")

            # Determine whether the analysis has usable clinical content
            has_content = bool(
                analysis.get("clinical_observation") or
                (analysis.get("conditions") and len(analysis["conditions"]) > 0) or
                analysis.get("skin_type", "unknown") != "unknown"
            )

            if analysis.get("is_clear") is False:
                if has_content:
                    # Partial analysis exists despite is_clear=False — degrade gracefully
                    logger.warning(f"Vision: is_clear=False but has_content=True (conf={confidence:.2f}) — continuing with low_confidence")
                    analysis["low_confidence"] = True
                    if not analysis.get("clarity_feedback"):
                        analysis["clarity_feedback"] = "Photo quality is limited — results may be less accurate."
                    # Fall through to continue pipeline
                else:
                    # Genuinely unusable — no clinical content at all
                    feedback = analysis.get("clarity_feedback", "The photo was not clear enough for analysis. Please try in better lighting.")
                    logger.warning("Vision: is_clear=False AND no usable content — routing to conversational")
                    return {
                        "skin_analysis": analysis,
                        "vision_feedback": feedback,
                        "identified_concern": "unclear_image",
                        "current_agent": "conversational",
                        "agent_history": state.agent_history + ["vision"],
                    }
            elif confidence < 0.35 and has_content:
                # is_clear=True but low confidence — warn and continue
                logger.info(f"Vision: low confidence ({confidence:.2f}) — setting low_confidence flag")
                analysis["low_confidence"] = True
                if not analysis.get("clarity_feedback"):
                    analysis["clarity_feedback"] = "Analysis confidence is limited — consider retaking with better lighting."

            concern = analysis.get("primary_concern") or state.identified_concern or "general"
            metrics = analysis.get("metrics", {})
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

        # ── PHASE 1: Intake check skipped to save API calls ───────────────
        # Previously this was a separate LLM call to decide "do we have enough info?"
        # In practice it added latency, burned quota, and the differential prompt
        # itself can handle missing info (it falls back to product-search recommendations
        # when specifics are absent). The differential prompt is conservative enough
        # to phrase findings as "consistent with X" rather than confident diagnosis,
        # so running it directly is safe.
        logger.info("Intake check skipped (saves 1 API call) — running differential directly")

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
            raw_diff = (await _agenerate(
                [self.DIFFERENTIAL_SYSTEM, diff_prompt],
                max_tokens=350,
                retries=1,
            )).strip()
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

        # Vision pipeline routed here — distinguish model failure from actual unclear photo
        if state.vision_feedback:
            if state.identified_concern == "vision_error":
                # Model/API failure — photo was fine, backend couldn't process it
                reply = state.vision_feedback
            else:
                # Genuinely unclear/unusable photo — guide user to retake
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

        reply = (await _agenerate(
            [self.SYSTEM, context],
            max_tokens=300,
            retries=0,
        )).strip()

        if not reply:
            if not settings.GEMINI_API_KEY:
                logger.error("ConversationalAgent: GEMINI_API_KEY missing — cannot call LLM")
                reply = "The AI backend is not configured yet. The GEMINI_API_KEY environment variable needs to be set on the server."
            else:
                logger.warning("ConversationalAgent: Gemini returned empty (likely daily quota hit on Google AI Studio free tier — resets at midnight Pacific time)")
                reply = "The AI service has hit its daily free-tier quota and will reset within 24 hours. Please try again later — no other workaround until the quota window resets."

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
        if not state.identified_concern or state.identified_concern in ("none", "unclear_image", "vision_error", ""):
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
    # Concern-to-actives fallback — used when LLM/RAG returns empty actives
    ACTIVES_FALLBACK: Dict[str, List[Dict[str, str]]] = {
        "acne": [
            {"name": "Niacinamide", "mechanism": "reduce sebum + calm inflammation", "target_concern": "acne + oiliness"},
            {"name": "Salicylic Acid", "mechanism": "exfoliate + unclog pores", "target_concern": "blackheads + breakouts"},
            {"name": "Benzoyl Peroxide", "mechanism": "eliminate acne bacteria", "target_concern": "active breakouts"},
        ],
        "dry_skin": [
            {"name": "Hyaluronic Acid", "mechanism": "attract + retain moisture", "target_concern": "dehydration + tightness"},
            {"name": "Ceramides", "mechanism": "repair skin barrier", "target_concern": "dryness + flakiness"},
            {"name": "Glycerin", "mechanism": "humectant moisture binding", "target_concern": "dry + rough texture"},
        ],
        "oily_skin": [
            {"name": "Niacinamide", "mechanism": "regulate sebum production", "target_concern": "excess oil + enlarged pores"},
            {"name": "Salicylic Acid", "mechanism": "exfoliate + reduce oiliness", "target_concern": "shine + congestion"},
            {"name": "Zinc", "mechanism": "control oil secretion", "target_concern": "oiliness + breakouts"},
        ],
        "dark_spots": [
            {"name": "Vitamin C", "mechanism": "inhibit melanin synthesis", "target_concern": "hyperpigmentation + dullness"},
            {"name": "Alpha Arbutin", "mechanism": "block tyrosinase enzyme", "target_concern": "melasma + PIH"},
            {"name": "Niacinamide", "mechanism": "reduce pigment transfer", "target_concern": "uneven tone + dark spots"},
        ],
        "sensitive_skin": [
            {"name": "Centella Asiatica", "mechanism": "calm + repair skin barrier", "target_concern": "redness + sensitivity"},
            {"name": "Ceramides", "mechanism": "restore barrier function", "target_concern": "reactivity + irritation"},
            {"name": "Allantoin", "mechanism": "soothe + protect skin", "target_concern": "sensitivity + redness"},
        ],
        "anti_aging": [
            {"name": "Retinol", "mechanism": "boost cell turnover + collagen", "target_concern": "fine lines + texture"},
            {"name": "Vitamin C", "mechanism": "stimulate collagen synthesis", "target_concern": "wrinkles + firmness"},
            {"name": "Peptides", "mechanism": "signal collagen production", "target_concern": "sagging + elasticity"},
        ],
        "hair_loss": [
            {"name": "Minoxidil", "mechanism": "increase follicle blood flow", "target_concern": "thinning + shedding"},
            {"name": "Biotin", "mechanism": "support keratin production", "target_concern": "weak + thinning strands"},
            {"name": "Caffeine", "mechanism": "stimulate follicle growth phase", "target_concern": "hair density + loss"},
        ],
        "dandruff": [
            {"name": "Zinc Pyrithione", "mechanism": "inhibit Malassezia yeast", "target_concern": "dandruff + scalp flaking"},
            {"name": "Salicylic Acid", "mechanism": "dissolve scalp buildup", "target_concern": "flakes + itchy scalp"},
            {"name": "Ketoconazole", "mechanism": "antifungal scalp treatment", "target_concern": "seborrheic dermatitis"},
        ],
        "dull_skin": [
            {"name": "Vitamin C", "mechanism": "brighten + even skin tone", "target_concern": "dullness + radiance"},
            {"name": "AHA (Glycolic Acid)", "mechanism": "exfoliate dead surface cells", "target_concern": "dullness + texture"},
            {"name": "Niacinamide", "mechanism": "improve skin clarity + glow", "target_concern": "uneven tone + dullness"},
        ],
        "large_pores": [
            {"name": "Niacinamide", "mechanism": "tighten + minimise pore appearance", "target_concern": "enlarged pores + oiliness"},
            {"name": "Retinol", "mechanism": "increase cell turnover + firm skin", "target_concern": "pore size + texture"},
            {"name": "Salicylic Acid", "mechanism": "clear pore congestion", "target_concern": "blocked pores + blackheads"},
        ],
        "general_skincare": [
            {"name": "SPF Sunscreen", "mechanism": "block UV-induced damage", "target_concern": "photoaging + protection"},
            {"name": "Niacinamide", "mechanism": "multi-benefit barrier support", "target_concern": "general skin health"},
            {"name": "Hyaluronic Acid", "mechanism": "hydrate + plump skin", "target_concern": "hydration + texture"},
        ],
    }

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
                    logger.info(f"Stage1 LLM: using KB context ({len(state.kb_context)} chars)")
                else:
                    ctx += "Apply established clinical dermatology knowledge for this concern."
                    logger.info("Stage1 LLM: no KB context — using clinical training knowledge")
                raw = (await _agenerate(
                    [self.INGREDIENTS_SYSTEM, ctx],
                    max_tokens=350,
                    retries=1,
                )).strip()
                parsed = _parse_json(raw)
                actives = parsed.get("actives", []) if parsed else []
                if actives:
                    ingredient_rationale = "\n".join(
                        f"• {a['name']} — {a.get('mechanism', '')} — {a.get('target_concern', '')}"
                        for a in actives
                    )
                    logger.info(f"Stage1: LLM returned {len(actives)} actives")
                else:
                    logger.warning(f"Stage1: LLM returned empty actives (raw={raw[:80]!r}) — will use fallback")
            except Exception as e:
                logger.error(f"Stage1 ingredient rationale LLM failed: {e}")

        # Fallback: use curated concern-to-actives map when LLM returns nothing
        if not actives:
            concern_key = state.identified_concern or "general_skincare"
            fallback = self.ACTIVES_FALLBACK.get(concern_key) or self.ACTIVES_FALLBACK.get("general_skincare", [])
            if fallback:
                actives = fallback[:3]
                ingredient_rationale = "\n".join(
                    f"• {a['name']} — {a.get('mechanism', '')} — {a.get('target_concern', '')}"
                    for a in actives
                )
                logger.info(f"Stage1 fallback: using curated actives for '{concern_key}' ({len(actives)} items)")
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
                raw = (await _agenerate(
                    [self.PRODUCTS_SYSTEM, ctx],
                    max_tokens=350,
                    retries=1,
                )).strip()
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
                raw = (await _agenerate(
                    [self.SYSTEM, user_ctx],
                    max_tokens=450,
                    retries=1,
                )).strip()
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
                guidance = (await _agenerate(
                    [self.SYSTEM, ctx],
                    max_tokens=220,
                    retries=1,
                )).strip()
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
