"""
Progress Store — persists skin/hair analysis results per user,
computes trends, lighting-consistency flags, and AI insight summaries.
"""
from __future__ import annotations

import json
import logging
import math
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Storage location ──────────────────────────────────────────────────────────
DATA_DIR = Path(os.getenv("AUDITO_DATA_DIR", "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _user_file(user_id: str) -> Path:
    safe = "".join(c for c in user_id if c.isalnum() or c in "-_")[:64]
    return DATA_DIR / f"{safe}.json"


# ── Metric keys exposed by Vision Agent ──────────────────────────────────────
METRIC_KEYS = [
    "acne_severity",
    "redness",
    "pigmentation",
    "hydration",
    "texture",
    "dark_circles",
    "wrinkles",
    "hair_thinning",
    "scalp_condition",
    "confidence_score",
]

# Friendly labels for UI
METRIC_LABELS: Dict[str, str] = {
    "acne_severity":  "Acne",
    "redness":        "Redness",
    "pigmentation":   "Pigmentation",
    "hydration":      "Hydration",
    "texture":        "Texture",
    "dark_circles":   "Dark Circles",
    "wrinkles":       "Wrinkles",
    "hair_thinning":  "Hair Thinning",
    "scalp_condition":"Scalp",
    "confidence_score":"Confidence",
}

# For these metrics LOWER is better (e.g. "acne severity = 8 is bad")
LOWER_IS_BETTER = {
    "acne_severity", "redness", "pigmentation",
    "dark_circles", "wrinkles", "hair_thinning",
    "hydration", "texture", "scalp_condition"
}


from backend.supabase_client import get_supabase

def _load(user_id: str) -> List[Dict]:
    client = get_supabase()
    if client:
        try:
            res = client.table('progress_history').select('data').eq('user_id', user_id).execute()
            if res.data:
                return res.data[0].get('data', [])
            return []
        except Exception as e:
            logger.error(f"Supabase load error: {e}")
            # fallback to local
            
    fp = _user_file(user_id)
    if not fp.exists():
        return []
    try:
        return json.loads(fp.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"Progress load error for {user_id}: {e}")
        return []


def _save(user_id: str, records: List[Dict]) -> None:
    client = get_supabase()
    if client:
        try:
            # upsert record
            client.table('progress_history').upsert({
                'user_id': user_id,
                'data': records,
                'updated_at': datetime.utcnow().isoformat()
            }).execute()
        except Exception as e:
            logger.error(f"Supabase save error: {e}")
            
    fp = _user_file(user_id)
    fp.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")


def _delta(new_val: float, old_val: float, metric: str) -> Dict:
    diff = new_val - old_val
    if metric in LOWER_IS_BETTER:
        improved = diff < -0.5
        worsened = diff > 0.5
    else:
        improved = diff > 0.5
        worsened = diff < -0.5
    return {
        "diff": round(diff, 2),
        "improved": improved,
        "worsened": worsened,
        "neutral": not improved and not worsened,
    }


def _lighting_divergence(a: Dict, b: Dict) -> float:
    """
    Rough lighting-consistency check.
    Uses 'brightness' and 'contrast' if recorded, else 'confidence_score' delta.
    Returns 0-1 where >0.4 = suspect.
    """
    c1 = a.get("metrics", {}).get("confidence_score", 0.8)
    c2 = b.get("metrics", {}).get("confidence_score", 0.8)
    # Also check angle consistency via skin_type change
    same_skin = a.get("skin_type") == b.get("skin_type")
    conf_diff = abs(c1 - c2)
    base = conf_diff * 0.6
    if not same_skin:
        base += 0.25
    return min(base, 1.0)


def _natural_language_delta(metric: str, d: Dict) -> Optional[str]:
    label = METRIC_LABELS.get(metric, metric)
    diff = d["diff"]
    if d["improved"]:
        return f"{label} appears **improved** since your previous upload."
    elif d["worsened"]:
        return f"{label} appears **higher** than your previous upload — worth monitoring."
    return None


# ── Public API ─────────────────────────────────────────────────────────────────

def _is_meaningful(metrics: Dict[str, Any]) -> bool:
    """Return True only if the vision agent produced a real analysis (not a fallback/fake)."""
    conf = metrics.get("confidence_score", 0)
    if conf < 0.1:
        return False
    meaningful_keys = ["acne_severity", "redness", "pigmentation", "wrinkles",
                       "hair_thinning", "hydration", "texture", "dark_circles", "scalp_condition"]
    total = sum(float(metrics.get(k, 0)) for k in meaningful_keys)
    return total > 0


def save_analysis(user_id: str, metrics: Dict[str, Any], skin_analysis: Dict) -> Optional[str]:
    """
    Persist one analysis snapshot. Returns the generated record_id, or None if
    the metrics are not meaningful (e.g. vision fallback / fake image).
    """
    if not _is_meaningful(metrics):
        logger.info(f"Skipping save for {user_id} — metrics are empty/meaningless (confidence={metrics.get('confidence_score',0)})")
        return None

    records = _load(user_id)
    record_id = str(uuid.uuid4())
    record = {
        "record_id": record_id,
        "timestamp": datetime.utcnow().isoformat(),
        "metrics": {k: metrics.get(k, 0.0) for k in METRIC_KEYS},
        "skin_type": skin_analysis.get("skin_type", "unknown"),
        "conditions": skin_analysis.get("conditions", []),
        "primary_concern": skin_analysis.get("primary_concern", ""),
        "severity": skin_analysis.get("severity", "mild"),
        "clinical_observation": skin_analysis.get("clinical_observation", ""),
    }
    records.append(record)
    _save(user_id, records)
    logger.info(f"Saved analysis {record_id} for user {user_id} (confidence={metrics.get('confidence_score',0):.2f})")
    return record_id


def get_history(user_id: str) -> List[Dict]:
    return _load(user_id)


def get_progress_report(user_id: str) -> Dict:
    """
    Returns:
    - all records (timeline)
    - trend data per metric (list of {date, value} for charting)
    - comparison vs last record
    - lighting_warning if images diverge significantly
    - AI insight summary (natural language)
    """
    records = _load(user_id)
    if not records:
        return {
            "records": [],
            "trends": {},
            "comparison": None,
            "lighting_warning": False,
            "insight_summary": [],
            "streak": 0,
        }

    # ── Trends ────────────────────────────────────────────────────────────────
    trends: Dict[str, List[Dict]] = {k: [] for k in METRIC_KEYS}
    for r in records:
        ts = r["timestamp"][:10]  # YYYY-MM-DD
        for k in METRIC_KEYS:
            val = r["metrics"].get(k)
            if val is not None:
                trends[k].append({"date": ts, "value": round(float(val), 2)})

    # ── Comparison with previous ───────────────────────────────────────────────
    # Only compare meaningful records (confidence > 0, real image analysis)
    meaningful = [r for r in records if _is_meaningful(r.get("metrics", {}))]

    comparison = None
    lighting_warning = False
    if len(meaningful) >= 2:
        latest = meaningful[-1]
        previous = meaningful[-2]
        deltas = {}
        for k in METRIC_KEYS:
            nv = latest["metrics"].get(k, 0.0)
            ov = previous["metrics"].get(k, 0.0)
            deltas[k] = _delta(nv, ov, k)

        div = _lighting_divergence(latest, previous)
        lighting_warning = div > 0.4

        comparison = {
            "latest_id": latest["record_id"],
            "previous_id": previous["record_id"],
            "latest_date": latest["timestamp"][:10],
            "previous_date": previous["timestamp"][:10],
            "deltas": deltas,
            "lighting_divergence": round(div, 3),
        }

    # ── Natural language insight summary ──────────────────────────────────────
    insight_summary: List[str] = []
    if comparison:
        for k, d in comparison["deltas"].items():
            msg = _natural_language_delta(k, d)
            if msg:
                insight_summary.append(msg)

    # Recurring concern check — only across meaningful records
    if len(meaningful) >= 3:
        all_concerns = [r.get("primary_concern", "") for r in meaningful[-5:]]
        from collections import Counter
        top = Counter(all_concerns).most_common(1)
        if top and top[0][1] >= 3 and top[0][0]:
            insight_summary.append(
                f"**{top[0][0].replace('_', ' ').title()}** has been a recurring concern "
                f"across {top[0][1]} of your recent uploads."
            )

    # Improvement pattern
    if len(records) >= 3:
        for k in ["acne_severity", "redness", "hair_thinning"]:
            vals = [r["metrics"].get(k, 0.0) for r in records[-3:]]
            if all(b < a for a, b in zip(vals, vals[1:])):
                label = METRIC_LABELS[k]
                insight_summary.append(
                    f"Great news — **{label}** has been consistently improving over your last 3 uploads! 🎉"
                )

    # Upload streak (consecutive days)
    streak = 0
    if records:
        from datetime import date, timedelta
        today = date.today()
        for r in reversed(records):
            try:
                d = date.fromisoformat(r["timestamp"][:10])
                if (today - d).days <= streak + 1:
                    streak += 1
                    today = d
                else:
                    break
            except Exception:
                break

    return {
        "records": records,
        "trends": trends,
        "comparison": comparison,
        "lighting_warning": lighting_warning,
        "insight_summary": insight_summary,
        "streak": streak,
    }


def delete_history(user_id: str) -> None:
    fp = _user_file(user_id)
    if fp.exists():
        fp.unlink()
        logger.info(f"Deleted history for {user_id}")
