"""Local product store using keyword-based search against products_db.json.
Designed to be swapped for Pinecone when ready.
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Concern keyword map: maps query words → concern tags in the DB
CONCERN_KEYWORDS: Dict[str, List[str]] = {
    "acne": [
        "acne", "pimple", "pimples", "breakout", "zit", "blemish",
        "blackhead", "whitehead", "comedone", "clogged pore", "spots on face"
    ],
    "oily_skin": [
        "oily", "greasy", "shiny", "excess oil", "sebum", "oily skin",
        "large pores", "open pores", "enlarged pores"
    ],
    "dry_skin": [
        "dry", "flaky", "tight", "dehydrated", "rough", "scaly",
        "dry skin", "peeling", "cracked skin", "moisture"
    ],
    "sensitive_skin": [
        "sensitive", "irritated", "redness", "red", "inflamed", "inflammation",
        "reactive", "rash", "itching", "burning", "stinging", "allergy"
    ],
    "dark_spots": [
        "dark spot", "dark spots", "hyperpigmentation", "uneven tone",
        "discoloration", "melasma", "tan", "patch", "mark", "acne scar",
        "post acne", "uneven skin"
    ],
    "anti_aging": [
        "wrinkle", "wrinkles", "fine line", "fine lines", "aging", "age",
        "sagging", "elasticity", "firmness", "crow's feet", "anti aging"
    ],
    "hair_loss": [
        "hair loss", "hair fall", "thinning hair", "thinning", "shedding",
        "bald", "receding", "alopecia", "hair thinning", "losing hair"
    ],
    "dandruff": [
        "dandruff", "flaking", "scalp flakes", "dry scalp", "itchy scalp",
        "scalp itch", "flaky scalp", "seborrheic"
    ],
    "dull_skin": [
        "dull", "tired skin", "lifeless", "radiance", "glow", "brightening",
        "luminous", "dark complexion", "no glow"
    ],
    "large_pores": [
        "large pores", "open pores", "enlarged pores", "pores"
    ],
}

# Load product DB once at module level
_DB_PATH = Path(__file__).parent.parent / "products_db.json"
_PRODUCTS: List[Dict[str, Any]] = []

def _load_products() -> List[Dict[str, Any]]:
    global _PRODUCTS
    if _PRODUCTS:
        return _PRODUCTS
    try:
        with open(_DB_PATH, "r", encoding="utf-8") as f:
            _PRODUCTS = json.load(f)
        logger.info(f"Loaded {len(_PRODUCTS)} products from local DB")
    except Exception as e:
        logger.error(f"Failed to load products_db.json: {e}")
        _PRODUCTS = []
    return _PRODUCTS


def _detect_concerns(query: str) -> List[str]:
    """Map a query string to concern tags."""
    q = query.lower()
    matched: Dict[str, int] = {}
    for concern, keywords in CONCERN_KEYWORDS.items():
        for kw in keywords:
            if kw in q:
                matched[concern] = matched.get(concern, 0) + 1
    # Sort by match count desc
    return [c for c, _ in sorted(matched.items(), key=lambda x: -x[1])]


def _score_product(product: Dict[str, Any], concerns: List[str]) -> float:
    """Score a product based on how many detected concerns it addresses."""
    if not concerns:
        return 0.0
    product_concerns = product.get("concerns", [])
    hits = sum(1 for c in concerns if c in product_concerns)
    return hits / len(concerns)


class LocalProductStore:
    """Keyword-based local product search. Drop-in replacement for Pinecone."""

    def __init__(self):
        self.products = _load_products()
        self.enabled = len(self.products) > 0
        logger.info(f"LocalProductStore ready. {len(self.products)} products.")

    def search_products(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Search products by concern keyword matching."""
        concerns = _detect_concerns(query)
        logger.info(f"Query '{query[:60]}' → concerns: {concerns}")

        if not concerns:
            # No specific concern detected — return empty (no unsolicited recommendations)
            logger.info("No concern keywords found in query — returning empty product list")
            return []

        scored = []
        for p in self.products:
            score = _score_product(p, concerns)
            if score > min_score:
                scored.append({**p, "similarity_score": round(score, 2)})

        scored.sort(key=lambda x: -x["similarity_score"])

        # If we got fewer than top_k relevant products, pad with general ones
        result = scored[:top_k]
        if len(result) < 2:
            seen_ids = {r["id"] for r in result}
            for p in self.products:
                if p["id"] not in seen_ids and len(result) >= top_k:
                    break
                if p["id"] not in seen_ids:
                    result.append({**p, "similarity_score": 0.1})

        return result[:top_k]

    def get_all(self) -> List[Dict[str, Any]]:
        return self.products


# Singleton
_store: Optional[LocalProductStore] = None


async def get_pinecone_store() -> "LocalProductStore":
    """Backwards-compatible async getter — returns LocalProductStore."""
    global _store
    if _store is None:
        _store = LocalProductStore()
    return _store
