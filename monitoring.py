"""Optional Langfuse integration for Audito."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def setup_langfuse(public_key: str, secret_key: str) -> None:
    """Initialize Langfuse if the dependency is available."""

    if not public_key or not secret_key:
        logger.info("Langfuse skipped because credentials are not configured")
        return

    try:
        from langfuse import Langfuse
    except Exception as exc:
        logger.warning("Langfuse integration unavailable: %s", exc)
        return

    Langfuse(public_key=public_key, secret_key=secret_key)
    logger.info("Langfuse initialized")
