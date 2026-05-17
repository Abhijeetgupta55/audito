"""Application settings for Audito."""
from __future__ import annotations
import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

# Always load from the .env file next to this project root, overriding any stale env vars
_ENV_FILE = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_FILE, override=True)


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    return default if v is None else v.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    API_TITLE: str = os.getenv("API_TITLE", "Audito API")
    API_VERSION: str = os.getenv("API_VERSION", "1.0.0")
    DEBUG: bool = _env_bool("DEBUG", True)

    # Gemini API (via google-generativeai SDK)
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
    GEMINI_VISION_MODEL: str = os.getenv("GEMINI_VISION_MODEL", "gemini-3.1-flash-lite")

    # These are kept for backwards-compat with any lingering references
    OPENAI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-004")

    # Product search
    TOP_K_PRODUCTS: int = _env_int("TOP_K_PRODUCTS", 5)
    MIN_SIMILARITY_SCORE: float = _env_float("MIN_SIMILARITY_SCORE", 0.0)

    # Pinecone (optional, falls back to local DB)
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "audito-products")
    PINECONE_NAMESPACE: str = os.getenv("PINECONE_NAMESPACE", "products")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")

    # Supabase
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")

    # CORS — comma-separated list of allowed origins, e.g. "https://audito.vercel.app,https://www.audito.app"
    # Leave blank to allow all origins (not recommended for production)
    ALLOWED_ORIGINS: str = os.getenv("ALLOWED_ORIGINS", "")

    # ChromaDB persistence path — override to a writable absolute path on hosted envs
    # e.g. /tmp/chroma_db on Render free tier (ephemeral) or a mounted persistent disk
    CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "")

    # Monitoring (optional)
    LANGFUSE_ENABLED: bool = _env_bool("LANGFUSE_ENABLED", False)
    LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "")


settings = Settings()
