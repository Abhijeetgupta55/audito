import logging
from typing import Optional
from supabase import create_client, Client

from backend.config import settings

logger = logging.getLogger(__name__)

_supabase: Optional[Client] = None

def get_supabase() -> Optional[Client]:
    """Return Supabase client if configured, otherwise None."""
    global _supabase
    if _supabase is None:
        if settings.SUPABASE_URL and settings.SUPABASE_KEY:
            try:
                _supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
                logger.info("Supabase client initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize Supabase: {e}")
                _supabase = None
        else:
            logger.info("Supabase URL or KEY not provided, using local JSON store.")
            
    return _supabase
