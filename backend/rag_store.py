"""RAG Pipeline and Vector Store using LangChain and ChromaDB.

Uses a custom GeminiEmbeddings wrapper around the google-genai SDK so that
the embedding model (text-embedding-004) is accessed via the v1 API, not the
deprecated v1beta path used by langchain-google-genai.
"""
import asyncio
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from backend.config import settings

logger = logging.getLogger(__name__)

# ── Optional RAG dependencies ──────────────────────────────────────────────────
try:
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    _DEPS_OK = True
except ImportError as _import_err:
    logger.warning(f"RAG dependencies unavailable ({_import_err}). Falling back to keyword search.")
    _DEPS_OK = False

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
# CHROMA_DB_PATH env var lets hosted environments (e.g. Render) redirect to /tmp or a persistent disk.
# Falls back to chroma_db/ next to the project root when not set.
_chroma_override = settings.CHROMA_DB_PATH
DB_DIR = Path(_chroma_override) if _chroma_override else BASE_DIR / "chroma_db"
PRODUCTS_PATH = BASE_DIR / "products_db.json"
KB_PATH = Path(__file__).parent / "knowledge_base" / "dermatology_kb.md"


# ── Custom embedding wrapper (uses google-genai SDK, not deprecated generativeai) ──
class GeminiEmbeddings(Embeddings):
    """LangChain-compatible embeddings backed by the google-genai SDK.

    The langchain-google-genai package routes embed calls through the v1beta API
    which does not support text-embedding-004. This wrapper calls the v1 endpoint
    directly via google.genai, matching what agents.py already uses for generation.
    """

    def __init__(self, model: str, api_key: str):
        from google import genai as _genai
        self._client = _genai.Client(api_key=api_key)
        self._model = model

    def _embed(self, text: str) -> List[float]:
        result = self._client.models.embed_content(
            model=self._model,
            contents=text,
        )
        return list(result.embeddings[0].values)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


# ── RagStore ───────────────────────────────────────────────────────────────────

class RagStore:
    def __init__(self):
        if not _DEPS_OK:
            raise RuntimeError("RAG dependencies (chromadb / langchain) not available")

        self.embeddings = GeminiEmbeddings(
            model="models/gemini-embedding-001",
            api_key=settings.GEMINI_API_KEY,
        )

        self.kb_vectorstore = Chroma(
            collection_name="knowledge_base",
            embedding_function=self.embeddings,
            persist_directory=str(DB_DIR),
        )

        self.product_vectorstore = Chroma(
            collection_name="products",
            embedding_function=self.embeddings,
            persist_directory=str(DB_DIR),
        )

        self._ensure_initialized()

    def _ensure_initialized(self):
        """Populate ChromaDB collections on first run (idempotent)."""
        kb_count = self.kb_vectorstore._collection.count()
        if kb_count == 0:
            logger.info("Initializing Knowledge Base in ChromaDB…")
            if KB_PATH.exists():
                loader = TextLoader(str(KB_PATH), encoding="utf-8")
                docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50,
                    separators=["\n## ", "\n### ", "\n- ", "\n\n", "\n", " ", ""],
                )
                splits = splitter.split_documents(docs)
                self.kb_vectorstore.add_documents(splits)
                logger.info(f"Added {len(splits)} knowledge chunks to ChromaDB.")
            else:
                logger.warning(f"KB file not found at {KB_PATH}")

        prod_count = self.product_vectorstore._collection.count()
        if prod_count == 0:
            logger.info("Initializing Products in ChromaDB…")
            if PRODUCTS_PATH.exists():
                with open(PRODUCTS_PATH, "r", encoding="utf-8") as f:
                    products = json.load(f)

                prod_docs = []
                for p in products:
                    content = (
                        f"Product: {p.get('name', '')}\n"
                        f"Brand: {p.get('brand', '')}\n"
                        f"Description: {p.get('description', '')}\n"
                        f"Ingredients: {', '.join(p.get('key_ingredients', []))}\n"
                        f"Concerns Addressed: {', '.join(p.get('concerns', []))}\n"
                        f"Skin Types: {', '.join(p.get('suitable_for', []))}\n"
                        f"Usage: {p.get('how_to_use', '')}"
                    )
                    metadata = {
                        "id": p.get("id", ""),
                        "name": p.get("name", ""),
                        "brand": p.get("brand", ""),
                        "json_data": json.dumps(p),
                    }
                    prod_docs.append(Document(page_content=content, metadata=metadata))

                self.product_vectorstore.add_documents(prod_docs)
                logger.info(f"Added {len(prod_docs)} products to ChromaDB.")
            else:
                logger.warning(f"Products DB not found at {PRODUCTS_PATH}")

    def search_knowledge(self, query: str, top_k: int = 3) -> str:
        """Return concatenated KB chunks most relevant to the query."""
        docs = self.kb_vectorstore.similarity_search(query, k=top_k)
        if not docs:
            return ""
        return "\n\n".join(f"[{i+1}] {doc.page_content}" for i, doc in enumerate(docs))

    def search_products(self, query: str, top_k: int = 5, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Semantic product search. Returns products sorted by relevance (0-1)."""
        docs_and_scores = self.product_vectorstore.similarity_search_with_score(query, k=top_k)
        results = []
        for doc, score in docs_and_scores:
            p_data = json.loads(doc.metadata["json_data"])
            # L2 distance → 0-1 relevance (higher = more relevant)
            p_data["similarity_score"] = round(1.0 / (1.0 + float(score)), 3)
            results.append(p_data)
        results.sort(key=lambda x: -x["similarity_score"])
        return results


# ── Singleton ──────────────────────────────────────────────────────────────────
_store: Optional[RagStore] = None


async def get_rag_store() -> Optional[RagStore]:
    """Return the RagStore singleton. Init runs in a thread to avoid blocking the event loop."""
    global _store
    if not _DEPS_OK:
        return None
    if _store is None:
        try:
            _store = await asyncio.to_thread(RagStore)
        except Exception as e:
            logger.error(f"RagStore init failed: {e}. RAG unavailable for this session.")
            return None
    return _store
