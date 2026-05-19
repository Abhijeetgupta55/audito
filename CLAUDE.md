# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Audito** is an AI-powered skin and hair diagnostic assistant. It uses a multi-agent pipeline backed by Google Gemini (vision + text) to analyze skin photos, produce differential diagnoses, and recommend dermatology products. The backend is FastAPI (Python), the frontend is React + Vite, and the AI knowledge base uses ChromaDB with Gemini embeddings.

## Commands

### Backend

```powershell
# From a:\audito\audito\
.venv\Scripts\Activate.ps1
uvicorn main:app --reload --port 8000
```

```bash
# Health check and connectivity test
curl http://localhost:8000/health
curl http://localhost:8000/test-gemini
```

### Frontend

```bash
cd frontend
npm install
npm run dev       # Dev server (proxies /api/* → localhost:8000)
npm run build     # Production build → dist/
npm run lint      # ESLint
```

### Dependencies

```powershell
pip install -r requirements.txt
```

No test suite exists; manual verification uses the `/health` and `/test-gemini` endpoints.

## Architecture

### Multi-Agent Pipeline (`backend/agents.py`)

All agents share an `AgentState` dataclass that flows through the pipeline. Each agent writes its output to state fields and sets `state.current_agent` to the next agent name. The `MultiAgentOrchestrator` dispatches accordingly.

**Routing paths:**

| Input type | Agent sequence |
|---|---|
| Text concern | Triage → Diagnosis → ProductSearch → Recommendation → Safety |
| Image (Stage 1) | Triage → Vision → Diagnosis → ProductSearch → Recommendation |
| Image (Stage 2) | ProductSearch → Recommendation → Safety |
| Chit-chat / fallback | Triage → Conversational |
| Severe case | Triage → Diagnosis → DoctorExpert → Safety |

**Two-stage image flow:** Stage 1 (`/api/analyze-image`) returns photo analysis + active ingredients immediately and caches a product pre-fetch (300s TTL) under a `session_key`. The frontend then calls Stage 2 (`/api/recommend-products`) with that key to get the final product cards, giving the user fast initial feedback.

**Gemini calls:**
- `_generate()` — synchronous, no retry (used in non-critical paths)
- `_agenerate()` — async with exponential backoff (0–2 retries, scaled by agent urgency)
- `_extract_response()` — skips Gemini 2.5 thought tokens (`thought=True` parts), logs finish reason
- `thinking_budget=0` — extended thinking disabled on text models for latency; vision calls are unaffected
- `_SAFETY_OFF` — all Gemini safety filters disabled (medical domain requirement)

### RAG / Knowledge Base (`backend/rag_store.py`)

- `GeminiEmbeddings` — custom LangChain embeddings wrapper using `text-embedding-001` (v1 API)
- `RagStore` — manages two ChromaDB collections: `dermatology_kb` (chunked from `backend/knowledge_base/dermatology_kb.md`) and `products` (from `products_db.json`)
- Falls back to keyword search (`backend/vector_store.py`) if ChromaDB is unavailable

### API Surface (`main.py`)

| Endpoint | Purpose |
|---|---|
| `GET /health` | Service status + config |
| `POST /api/chat` | Full text pipeline |
| `POST /api/analyze-image` | Stage 1 — vision + diagnosis |
| `POST /api/recommend-products` | Stage 2 — products + safety |
| `GET /api/progress/{user_id}` | Per-user skin metric history |
| `GET /api/search` | Product keyword search |

### Frontend (`frontend/src/`)

- **`Consultation.jsx`** — main chat UI; manages message state, image uploads, and the Stage 1 → Stage 2 orchestration
- **`DiagnosticDeck.jsx`** — swipeable card stack (Framer Motion spring physics); `buildCards()` maps API response fields to typed cards: `photo`, `clinical_summary`, `actives`, `products`, `progress`, `warning`, `intake`
- `vite.config.js` proxies `/api/*` to `localhost:8000` in dev; `vercel.json` excludes `/api/` from the SPA rewrite in production

## Environment Variables

**Required:**
- `GEMINI_API_KEY` — Google AI Studio key

**Optional:**
- `GEMINI_MODEL` / `GEMINI_VISION_MODEL` — defaults to `gemini-2.5-flash`
- `CHROMA_DB_PATH` — vector DB location (ephemeral on Render free tier)
- `SUPABASE_URL` / `SUPABASE_KEY` — persistent progress storage (otherwise written to `data/` as JSON)
- `LANGFUSE_ENABLED` / `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` — LLM observability
- `ALLOWED_ORIGINS` — CORS whitelist (comma-separated; defaults to open)
- `DEBUG` — verbose logging

**Frontend (`frontend/.env.production`):**
- `VITE_API_URL` — Render backend URL

## Key Conventions

- **Graceful degradation throughout** — vision failures, ChromaDB unavailability, and Gemini errors all have keyword/template fallbacks rather than hard crashes
- **Skin metrics use a 0–10 scale** — `acne_severity`, `redness`, `pigmentation`, `hydration`, `texture`, `dark_circles`, `wrinkles`, `hair_thinning`, `scalp_condition`, `confidence_score`
- **Session cache** — in-memory dict in `main.py` keyed by `session_key`; 300s TTL; used to hand Stage 1 pre-fetched products to Stage 2
- **Products** are loaded from the static `products_db.json` (35 curated entries) into both ChromaDB and an in-memory dict at startup
- **CORS** is wide open by default — lock `ALLOWED_ORIGINS` to the Vercel domain in production
