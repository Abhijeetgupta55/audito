# Audito — AI Skin & Hair Diagnostic Assistant

Audito is a personal skin and hair health tracker powered by a multi-agent AI pipeline. You describe a concern or upload a photo, and the system runs it through a sequence of specialised AI agents — triage, vision analysis, clinical diagnosis, knowledge-base retrieval, and ingredient recommendation — before returning a structured, actionable response. It is not a replacement for a dermatologist, but it gives you the kind of grounded, specific feedback that generic search results never do.

---

## What It Does

- Analyses uploaded skin or scalp photos using Google Gemini's vision capabilities
- Runs differential diagnosis based on what it sees and what you describe
- Recommends specific active ingredients grounded in a curated dermatology knowledge base
- Matches products from a real product database to your concern and skin type
- Tracks your skin metrics over time and surfaces trends across sessions
- Handles conversational follow-up — you can ask questions, clarify, or go back and forth naturally

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18, Vite, Framer Motion v12 |
| Backend | Python 3.11, FastAPI, Uvicorn |
| AI Model | Google Gemini 2.5 Flash (text + vision) |
| RAG / Knowledge Base | ChromaDB + LangChain + Gemini Embeddings |
| Product Database | Local JSON (35 products), optional Pinecone |
| Progress Storage | Local JSON files, optional Supabase |
| Frontend Hosting | Vercel |
| Backend Hosting | Render |

---

## Project Structure

```
audito/
├── main.py                    # FastAPI app, all API endpoints
├── backend/
│   ├── agents.py              # All 8 agents + orchestrator + Gemini integration layer
│   ├── config.py              # Settings loaded from environment variables
│   ├── vector_store.py        # Keyword-based local product search
│   ├── rag_store.py           # ChromaDB vector store + Gemini embeddings
│   ├── progress_store.py      # Per-user skin metric history and trend tracking
│   ├── evaluations.py         # Response quality metrics
│   ├── monitoring.py          # Langfuse observability (optional)
│   └── logging_config.py      # Structured logging setup
├── data/
│   └── products_db.json       # Curated dermatology product database (35 products)
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Consultation.jsx    # Main chat interface, all state management
│   │   │   ├── DiagnosticDeck.jsx  # Card-stack UI for structured AI responses
│   │   │   ├── DiagnosticDeck.css  # Card styles, animations
│   │   │   └── Home.jsx            # Landing page
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── .env.production        # VITE_API_URL pointing to Render
│   └── vercel.json            # SPA rewrite rules
└── .env                       # Local secrets — never committed
```

---

## Architecture

### Backend: Multi-Agent Pipeline

The core of Audito is a sequential agent pipeline orchestrated by `MultiAgentOrchestrator`. Each agent does one job, writes its output into a shared `AgentState` dataclass, and hands off to the next agent. The orchestrator decides which agent runs next based on the `current_agent` field each agent sets when it returns.

```
Text input path:
  Triage → Diagnosis → Product Search → Recommendation → Safety

Image input path:
  Triage → Vision → Diagnosis → Product Search → Recommendation → Safety

Severe concern path:
  Triage → Diagnosis → Doctor Expert → Safety

Conversational / chit-chat path:
  Triage → Conversational
```

#### Agent Descriptions

**1. Triage Agent**
Reads the user's message and decides what to do with it. It classifies intent into one of four categories: `concern` (a real skin or hair issue), `chit_chat` (greetings and small talk), `follow_up` (continuing a prior conversation), or `unknown`. If an image is attached, it skips the LLM entirely and routes straight to Vision. Has a keyword-based fallback so routing still works if the LLM is unavailable.

**2. Vision Agent**
Takes a base64-encoded image and runs it through Gemini's vision model. It checks image quality first, then if the image is usable, extracts: skin type, visible conditions, severity, a primary concern tag, a clinical observation (2–3 factual sentences), and 10 structured skin metrics on a 0–10 scale (acne severity, redness, pigmentation, hydration, texture, dark circles, wrinkles, hair thinning, scalp condition, confidence score). Handles three distinct failure modes separately — model API failure, JSON parse failure, and genuinely unclear photos — so a clear photo never gets told "photo not clear enough" due to a backend error.

**3. Diagnosis Agent**
Runs in two phases. Phase 1 is a clinical intake check — it reviews the conversation history and decides if there is enough information to make a differential diagnosis (needs at minimum: duration, location, and symptoms). If not, it asks 2–3 targeted follow-up questions and ends the turn. Phase 2 runs the differential diagnosis and returns structured JSON with a concerns list, severity, three diagnosis summary bullets, cautions, and a `requires_doctor` flag. When an image has been analysed, the photo findings feed directly into the diagnosis context and the intake phase is skipped.

**4. Conversational Agent**
Handles everything that does not need a clinical pipeline: greetings, general questions, off-topic queries, and error states from the vision pipeline (model failures vs. genuinely unclear photos are handled with different messages). Keeps responses short and direct. Uses `retries=0` — the fastest path in the pipeline.

**5. Product Search Agent**
Retrieves relevant products and dermatology knowledge from the knowledge base. In `stage1` (image workflow), it fetches KB context only — product retrieval happens later in `stage2`. In `stage2` and `full` mode, it retrieves both. Falls back gracefully from ChromaDB to keyword search if the vector store is unavailable.

**6. Recommendation Agent**
Runs in three modes. In `stage1` it generates recommended active ingredients grounded in the KB context, with a curated hardcoded fallback for every concern category if the LLM returns empty. In `stage2` it writes per-product clinical rationale. In `full` mode (text pipeline) it does both in one LLM call. All outputs are structured JSON.

**7. Doctor Expert Agent**
Fires only for severe presentations or when `requires_doctor` is true. Returns exactly four bullets covering urgency, immediate safe steps, what specialist to see, and what to avoid before the appointment.

**8. Safety Agent**
Final step on every path. Assembles the final response from diagnosis and recommendation text. Appends safety warnings for severe conditions.

---

### Gemini Integration Layer

All LLM calls go through a three-layer abstraction in `agents.py`:

```
_extract_response(response, model_name)
    Parses candidates/parts directly — never touches response.text.
    Logs: finish_reason, safety_ratings, token usage, prompt-level blocks.
    Skips thought parts (Gemini 2.5 internal reasoning tokens).

_generate(prompt_parts, model_name, max_tokens)
    Single synchronous call to Gemini. Never raises. Uses _extract_response.
    Logs full diagnostics on every call — empty responses include finish_reason.

_agenerate(prompt_parts, model_name, max_tokens, retries)
    Async wrapper around _generate with non-blocking asyncio.sleep retry.
    retries=0 → conversational (fast, fail fast)
    retries=1 → diagnosis, triage, recommendation
    retries=2 → vision (most critical, image data cannot be resent)
```

`thinking_budget=0` is applied only when the model name contains `"2.5"` — this disables Gemini 2.5's extended thinking mode for speed, and is skipped entirely for older models that do not support it.

---

### Frontend: Two-Stage Response Model

The image analysis pipeline splits into two stages to keep the initial response fast:

**Stage 1** fires immediately when you send an image. It runs Vision → Diagnosis → KB Retrieval → Ingredient Recommendation and returns within the main request. The frontend shows the result as a `DiagnosticDeck` with photo analysis, clinical summary, and recommended actives.

**Stage 2** auto-triggers right after Stage 1 completes. It calls `/api/recommend-products` which runs Product Search → Product Recommendation → Safety. The backend pre-fetches products in the background during Stage 1 using a session key and an in-memory cache (300s TTL), so Stage 2 is usually fast.

#### DiagnosticDeck

Structured AI responses render as a swipeable card stack (Framer Motion with spring physics). Each card type has its own visual style and data layout:

| Card | When it shows |
|---|---|
| Photo Analysis | Image was successfully analysed — shows skin type, conditions, confidence |
| Clinical Summary | Diagnosis agent returned structured bullets |
| Recommended Actives | Stage 1 returned ingredient recommendations |
| Products | Stage 2 returned matched products from the database |
| Progress | User has prior sessions — shows metric deltas and trends |
| Warning | Severity is severe or `requires_doctor` is true |
| A few questions | Diagnosis agent needs more info — intake phase |

Cards are built by a pure `buildCards(msg)` function that maps message state to typed card descriptors. Conversational responses (no structured data) fall through to a plain text bubble.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Service health, Gemini configuration status |
| `GET` | `/test-gemini` | Isolated Gemini connectivity test with retry |
| `POST` | `/api/chat` | Text conversation — full pipeline or conversational |
| `POST` | `/api/analyze-image` | Stage 1 image analysis — vision through recommendation |
| `POST` | `/api/recommend-products` | Stage 2 product retrieval — uses session cache |
| `GET` | `/api/progress/{user_id}` | Skin metric history for a user |

---

## Environment Variables

```env
# Required
GEMINI_API_KEY=...           # Google AI Studio API key

# Model selection (both default to gemini-2.5-flash)
GEMINI_MODEL=gemini-2.5-flash
GEMINI_VISION_MODEL=gemini-2.5-flash

# Optional — vector store
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=audito-products
CHROMA_DB_PATH=/tmp/chroma_db

# Optional — persistence
SUPABASE_URL=...
SUPABASE_KEY=...

# Optional — observability
LANGFUSE_ENABLED=false
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...

# CORS — comma-separated origins; leave blank to allow all
ALLOWED_ORIGINS=https://your-app.vercel.app
```

Frontend (`.env.production`):
```env
VITE_API_URL=https://your-backend.onrender.com
```

---

## Running Locally

**Backend**
```bash
cd audito
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
cp .env.example .env          # fill in GEMINI_API_KEY at minimum
uvicorn main:app --reload --port 8000
```

**Frontend**
```bash
cd frontend
npm install
npm run dev
```

The frontend dev server proxies are not configured — set `VITE_API_URL=http://localhost:8000` in `frontend/.env.local` to point at your local backend.

---

## Deployment

**Backend → Render**
- Connect your GitHub repo to a new Render web service
- Set all required environment variables in the Render dashboard
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

**Frontend → Vercel**
- Import the `frontend/` directory as a Vite project
- Set `VITE_API_URL` to your Render service URL
- `vercel.json` handles SPA routing — the rewrite rule excludes `/api/` paths so backend calls are not intercepted

---

## Resilience Design

The system is designed to degrade gracefully at every layer rather than fail hard:

- **RAG unavailable** — falls back to keyword search against the local product JSON
- **LLM empty response** — retries up to 2 times with async backoff before giving up
- **Actives LLM failure** — falls back to a curated hardcoded ingredient map (11 concern categories)
- **Vision model failure** — distinguished from a blurry photo; user gets "temporarily unavailable" not "photo not clear"
- **Triage LLM failure** — falls back to a keyword-based rule set covering all concern categories
- **Diagnosis LLM failure** — returns a safe fallback clinical summary pointing the user to a dermatologist

---

## Limitations

- Not a medical device and not a substitute for professional dermatological advice
- Product database is curated and limited — not every product on the market is covered
- Progress tracking resets if the Render instance restarts (ephemeral storage on free tier)
- Analysis quality depends on photo conditions — frontal, well-lit, in-focus images produce significantly better results
