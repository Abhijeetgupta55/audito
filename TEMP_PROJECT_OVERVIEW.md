# Audito — Complete A-to-Z Project Overview (TEMP / LOCAL ONLY)

> This file is gitignored and never pushed. It is your personal reference doc.

---

## 1. What Is This Project?

**Audito** is an AI-powered skin and hair diagnostic assistant. A user uploads a photo of their face or scalp (or just types a concern), and the system returns:

- A clinical analysis of what it sees in the image
- A differential diagnosis (ranked list of likely conditions)
- Recommended active ingredients with mechanism explanations
- Specific skincare products from a curated 35-product database
- A progress tracker that compares metrics across sessions
- A "see a doctor" escalation path for severe presentations

**The core philosophy:** not a replacement for a dermatologist, but gives grounded, specific feedback that a generic Google search never does.

**Live deployments:**
- Backend: `https://audito-nti3.onrender.com` (Render, free tier)
- Frontend: Vercel (root at `frontend/`)

---

## 2. Repository Layout

```
a:\audito\audito\
│
├── main.py                          ← FastAPI app: all API endpoints, session cache
├── requirements.txt                 ← Python deps
├── render.yaml                      ← Render deploy config (build + start commands)
├── products_db.json                 ← 35 curated skincare/haircare products (static data)
├── evaluations.py                   ← Quality metrics (root-level import shim)
├── monitoring.py                    ← Langfuse integration (root-level shim, 82 bytes)
├── logging_config.py                ← JSON structured logging setup
├── .env                             ← Local secrets (gitignored)
├── CLAUDE.md                        ← Claude Code instructions (gitignored)
├── DEPLOYMENT.md                    ← How to deploy step-by-step
├── README.md                        ← Full project documentation
│
├── backend/
│   ├── agents.py                    ← THE BRAIN: 8 agents + orchestrator (65KB)
│   ├── config.py                    ← Reads all env vars into a Settings object
│   ├── rag_store.py                 ← ChromaDB setup + Gemini embeddings
│   ├── vector_store.py              ← Keyword fallback search (no vector DB needed)
│   ├── progress_store.py            ← Per-user skin metric history + trend analysis
│   ├── supabase_client.py           ← Optional Supabase connector for persistence
│   ├── evaluations.py               ← Actual EvaluationMetrics class
│   ├── monitoring.py                ← Actual Langfuse wrapper
│   ├── logging_config.py            ← Actual JSON logger
│   └── knowledge_base/
│       └── dermatology_kb.md        ← 36KB medical reference (ingredients, mechanisms)
│
├── data/                            ← Per-user JSON progress files (gitignored, ephemeral)
├── chroma_db/                       ← ChromaDB vector store files (gitignored, ephemeral)
├── .venv/                           ← Python virtual environment
│
└── frontend/
    ├── src/
    │   ├── main.jsx                 ← React 19 entry point
    │   ├── App.jsx                  ← Root router (just renders <Consultation />)
    │   ├── App.css
    │   ├── index.css
    │   └── components/
    │       ├── Consultation.jsx     ← Main chat UI: all state, uploads, API calls (25KB)
    │       ├── Consultation.css     ← Chat bubble styling, input box (23KB)
    │       ├── DiagnosticDeck.jsx   ← Swipeable animated card stack (17KB)
    │       ├── DiagnosticDeck.css   ← Card spring animations (15KB)
    │       ├── Home.jsx             ← Landing page (3.7KB, not in active flow)
    │       └── Home.css
    ├── public/
    ├── dist/                        ← Production build output (gitignored)
    ├── package.json                 ← Frontend deps
    ├── vite.config.js               ← Dev proxy /api/* → localhost:8000
    ├── vercel.json                  ← SPA rewrite (excludes /api/)
    ├── .env.production              ← VITE_API_URL=https://audito-nti3.onrender.com
    └── index.html
```

---

## 3. Full Tech Stack

### Backend

| Layer | Technology | Version | Why |
|---|---|---|---|
| Language | Python | 3.11 | Async support, rich AI ecosystem |
| Web Framework | FastAPI | 0.110+ | Auto docs, async, pydantic models |
| Server | Uvicorn | 0.29+ | ASGI server for FastAPI |
| AI / LLM | Google Gemini 2.5 Flash | via `google-genai >= 1.0` | Vision + text in one model, fast |
| Embeddings | Gemini `text-embedding-001` | google-genai SDK | Same vendor, consistent quality |
| Vector DB | ChromaDB | 0.5+ | Local, no external dependency |
| RAG Framework | LangChain | langchain-core/community/text-splitters | Document chunking + retrieval |
| Image Processing | Pillow | 10+ | Resize/encode before sending to Gemini |
| Observability | Langfuse | 2.0+ | LLM tracing (optional, off by default) |
| Persistence | Local JSON + Supabase | supabase-py 2.0+ | Progress history, optional cloud |
| Config | python-dotenv | — | Load .env file |
| File Uploads | python-multipart | — | FastAPI multipart form handling |

### Frontend

| Layer | Technology | Version | Why |
|---|---|---|---|
| Language | JavaScript (JSX) | ES2022+ | Standard React ecosystem |
| Framework | React | 19.2.6 | Latest; concurrent mode |
| Build Tool | Vite | 8.0.12 | Fast HMR, easy proxy config |
| Animation | Framer Motion | 12.38.0 | Spring physics for card stack |
| HTTP | axios | 1.16.1 | Promise-based, interceptors |
| Icons | lucide-react | 1.16.0 | Consistent icon set |
| Routing | react-router-dom | 7.15.1 | SPA routing |
| Camera | react-webcam | 7.2.0 | In-browser photo capture |
| Linting | ESLint | 10.3.0 | Code quality |

### Infrastructure

| Concern | Tool | Notes |
|---|---|---|
| Backend hosting | Render | Free tier, ephemeral disk, auto-deploy from git |
| Frontend hosting | Vercel | Auto-deploy from git, SPA rewrites |
| Vector DB (prod) | Pinecone | Reserved/future; currently local ChromaDB |
| Persistent DB | Supabase | Optional PostgreSQL for progress history |
| LLM Observability | Langfuse | Optional; traces every Gemini call |

---

## 4. The Multi-Agent Pipeline (The Brain)

This is the most important part of the codebase. Everything lives in `backend/agents.py`.

### How Agents Work (Architecture Pattern)

The system uses a **shared state machine** pattern:
- All agents share one `AgentState` dataclass
- Each agent reads from state, does its work, writes results back to state
- Each agent sets `state.current_agent` to point at the next agent
- `MultiAgentOrchestrator` reads `state.current_agent` and dispatches the loop

```
Request arrives
     ↓
MultiAgentOrchestrator.run()
     ↓
while state.current_agent != "complete":
    agent = agents[state.current_agent]
    state = await agent.run(state)
     ↓
Return state (mapped to API response)
```

### The 8 Agents

#### 1. Triage Agent
**What it does:** Classifies the user's intent so the pipeline takes the right path.

**Outputs one of four intents:**
- `concern` → user has a real skin/hair problem → goes to Diagnosis
- `chit_chat` → greeting or off-topic → goes to Conversational
- `follow_up` → continuing a previous conversation → goes to Diagnosis with context
- `unknown` → can't tell → goes to Conversational (safe default)

**Special case:** If there's an image attached, it skips LLM classification entirely and routes directly to Vision Agent.

**Fallback:** If Gemini is unavailable, uses keyword matching on the message text.

**Retry count:** 1

---

#### 2. Vision Agent
**What it does:** Analyzes a base64-encoded skin or scalp photo using Gemini's multimodal capability.

**Input:** Base64 image (Pillow resizes to max 1024px before encoding)

**Output (structured JSON):**
```json
{
  "skin_type": "oily / dry / combination / normal",
  "conditions": ["acne", "hyperpigmentation", ...],
  "severity": "mild / moderate / severe",
  "primary_concern": "one-line summary",
  "clinical_observation": "detailed paragraph",
  "metrics": {
    "acne_severity": 0-10,
    "redness": 0-10,
    "pigmentation": 0-10,
    "hydration": 0-10,
    "texture": 0-10,
    "dark_circles": 0-10,
    "wrinkles": 0-10,
    "hair_thinning": 0-10,
    "scalp_condition": 0-10,
    "confidence_score": 0-10
  }
}
```

**Why 2 retries:** This is the most critical agent — if vision fails, the whole image flow fails. It also distinguishes between LLM errors (retry) and blurry/unusable photos (tell the user).

**Important:** Gemini safety filters are disabled (`_SAFETY_OFF`) because medical skin content would otherwise be blocked.

---

#### 3. Diagnosis Agent
**What it does:** Produces a differential diagnosis — a ranked list of likely skin conditions.

**Two phases:**
1. **Intake check** — does the system have enough clinical information to diagnose? If not, asks the user for more (e.g., "how long has this been present?"). Skipped if image was already analyzed.
2. **Diagnosis generation** — produces structured JSON with conditions list, severity, summary, cautions, and a `requires_doctor` boolean.

**Output:**
```json
{
  "concerns": ["acne vulgaris", "post-inflammatory hyperpigmentation"],
  "severity": "moderate",
  "diagnosis_summary": ["Comedonal acne with mild inflammatory component..."],
  "cautions": ["Avoid picking", "Use SPF daily"],
  "requires_doctor": false
}
```

**Routing after diagnosis:**
- `requires_doctor = true` → DoctorExpert Agent
- Otherwise → ProductSearch Agent

**Retry count:** 1

---

#### 4. ProductSearch Agent
**What it does:** Retrieves relevant knowledge base context and products based on the diagnosed concern.

**Two stages:**
- **Stage 1** (image flow): retrieves KB-only context (no products yet) — fast, for Vision → Diagnosis → KB retrieval path
- **Stage 2**: retrieves products from ChromaDB using semantic similarity + keyword fallback

**Storage searched:**
- ChromaDB collection `dermatology_kb` (chunked from `dermatology_kb.md`)
- ChromaDB collection `products` (35 products)
- Falls back to keyword search (`vector_store.py`) if ChromaDB is down

**Output:** `kb_context` (text), `products` (list of matching product objects)

---

#### 5. Recommendation Agent
**What it does:** Takes the diagnosis + KB context + products and generates clinical rationale.

**Two phases:**
1. **Active ingredients generation** — for each condition, generates a list of recommended actives:
   ```json
   [{"name": "Niacinamide", "mechanism": "inhibits melanosome transfer", "target_concern": "hyperpigmentation"}]
   ```
2. **Per-product rationale** — for each retrieved product, generates a 1-2 sentence clinical explanation of why it's appropriate for this specific user.

**Fallback:** If Gemini returns empty actives, uses a hardcoded map for 11 concern categories (acne, redness, hyperpigmentation, dry skin, oily skin, sensitive skin, anti-aging, dark circles, rosacea, eczema, hair loss).

**Retry count:** 1

---

#### 6. Conversational Agent
**What it does:** Handles everything that isn't a skin concern — greetings, off-topic questions, error fallbacks.

**Retry count:** 0 (fastest path, fail fast)

---

#### 7. DoctorExpert Agent
**What it does:** Fires when `requires_doctor = true` from Diagnosis. Gives medically-grounded escalation guidance.

**Output:**
- Urgency level (routine / soon / urgent / emergency)
- Immediate steps the user can take now
- What type of specialist to see (dermatologist, GP, etc.)
- What to avoid doing

---

#### 8. Safety Agent
**What it does:** Always runs last on all paths. Assembles the final response and appends safety disclaimers appropriate to the severity level.

**Why it exists:** Ensures no matter which path ran, the response always:
- Has appropriate medical disclaimers
- Is assembled in a consistent format
- Adds severity-appropriate warnings (mild / moderate / severe caveats differ)

---

### Routing Map (Visual)

```
User sends text concern
    → Triage (concern) → Diagnosis → ProductSearch → Recommendation → Safety → DONE

User sends image (Stage 1 via /api/analyze-image)
    → Triage (image) → Vision → Diagnosis → ProductSearch(KB only) → Recommendation(actives only) → DONE
    → Response returned immediately + session_key cached

Frontend auto-calls Stage 2 (/api/recommend-products)
    → ProductSearch(products) → Recommendation(rationale) → Safety → DONE

User sends chit-chat or greeting
    → Triage (chit_chat) → Conversational → DONE

Severe case detected
    → Triage → Diagnosis (requires_doctor=true) → DoctorExpert → Safety → DONE
```

---

### Gemini Integration (How LLM Calls Work)

All Gemini calls funnel through three helpers in `agents.py` (lines ~50-212):

**`_extract_response(response)`**
- Parses Gemini's response object
- **Skips thought tokens** (Gemini 2.5 generates internal "thinking" that appears as parts with `thought=True` — these are filtered out so only the actual response text is used)
- Logs: `finish_reason`, `safety_ratings`, token counts (prompt/output/thought)
- Returns plain text

**`_generate(prompt, ...)`**
- Synchronous single call, no retry
- Used for non-critical paths
- Never raises — returns empty string on failure

**`_agenerate(prompt, ...)`**
- Async with exponential backoff
- Retries 0, 1, or 2 times depending on `agent.retries` setting
- Used for all main pipeline agents

**`thinking_budget=0`**
- Applied only on `gemini-2.5` text models
- Disables extended thinking (chain-of-thought) to reduce latency
- Vision model calls are NOT given `thinking_budget` (different API path)

**`_SAFETY_OFF`**
- All four Gemini content safety filters set to `BLOCK_NONE`
- Required because medical skin imagery and clinical descriptions would otherwise be blocked by default filters

---

## 5. API Endpoints (Complete Reference)

All endpoints defined in `main.py`.

### Core Endpoints

#### `POST /api/chat`
Full text pipeline. Used when the user types a concern without uploading an image.

**Request:**
```json
{
  "message": "I have red bumps on my forehead",
  "user_id": "user_abc",
  "session_id": "session_xyz",
  "conversation_history": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

**Response (`ChatResponse`):**
```json
{
  "response": "Based on your description...",
  "intent": "concern",
  "concern": "acne",
  "severity": "mild",
  "skin_type": null,
  "diagnosis": "Inflammatory acne vulgaris...",
  "diagnosis_data": {"concerns": [...], "requires_doctor": false},
  "actives": [{"name": "Salicylic Acid", "mechanism": "..."}],
  "products": [...],
  "agent_path": ["triage", "diagnosis", "product_search", "recommendation", "safety"],
  "latency_ms": 4823,
  "session_id": "session_xyz"
}
```

---

#### `POST /api/analyze-image` (Stage 1)
Vision + diagnosis + actives. Fast initial response. Caches products for Stage 2.

**Request:** `multipart/form-data`
- `image`: file upload
- `concern`: text description (optional)
- `user_id`: string
- `session_id`: string
- `conversation_history`: JSON string

**Response:** Same `ChatResponse` shape, but `products` may be empty (Stage 2 fills it).
Also returns `session_key` used to retrieve cached products.

**Side effect:** Kicks off a background task to pre-fetch products from ChromaDB and caches them under `session_key` with a 300-second TTL.

---

#### `POST /api/recommend-products` (Stage 2)
Retrieves pre-fetched products and generates rationale. Called automatically by frontend after Stage 1.

**Request:**
```json
{
  "session_key": "...",
  "concern": "acne",
  "severity": "mild",
  "skin_type": "oily",
  "kb_context": "...",
  "skin_analysis": {...}
}
```

**Response:** Products with per-product clinical rationale + safety wrap-up.

---

### Supporting Endpoints

| Endpoint | Method | What it does |
|---|---|---|
| `GET /health` | GET | Service status: Gemini API configured?, product count, version |
| `GET /test-gemini` | GET | Isolated Gemini ping (retries once), returns model name + response |
| `GET /api/progress/{user_id}` | GET | User's skin metric history, trends, natural language insights |
| `GET /api/progress/{user_id}/history` | GET | Raw array of all progress records |
| `DELETE /api/progress/{user_id}` | DELETE | Remove all progress data for a user |
| `GET /api/search?concern=acne` | GET | Direct keyword product search (bypass agents) |
| `POST /api/feedback` | POST | Submit doctor quality rating (updates EvaluationMetrics) |
| `GET /api/agents` | GET | List all 8 agents with descriptions |
| `GET /api/metrics` | GET | All evaluation metrics as JSON |

---

## 6. Frontend Architecture

### Component Hierarchy

```
main.jsx
  └── App.jsx (react-router)
        ├── / → Home.jsx (landing, mostly unused)
        └── /consultation → Consultation.jsx
                              └── DiagnosticDeck.jsx
```

### Consultation.jsx — The Main Component

This is the entire user experience. State it manages:

| State var | Type | Purpose |
|---|---|---|
| `messages` | array | Chat message history (role + content + cards) |
| `currentImage` | File | The currently selected/uploaded image |
| `userMessage` | string | Typed text input |
| `isLoading` | bool | Stage 1 loading state |
| `stage2Loading` | bool | Stage 2 loading state |
| `stage2Pending` | bool | Whether Stage 2 hasn't fired yet |
| `sessionKey` | string | Cache key from Stage 1 for Stage 2 |
| `conversationHistory` | array | Sent to backend for context |

**Two-stage image flow in the component:**

1. User picks an image → base64 encoded client-side
2. POST to `/api/analyze-image` → get Stage 1 response → render initial cards
3. Automatically POST to `/api/recommend-products` using the `session_key` → get product cards → update message with products

This gives the user fast initial feedback (clinical analysis) while products load in the background.

**Message rendering:** Each message object can have a `cards` array. If `cards` exists, `DiagnosticDeck` is rendered instead of a plain text bubble.

---

### DiagnosticDeck.jsx — The Card Stack

A swipeable card stack built with Framer Motion. Think of it like a presentation deck embedded in the chat.

**Card types and what they show:**

| Card type | Content |
|---|---|
| `photo` | The uploaded image |
| `clinical_summary` | Skin type, detected conditions, severity badge |
| `intake` | "Tell me more" intake questions from Diagnosis Agent |
| `actives` | List of recommended active ingredients with mechanisms |
| `actives_pending` | Loading placeholder while Stage 2 fetches |
| `products_cta` | "Loading products..." call-to-action |
| `products_loading` | Spinner while Stage 2 is in progress |
| `products` | Final product recommendations with rationale |
| `progress` | User's metric history chart/trend summary |
| `warning` | Doctor escalation card with urgency + steps |

**`buildCards()` function:** Maps the API response fields to typed card objects. Called when a new assistant message arrives.

**Animation:** Framer Motion spring physics
- `stiffness: 420`, `damping: 36`, `mass: 0.75`
- Cards stack with 3D perspective tilt
- Next/prev navigation + dot indicators

**HTTP:** All API calls use axios. `VITE_API_URL` env var sets the base URL. In dev, Vite proxies `/api/*` to `localhost:8000`.

---

## 7. Data Layer

### products_db.json (Static, 35 Products)

Each product has this shape:
```json
{
  "id": "cerave-foaming-cleanser",
  "name": "CeraVe Foaming Facial Cleanser",
  "brand": "CeraVe",
  "description": "...",
  "key_ingredients": ["Niacinamide", "Hyaluronic Acid", "Ceramides"],
  "concerns": ["oily skin", "acne"],
  "suitable_for": ["oily", "combination", "normal"],
  "how_to_use": "...",
  "price_range": "$",
  "format": "cleanser"
}
```

Loaded at startup into:
1. In-memory dict (fast lookup by ID)
2. ChromaDB `products` collection (semantic search)

---

### ChromaDB (Vector Store)

Two collections:
- **`dermatology_kb`** — the knowledge base MD file chunked into ~500-token overlapping segments
- **`products`** — each product's description + ingredients embedded for semantic retrieval

**Embedding model:** `text-embedding-001` via `GeminiEmbeddings` (custom LangChain wrapper in `rag_store.py`)

**Stored at:** `chroma_db/` locally, `/tmp/chroma_db` on Render

**Ephemeral on Render free tier** — rebuilt from `products_db.json` and `dermatology_kb.md` on every deploy/restart. This is fine since both source files are static.

---

### Progress Store (progress_store.py)

Tracks skin metric history per user across sessions.

**What gets stored per scan:**
```json
{
  "user_id": "user_abc",
  "timestamp": "2026-05-19T10:00:00Z",
  "metrics": {
    "acne_severity": 6,
    "redness": 4,
    "pigmentation": 5,
    ...
  },
  "skin_type": "oily",
  "conditions": ["acne", "hyperpigmentation"]
}
```

**What it computes:**
- Delta from previous scan (each metric went up/down/same)
- Trend over last N scans (improving / worsening / stable)
- Lighting consistency check (flags if images were taken in very different lighting)
- Natural language insight: "Your acne has improved by 2 points since last week"

**Storage hierarchy:**
1. Supabase table `progress_history` (if configured)
2. Local JSON file `data/{user_id}.json` (fallback, ephemeral on Render)

---

### In-Memory Session Cache (main.py)

```python
_reco_cache: Dict[str, Dict] = {}
# Key: session_key (UUID generated per Stage 1 request)
# Value: {products, kb_context, skin_analysis, expires_at}
# TTL: 300 seconds
```

Used to pass Stage 1 pre-fetched products to Stage 2 without re-fetching from ChromaDB.

---

### Knowledge Base (dermatology_kb.md, 36KB)

Medical reference document with detailed entries for each ingredient:

**Ingredients covered:** Retinol, Niacinamide, Salicylic Acid, Hyaluronic Acid, Vitamin C (Ascorbic Acid), Ceramides, AHAs (Glycolic/Lactic Acid), Benzoyl Peroxide, Azelaic Acid, Zinc, Centella Asiatica

**Each entry covers:**
- Mechanism of action (how it actually works at the cellular level)
- Clinical benefits
- Recommended concentrations
- Side effects and contraindications
- Ingredient compatibility (what to combine/avoid)
- Usage instructions
- Which skin types it suits

This KB is the grounding source for Recommendation Agent's rationale — Gemini's answers are constrained to what's in this document via RAG retrieval.

---

## 8. Configuration System

All env vars are read in `backend/config.py` into a Pydantic `Settings` object. This is the single source of truth for configuration.

### Required

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Google AI Studio API key |

### Optional (with defaults)

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_MODEL` | `gemini-2.5-flash` | Text model for all non-vision agents |
| `GEMINI_VISION_MODEL` | `gemini-2.5-flash` | Vision model for image analysis |
| `CHROMA_DB_PATH` | `chroma_db` | Where to store vector DB |
| `AUDITO_DATA_DIR` | `data` | Where to store JSON progress files |
| `DEBUG` | `False` | Verbose logging |
| `API_TITLE` | `Audito API` | FastAPI title |
| `API_VERSION` | `1.0.0` | FastAPI version |
| `ALLOWED_ORIGINS` | `["*"]` | CORS whitelist (lock this in production!) |
| `SUPABASE_URL` | `""` | Supabase project URL |
| `SUPABASE_KEY` | `""` | Supabase anon/service key |
| `LANGFUSE_ENABLED` | `False` | Toggle LLM tracing |
| `LANGFUSE_PUBLIC_KEY` | `""` | Langfuse project public key |
| `LANGFUSE_SECRET_KEY` | `""` | Langfuse secret key |
| `PINECONE_API_KEY` | `""` | Reserved, not currently used |
| `PINECONE_INDEX_NAME` | `audito-products` | Reserved, not currently used |

### Frontend

| Variable | File | Purpose |
|---|---|---|
| `VITE_API_URL` | `.env.production` | Render backend URL for production builds |

---

## 9. Deployment

### Backend (Render)

**How it works:**
1. Push to git → Render auto-detects `render.yaml`
2. Runs: `pip install -r requirements.txt`
3. Starts: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. On startup, `main.py` initializes: logging → ChromaDB (loads products + KB) → orchestrator

**Ephemeral storage issue:**
- Render free tier has no persistent disk
- `chroma_db/` and `data/` are written to `/tmp` (lost on restart)
- ChromaDB rebuilds from source files on each startup (acceptable since products_db.json is small)
- Progress data is lost on restart unless Supabase is configured

**Environment variables:** Set in Render dashboard (not in render.yaml for secrets).

---

### Frontend (Vercel)

**How it works:**
1. Push to git → Vercel auto-builds
2. Root directory: `frontend/`
3. Build: `npm run build` (Vite → `dist/`)
4. `vercel.json` rewrites: all routes except `/api/` → `/index.html` (SPA behavior)
5. `.env.production` sets `VITE_API_URL` to Render backend

---

### Local Development

**Backend:**
```powershell
cd a:\audito\audito
.venv\Scripts\Activate.ps1
uvicorn main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev   # Starts on :5173, proxies /api/* → :8000
```

Vite's dev proxy (in `vite.config.js`) means the frontend at `localhost:5173` transparently forwards all `/api/*` calls to `localhost:8000` — no CORS issues in dev.

---

## 10. Observability & Quality

### Logging (logging_config.py)

JSON structured logs to stdout. Every line is a JSON object with:
- `timestamp`, `level`, `logger`, `message`
- Exception traces when present

### Evaluation Metrics (backend/evaluations.py)

`EvaluationMetrics` class tracks in-memory:

| Metric | How measured |
|---|---|
| Total requests | Counter |
| Success rate | Success / total |
| Hallucination rate | Product name found in response / total product mentions |
| Safety appropriateness | Keyword-based check on response content |
| Product relevance | 0-1 score from Recommendation Agent |
| Latency percentiles | p50, p95, p99, max, min |
| Agent path distribution | Which paths ran how often |
| Doctor feedback scores | Average of submitted DoctorFeedback objects |
| Monthly cost estimate | ~$0.009/request based on Gemini pricing |

Exposed at `GET /api/metrics`.

### Langfuse (monitoring.py)

Optional. When `LANGFUSE_ENABLED=true`:
- Every Gemini call is traced as a "generation"
- Includes prompt, response, tokens, latency
- Viewable in Langfuse dashboard

---

## 11. Key Design Decisions & Why

### Why Multi-Agent?

Each agent has a single responsibility. This makes each failure mode independent — if the Recommendation Agent fails, the Safety Agent still runs and gives a coherent (if sparse) response. It also makes the routing logic explicit and auditable.

### Why Two-Stage Image Flow?

Stage 1 (vision → diagnosis → actives) is fast ~3-4s and immediately useful. Product retrieval (ChromaDB search + rationale generation) takes additional time. Splitting into two API calls lets the user see clinical results immediately while products load asynchronously.

### Why Disable Gemini Safety Filters?

Medical skin content (lesions, rashes, dermatological conditions) is routinely flagged by default content filters. Disabling them is intentional and necessary for the app to function. The Safety Agent + medical disclaimer UI compensates.

### Why `thinking_budget=0`?

Gemini 2.5's extended thinking (chain-of-thought reasoning) adds 2-5 seconds of latency. For a conversational medical app, that's too slow. The model is capable enough without extended thinking for these structured JSON extraction tasks.

### Why Graceful Degradation Everywhere?

Gemini API is not perfectly reliable. A user uploading a photo expecting a response should get something useful even if one agent fails. The fallback chain ensures this:
- LLM failure → template/keyword fallback
- ChromaDB unavailable → keyword search
- No products found → generic ingredient recommendations

### Why Static products_db.json?

35 curated, vetted products is better than 3500 scraped ones. The curation ensures all recommendations are dermatologist-appropriate. Adding products requires manual review, not a scraper.

---

## 12. Known Limitations / Areas to Watch

| Issue | Impact | Notes |
|---|---|---|
| Render free tier = ephemeral storage | Progress data lost on restart | Fix: configure Supabase |
| ChromaDB rebuilt on every restart | ~5-10s cold start delay | Acceptable for 35 products |
| Gemini API key in `.env` file | Security risk if .env ever committed | Key should only be in Render dashboard |
| CORS wide open (`["*"]`) | Any origin can call the backend | Lock to Vercel URL in production |
| No test suite | Manual testing only | `/health` and `/test-gemini` are the only automated checks |
| In-memory session cache | Lost on restart, doesn't scale horizontally | Fine for single-instance Render deploy |
| Pinecone configured but unused | Dead config | Can remove or implement |

---

## 13. How It All Connects (End-to-End Flow)

### Text concern flow:

1. User types "I have red bumps on my cheeks" into Consultation.jsx
2. Frontend POSTs to `/api/chat` with message + conversation_history
3. `main.py` creates `AgentState`, sets `current_agent = "triage"`
4. Triage Agent → calls Gemini → classifies as `concern` → sets next to `diagnosis`
5. Diagnosis Agent → calls Gemini → returns conditions + severity → sets next to `product_search`
6. ProductSearch Agent → queries ChromaDB → retrieves KB context + matching products → sets next to `recommendation`
7. Recommendation Agent → calls Gemini for actives → calls Gemini for per-product rationale → sets next to `safety`
8. Safety Agent → assembles final response → sets next to `complete`
9. `main.py` maps state to `ChatResponse` → returns JSON
10. Frontend receives response → `buildCards()` → renders DiagnosticDeck cards

### Image flow:

1. User uploads photo → Consultation.jsx base64-encodes it
2. Frontend POSTs to `/api/analyze-image` (multipart)
3. `main.py` runs Stage 1 pipeline: Vision → Diagnosis → ProductSearch(KB) → Recommendation(actives)
4. Background task: ProductSearch(products) runs and caches result under `session_key`
5. Stage 1 response returned → frontend renders initial cards (clinical_summary, actives)
6. Frontend immediately POSTs to `/api/recommend-products` with `session_key`
7. `main.py` retrieves cached products → Recommendation(rationale) → Safety
8. Stage 2 response returned → frontend updates cards with product cards
9. Progress stored: `progress_store.py` saves metrics to `data/{user_id}.json` or Supabase

---

## 14. Files You Need to Know for Each Task

| Task | Key files |
|---|---|
| Adding a new agent | `backend/agents.py` + routing logic in `MultiAgentOrchestrator` |
| Adding a new API endpoint | `main.py` |
| Adding a product | `products_db.json` (restart rebuilds ChromaDB) |
| Changing the knowledge base | `backend/knowledge_base/dermatology_kb.md` |
| Changing UI cards | `frontend/src/components/DiagnosticDeck.jsx` + `buildCards()` |
| Changing chat UI | `frontend/src/components/Consultation.jsx` |
| Adding a new env var | `backend/config.py` + Render dashboard |
| Debugging Gemini calls | `backend/agents.py` → `_extract_response()`, `_agenerate()` |
| Debugging RAG | `backend/rag_store.py` |
| Debugging progress tracking | `backend/progress_store.py` |
| Checking metrics | `GET /api/metrics` or `backend/evaluations.py` |

---

*This file is for local reference only. Never commit it.*
