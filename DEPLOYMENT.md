# Deployment Guide

## Architecture

```
Vercel (React/Vite frontend)
        │  HTTPS calls
        ▼
Render (FastAPI backend)
        │
        ├── Gemini API (google-genai)
        ├── ChromaDB  (local /tmp on free Render — ephemeral)
        ├── Local JSON progress store (local /tmp — ephemeral)
        └── Supabase (optional — persistent progress store)
```

> **Note on persistence on Render free tier:**  
> `/tmp` is ephemeral — ChromaDB and progress JSON reset on redeploy.  
> To persist data across deploys, either add a Render Persistent Disk (paid) and point `CHROMA_DB_PATH` / `AUDITO_DATA_DIR` to it, or use Supabase for progress data (already supported in `progress_store.py`).

---

## 1 — Backend on Render

### 1.1 Create the service

1. Push this repo to GitHub.
2. Go to [render.com](https://render.com) → **New → Web Service**.
3. Connect your GitHub repo.
4. Render will detect `render.yaml` automatically. If not, set manually:
   - **Root Directory:** leave blank (project root)
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Environment:** Python 3

### 1.2 Set environment variables in Render dashboard

Go to **Environment** tab and add:

| Key | Value |
|-----|-------|
| `GEMINI_API_KEY` | Your Gemini API key |
| `GEMINI_MODEL` | `gemini-2.5-flash` |
| `GEMINI_VISION_MODEL` | `gemini-2.5-flash` |
| `DEBUG` | `false` |
| `AUDITO_DATA_DIR` | `/tmp/audito-data` |
| `CHROMA_DB_PATH` | `/tmp/chroma_db` |
| `ALLOWED_ORIGINS` | Your Vercel URL e.g. `https://audito.vercel.app` |
| `SUPABASE_URL` | *(optional)* Your Supabase project URL |
| `SUPABASE_KEY` | *(optional)* Your Supabase anon key |

### 1.3 Verify deployment

Once deployed, visit `https://<your-render-app>.onrender.com/health` — should return:
```json
{"status": "healthy", "products_loaded": 35, "model": "gemini-2.5-flash"}
```

Note your Render URL — you'll need it for the frontend.

---

## 2 — Frontend on Vercel

### 2.1 Import project

1. Go to [vercel.com](https://vercel.com) → **New Project**.
2. Import your GitHub repo.
3. Set:
   - **Root Directory:** `frontend`
   - **Build Command:** `npm run build` (auto-detected)
   - **Output Directory:** `dist` (auto-detected)

### 2.2 Set environment variable

In **Project Settings → Environment Variables**, add:

| Key | Value |
|-----|-------|
| `VITE_API_URL` | `https://<your-render-app>.onrender.com` |

> The `VITE_` prefix is required — Vite only exposes env vars with this prefix to the browser bundle.

### 2.3 Deploy

Trigger a deploy (or it will auto-deploy from the import). The `vercel.json` at `frontend/vercel.json` handles SPA routing so page refreshes work.

### 2.4 Update CORS on Render

After Vercel assigns your domain (e.g. `https://audito.vercel.app`), go back to Render and update:
```
ALLOWED_ORIGINS = https://audito.vercel.app
```

If you have a custom domain, add it comma-separated:
```
ALLOWED_ORIGINS = https://audito.vercel.app,https://www.audito.app
```

---

## 3 — After both are live

Test the full flow:

```bash
# Health
curl https://<render-url>.onrender.com/health

# Chat
curl -X POST https://<render-url>.onrender.com/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "hi", "user_id": "test"}'
```

Then visit the Vercel URL and upload an image to test the full pipeline.

---

## 4 — Local development (unchanged)

```bash
# Backend
cd audito
.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Frontend (separate terminal)
cd frontend
npm run dev
```

Frontend proxies `/api/*` to `localhost:8000` via `vite.config.js` — no `VITE_API_URL` needed locally.

---

## 5 — Environment variables reference

| Variable | Required | Default | Notes |
|----------|----------|---------|-------|
| `GEMINI_API_KEY` | Yes | — | Google AI Studio key |
| `GEMINI_MODEL` | No | `gemini-2.5-flash` | Text model |
| `GEMINI_VISION_MODEL` | No | `gemini-2.5-flash` | Vision model |
| `DEBUG` | No | `true` | Set `false` in production |
| `ALLOWED_ORIGINS` | No | `*` (all) | Comma-separated Vercel URLs |
| `AUDITO_DATA_DIR` | No | `data/` | Where progress JSON files are stored |
| `CHROMA_DB_PATH` | No | `chroma_db/` | Where ChromaDB persists |
| `SUPABASE_URL` | No | — | Enables persistent progress store |
| `SUPABASE_KEY` | No | — | Supabase anon key |
| `PINECONE_API_KEY` | No | — | Not used — reserved for future |
| `LANGFUSE_ENABLED` | No | `false` | LLM observability |
| `VITE_API_URL` | Frontend only | `""` (relative) | Render backend URL for Vercel build |
