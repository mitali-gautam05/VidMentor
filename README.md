# YT Doubt Solver❓

A Chrome extension that lets you ask questions about a YouTube video while you're watching it — grounded entirely in that video's own transcript using RAG (Retrieval-Augmented Generation).

Stuck on a concept mid-video? Open the chat, ask your doubt, and get an answer sourced directly from what the speaker actually said — with clickable timestamps that jump you back to the relevant part of the video.

---

## How it works

```
YouTube video page
   └── content script (chat widget, reads video_id + current playback time)
         │  fetch()
         ▼
FastAPI backend
   ├── /index → fetches transcript, chunks it (40s windows, timestamped),
   │            embeds chunks, builds a FAISS index, caches it (memory + disk)
   └── /chat  → embeds the question, retrieves top-k relevant chunks
                (biased toward the current playback timestamp),
                sends them + the question to an LLM (Groq),
                returns a grounded answer
```

**Pipeline:** `video_id → transcript → timestamped chunks → embeddings → FAISS index → [query + current_time] → retrieval (time-biased) → LLM answer`

---

## Tech stack

| Layer | Tool |
|---|---|
| Transcript extraction | `youtube-transcript-api` |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`, 384-dim) |
| Vector search | `faiss-cpu` |
| LLM generation | Groq (`openai/gpt-oss-20b`) |
| Backend | FastAPI + Uvicorn |
| Extension | Chrome Manifest V3 (content script + service worker) |

**Chunking strategy:** fixed time-window chunking — transcript segments are grouped into ~40-second windows (non-overlapping), each tagged with its start/end timestamp. This keeps chunks meaningful (a "complete thought") and lets retrieval be biased toward what's currently playing.

**Retrieval:** semantic similarity search via FAISS, re-ranked with a soft penalty for chunks far from the viewer's current timestamp — so answers stay relevant to where the user actually is in the video, not just wherever similar words appear.

---

## Features

-  Answers grounded only in the video's transcript (no hallucinated general knowledge)
-  Timestamp-aware retrieval — prioritizes context near your current playback position
-  Multi-turn conversation — follow-up questions ("what's *its* time complexity?") resolve correctly using chat history
-  Clickable timestamps in answers — jump the video to that exact moment
-  Per-video caching (memory + disk) — no re-indexing on repeat visits
-  Handles disabled/missing captions, invalid video IDs, and API rate limits gracefully
-  Multi-language transcript fallback (tries English/Hindi first, falls back to whatever's available)

---

## Project structure

```
YT/
├── rag-backend/
│   ├── main.py              # FastAPI app — all backend logic
│   ├── requirements.txt
│   ├── .env                 # GROQ_API_KEY (never committed)
│   ├── .gitignore
│   └── rag_cache/           # auto-created — cached FAISS indexes + chunks
│
└── yt-doubt-extension/
    ├── manifest.json         # Manifest V3 config
    ├── background.js         # service worker — handles API calls
    ├── content.js            # injects chat widget on YouTube pages
    ├── content.css           # widget styling
    └── icons/
        └── icon128.png
```

---

## Setup

### Backend

```bash
cd rag-backend
python -m venv venv
.\venv\Scripts\Activate.ps1      # Windows
# source venv/bin/activate       # Mac/Linux

pip install -r requirements.txt
```

Create a `.env` file in `rag-backend/`:
```
GROQ_API_KEY=your_groq_api_key_here
```
(Free key from [console.groq.com](https://console.groq.com/keys))

Run it:
```bash
uvicorn main:app --reload --port 8000
```

Verify: open `http://localhost:8000/health` → should return `{"status": "ok"}`.
Interactive API docs: `http://localhost:8000/docs`

### Extension

1. Open `chrome://extensions`
2. Enable **Developer mode** (top right)
3. Click **Load unpacked**
4. Select the `yt-doubt-extension` folder

Go to any YouTube video, refresh the page, and look for the **"❓ Ask Doubt"** button in the bottom-right corner.

> If you deploy the backend (see below), update `BACKEND_URL` in `background.js` and `host_permissions` in `manifest.json` to point to the deployed URL instead of `localhost:8000`.

---

## API Reference

### `POST /index`
Pre-builds the index for a video (optional — `/chat` builds it automatically on first use).
```json
{ "video_id": "VIDEO_ID" }
```

### `POST /chat`
```json
{
  "video_id": "VIDEO_ID",
  "query": "why did he use a hashmap instead of an array?",
  "current_time": 245
}
```
Response:
```json
{ "success": true, "answer": "..." }
```

### `GET /health`
Basic liveness check.

---

## Deployment

Deployed on [Render](https://render.com) (free tier):
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Environment variable: `GROQ_API_KEY`

**Known free-tier limitations:**
- Server spins down after ~15 min of inactivity → first request after idle has a cold-start delay (~30-50s)
- Disk is ephemeral — cached indexes and chat history reset on redeploy/restart

---

## Possible future improvements

- Overlapping chunk windows (reduce context loss at chunk boundaries)
- Move cache/session storage to Redis or a persistent volume
- Support for non-English videos beyond the current fallback
- Voice input for asking doubts
- Auto-pause video when the chat panel opens