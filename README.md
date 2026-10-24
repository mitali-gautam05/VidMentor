# VidMentor

A Chrome extension that lets you ask questions about a YouTube video while you're watching it, grounded entirely in that video's own transcript using RAG (Retrieval-Augmented Generation).

Stuck on a concept mid-video? Open the chat, ask your doubt, and get an answer sourced directly from what the speaker actually said, with clickable timestamps that jump you back to the relevant part of the video.

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
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`, 384-dim, configurable) |
| Vector search | `faiss-cpu` |
| LLM generation | Groq (`openai/gpt-oss-20b`) |
| Backend | FastAPI + Uvicorn |
| Extension | Chrome Manifest V3 (content script + service worker) |
| Tests | pytest |

**Chunking strategy:** fixed time-window chunking. Transcript segments are grouped into ~40-second windows (non-overlapping), each tagged with its start/end timestamp. This keeps chunks meaningful (a "complete thought") and lets retrieval be biased toward what's currently playing.

**Retrieval:** semantic similarity search via FAISS, re-ranked with a soft penalty for chunks far from the viewer's current timestamp, so answers stay relevant to where the user actually is in the video, not just wherever similar words appear.

---

## Features

- Answers grounded only in the video's transcript (no hallucinated general knowledge)
- Timestamp-aware retrieval that prioritizes context near your current playback position
- Multi-turn conversation: follow-up questions ("what's *its* time complexity?") resolve correctly using chat history
- Clickable timestamps in answers that jump the video to that exact moment
- Per-video caching (memory + disk), so repeat visits don't re-index
- Handles disabled/missing captions, invalid video IDs, empty transcripts, and API rate limits
- Multi-language transcript fallback (tries English/Hindi first, falls back to whatever is available)

---

## Project structure

```
VidMentor/
├── rag-backend/
│   ├── main.py              # FastAPI app and routes only
│   ├── config.py            # constants: models, window size, top-k, cache dir
│   ├── transcript.py        # fetch transcripts, language fallback
│   ├── chunking.py          # time-window chunking
│   ├── embeddings.py        # shared embedding model
│   ├── indexer.py           # build/load FAISS index per video
│   ├── cache.py             # memory + disk cache
│   ├── retrieval.py         # similarity search + time penalty
│   ├── prompts.py           # prompt building
│   ├── llm.py               # Groq client + retry
│   ├── schemas.py           # request models
│   ├── tests/               # pytest tests
│   ├── requirements.txt
│   ├── requirements-dev.txt # pytest (dev only)
│   ├── .env.example         # template for your API key
│   └── rag_cache/           # auto-created, cached indexes (gitignored)
│
└── yt-doubt-extension/
    ├── manifest.json        # Manifest V3 config
    ├── background.js        # service worker, calls the backend
    ├── content.js           # injects chat widget on YouTube pages
    ├── content.css          # widget styling
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

Create your `.env` from the template and add your key:

```bash
copy .env.example .env           # Windows
# cp .env.example .env           # Mac/Linux
```

```
GROQ_API_KEY=your_groq_api_key_here
GROQ_TIMEOUT_SEC=30.0

# Optional overrides:
# MAX_HISTORY_MESSAGES=20   # per-video chat history cap (even numbers only; defaults to 20)
# MAX_CHAT_SESSIONS=100     # LRU store cap for concurrent sessions (defaults to 100)
# REWRITE_ENABLED=1         # 0 turns off the follow-up question rewrite
# REWRITE_TIMEOUT_SEC=8.0   # timeout for the rewrite call (defaults to 8.0)
```

(Free key from [console.groq.com](https://console.groq.com/keys))

`GROQ_TIMEOUT_SEC` configures the per-request Groq API timeout (in seconds) and defaults to `30.0`.
Follow-up questions are rewritten into a standalone question before the transcript search (so "what's its time complexity?" finds the right part of the video). The rewrite is skipped when the chat has no history; with history it runs for short questions (under 8 words) or ones with a reference word like "it", "that" or "iska". Only the search uses the rewritten text; the prompt and history keep your original question, and on any error the original is used. `REWRITE_TIMEOUT_SEC` is the timeout for that extra Groq call (default `8.0`) and `REWRITE_ENABLED=0` turns it off.
`MAX_HISTORY_MESSAGES` caps how many messages of chat history are kept per video session (rounded down to the nearest even number so whole exchanges stay intact; default `20`), and `MAX_CHAT_SESSIONS` caps how many video sessions are held in memory at once — the least recently used session is evicted when the store is full (default `100`).

Run it:

```bash
uvicorn main:app --reload --port 8000
```

Verify: open `http://localhost:8000/health`, which should return `{"status": "ok"}`.
Interactive API docs: `http://localhost:8000/docs`

### Extension

1. Open `chrome://extensions`
2. Enable **Developer mode** (top right)
3. Click **Load unpacked**
4. Select the `yt-doubt-extension` folder

Go to any YouTube video and look for the **"❓ Ask Doubt"** button in the bottom-right corner.

> **Local development:** set `BACKEND_URL` in `yt-doubt-extension/background.js` to `http://localhost:8000`. `manifest.json` already allows both localhost and the deployed backend in `host_permissions`. Please do not commit the localhost URL.

### Running tests

```bash
cd rag-backend
pip install -r requirements-dev.txt
python -m pytest
```

The tests cover chunking, retrieval ranking (including the timestamp penalty), prompt building, and empty-transcript handling. They run in under a second and need no API key or model download.

---

## API Reference

### `POST /index`
Pre-builds the index for a video (optional, `/chat` builds it automatically on first use).
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

---

## Known limitations

- **Transcript fetching can fail on cloud hosts.** YouTube blocks many cloud IPs (including Render's free tier), so the deployed backend may return "could not retrieve a transcript". It works from a normal home connection when you run the backend locally. A client-side transcript fetch would avoid this and is a good contribution.
- **Embedding model:** By default, `all-MiniLM-L6-v2` is used for lightweight memory footprint (~200MB, well within the 512MB free-tier RAM limit). For Hindi transcripts, `EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2` can be opted into via `.env` (note: sits at ~445MB RAM; Roman-script Hinglish was not tested).
- **Chat history is kept per session and video**, where a session is one id per browser profile (stored in `chrome.storage.local`), and it resets on restart.
- **Free-tier hosting:** the server spins down after ~15 minutes of inactivity (first request after idle takes ~30-50s), and the disk is ephemeral, so cached indexes reset on redeploy/restart.

---

## Roadmap

- Overlapping chunk windows (reduce context loss at chunk boundaries)
- Multilingual model optimization for low-memory cloud hosts
- Client-side transcript fetching (avoids YouTube IP blocking)
- Move cache storage to Redis or a persistent volume
- Voice input for asking doubts
- Auto-pause video when the chat panel opens

See the [open issues](../../issues) for tasks you can pick up.

---

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first. It covers setup, how to pick an issue, and what a good pull request looks like. Issues are labeled `good first issue`, `intermediate`, and `advanced`.

---

## License

MIT, see [LICENSE](LICENSE).