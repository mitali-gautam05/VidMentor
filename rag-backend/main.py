import logging
from collections import OrderedDict
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import MAX_CHAT_SESSIONS, MAX_HISTORY_MESSAGES
from groq import APITimeoutError, RateLimitError

from indexer import get_or_build_index
from retrieval import retrieve
from llm import call_groq_with_retry
from prompts import build_prompt
from query_rewrite import rewrite_query
from schemas import IndexRequest, ChatRequest

logger = logging.getLogger(__name__)

app = FastAPI()

# Chrome extensions call from "chrome-extension://<id>" origin — allow that
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your extension's exact origin once you have the ID
    allow_methods=["*"],
    allow_headers=["*"],
)

# keyed by (session_id, video_id) so two users on the same video don't share history
chat_sessions: OrderedDict[tuple[str, str], list[dict]] = OrderedDict()


def _remember(key: tuple[str, str], query: str, answer: str) -> None:
    """Store one exchange, capping the session and evicting old sessions.

    Only the last `PROMPT_HISTORY_MESSAGES` messages feed the prompt anyway, so an
    unbounded session and an unbounded store are pure memory pressure.
    """
    history = chat_sessions.pop(key, [])
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": answer})
    if len(history) > MAX_HISTORY_MESSAGES:
        del history[: len(history) - MAX_HISTORY_MESSAGES]
    chat_sessions[key] = history
    while len(chat_sessions) > MAX_CHAT_SESSIONS:
        chat_sessions.popitem(last=False)


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Ensure consistent response contract for HTTP exceptions matching existing conventions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": str(exc.detail),
            "detail": str(exc.detail),
        },
    )


def _session_key(req: ChatRequest) -> tuple[str, str]:
    # older extension versions send no session_id, fall back to the video id
    return (req.session_id or req.video_id, req.video_id)


# ---------- Endpoints ----------
@app.post("/index")
def index_video(req: IndexRequest):
    result, error = get_or_build_index(req.video_id)
    if result is None:
        return {"success": False, "error": error}
    return {"success": True, "message": "Video indexed and ready."}


@app.post("/chat")
def chat(req: ChatRequest):
    result, error = get_or_build_index(req.video_id)
    if result is None:
        return {"success": False, "error": error}
    chunks, index = result

    key = _session_key(req)
    history = chat_sessions.get(key, [])
    # Only the search uses the rewritten question; the prompt and history keep the original.
    retrieved = retrieve(rewrite_query(req.query, history), chunks, index, req.current_time)
    prompt = build_prompt(retrieved, history, req.query)

    try:
        answer = call_groq_with_retry(prompt)
    except APITimeoutError as exc:
        logger.error("Groq request timed out for video %s: %s", req.video_id, exc)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Request to AI service timed out. Please try again.",
        )
    except RateLimitError as exc:
        logger.error("Groq rate limit reached for video %s: %s", req.video_id, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI service rate limit reached. Please try again later.",
        )
    except Exception as exc:
        logger.error("Groq request failed for video %s: %s", req.video_id, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI service temporarily unavailable. Please try again.",
        )

    _remember(key, req.query, answer)
    return {"success": True, "answer": answer}


@app.get("/video/{video_id}/status")
def video_status(video_id: str):
    result, error = get_or_build_index(video_id)
    if result is None:
        return {"success": False, "error": error}
    chunks, index = result
    duration = chunks[-1].get("end", 0.0) if chunks else 0.0
    return {
        "success": True,
        "video_id": video_id,
        "chunks_count": len(chunks),
        "duration_sec": duration,
    }


@app.get("/health")
def health():
    return {"status": "ok"}
