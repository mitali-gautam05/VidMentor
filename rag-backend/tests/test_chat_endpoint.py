import httpx
import pytest
from fastapi.testclient import TestClient
from groq import APITimeoutError, RateLimitError, BadRequestError

import main
from schemas import ChatRequest
from cache import get_cache_key

VALID_ID_1 = "vidsucces12"
VALID_ID_2 = "vidtimeou12"
VALID_ID_3 = "vidrl123456"
VALID_ID_4 = "vidupstram1"
VALID_ID_5 = "vidretryhis"


@pytest.fixture(autouse=True)
def clean_chat_sessions():
    """Ensure chat_sessions dictionary is cleaned between test runs."""
    main.chat_sessions.clear()
    yield
    main.chat_sessions.clear()


@pytest.fixture
def client():
    return TestClient(main.app)


def fake_index_result():
    chunks = [{"text": "Video content about Python programming", "start": 0, "end": 30}]
    index = object()  # dummy index
    return chunks, index


# 9. Successful /chat responses preserve the existing response contract
def test_chat_success_preserves_response_contract(client, monkeypatch):
    monkeypatch.setattr(main, "get_or_build_index", lambda vid: (fake_index_result(), None))
    monkeypatch.setattr(main, "retrieve", lambda query, chunks, idx, ct: chunks)
    monkeypatch.setattr(main, "call_groq_with_retry", lambda prompt: "Python is a high-level programming language.")

    response = client.post(
        "/chat",
        json={"video_id": VALID_ID_1, "query": "What is Python?", "current_time": 10.0},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["answer"] == "Python is a high-level programming language."

    # History is updated with the user query and assistant answer
    history = main.chat_sessions.get((VALID_ID_1, VALID_ID_1), [])
    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "What is Python?"}
    assert history[1] == {"role": "assistant", "content": "Python is a high-level programming language."}


# 8. /chat does not return success: true when the Groq call fails (timeout -> 504)
# 10. Failed LLM requests do not contaminate chat history
def test_chat_timeout_failure_returns_504_and_does_not_pollute_history(client, monkeypatch):
    monkeypatch.setattr(main, "get_or_build_index", lambda vid: (fake_index_result(), None))
    monkeypatch.setattr(main, "retrieve", lambda query, chunks, idx, ct: chunks)

    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    timeout_err = APITimeoutError(request=req)

    def mock_fail(prompt):
        raise timeout_err

    monkeypatch.setattr(main, "call_groq_with_retry", mock_fail)

    response = client.post(
        "/chat",
        json={"video_id": VALID_ID_2, "query": "Why did this time out?", "current_time": 0.0},
    )

    # 1. Non-success HTTP status
    assert response.status_code == 504
    data = response.json()

    # 2. Does NOT return success: true
    assert data.get("success") is False
    assert "error" in data
    assert "detail" in data
    # Sensitive provider/API info is not exposed
    assert "https://api.groq.com" not in data["error"]

    # 3. Chat history is NOT contaminated with the failed prompt or error message
    assert (VALID_ID_2, VALID_ID_2) not in main.chat_sessions or len(main.chat_sessions[(VALID_ID_2, VALID_ID_2)]) == 0


def test_chat_rate_limit_failure_returns_503_and_preserves_history(client, monkeypatch):
    monkeypatch.setattr(main, "get_or_build_index", lambda vid: (fake_index_result(), None))
    monkeypatch.setattr(main, "retrieve", lambda query, chunks, idx, ct: chunks)

    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    resp = httpx.Response(429, request=req)
    rate_limit_err = RateLimitError("Rate limit exceeded", response=resp, body=None)

    monkeypatch.setattr(main, "call_groq_with_retry", lambda prompt: (_ for _ in ()).throw(rate_limit_err))

    response = client.post(
        "/chat",
        json={"video_id": VALID_ID_3, "query": "Rate limited query"},
    )

    assert response.status_code == 503
    data = response.json()
    assert data.get("success") is False
    assert "error" in data
    assert (VALID_ID_3, VALID_ID_3) not in main.chat_sessions


def test_chat_upstream_error_returns_502_and_preserves_history(client, monkeypatch):
    monkeypatch.setattr(main, "get_or_build_index", lambda vid: (fake_index_result(), None))
    monkeypatch.setattr(main, "retrieve", lambda query, chunks, idx, ct: chunks)

    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    resp = httpx.Response(400, request=req)
    bad_req_err = BadRequestError("Invalid parameter", response=resp, body=None)

    monkeypatch.setattr(main, "call_groq_with_retry", lambda prompt: (_ for _ in ()).throw(bad_req_err))

    response = client.post(
        "/chat",
        json={"video_id": VALID_ID_4, "query": "Bad request query"},
    )

    assert response.status_code == 502
    data = response.json()
    assert data.get("success") is False
    assert "error" in data
    assert (VALID_ID_4, VALID_ID_4) not in main.chat_sessions


def test_chat_subsequent_request_succeeds_after_failed_attempt(client, monkeypatch):
    """Verify that a previous failed request leaves no residue in session history for future calls."""
    monkeypatch.setattr(main, "get_or_build_index", lambda vid: (fake_index_result(), None))
    monkeypatch.setattr(main, "retrieve", lambda query, chunks, idx, ct: chunks)

    # First call fails
    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    monkeypatch.setattr(main, "call_groq_with_retry", lambda prompt: (_ for _ in ()).throw(APITimeoutError(request=req)))
    r1 = client.post("/chat", json={"video_id": VALID_ID_5, "query": "First question that fails"})
    assert r1.status_code == 504

    # Second call succeeds
    monkeypatch.setattr(main, "call_groq_with_retry", lambda prompt: "Good answer")
    r2 = client.post("/chat", json={"video_id": VALID_ID_5, "query": "Second question that works"})
    assert r2.status_code == 200
    assert r2.json() == {"success": True, "answer": "Good answer"}

    # History should contain ONLY the second turn, not the first
    history = main.chat_sessions[(VALID_ID_5, VALID_ID_5)]
    assert len(history) == 2
    assert history[0]["content"] == "Second question that works"
    assert history[1]["content"] == "Good answer"


# --- video_id validation tests ---

INVALID_VIDEO_IDS = [
    "../x",
    "",
    "short",
    "012345678901",
    "abc def ghi",
    "abc/def/ghi",
    "abc.def.ghi",
    "abc@def#ghi",
    "a/../../b/c",  # 11 chars with slashes - tests character rule directly
]


@pytest.mark.parametrize("bad_id", INVALID_VIDEO_IDS)
def test_chat_invalid_video_id_returns_422(client, bad_id):
    """Invalid video_ids should return 422 before any route logic runs."""
    response = client.post("/chat", json={"video_id": bad_id, "query": "test"})
    assert response.status_code == 422
    data = response.json()
    # FastAPI validation error format has "detail" field with error details
    assert "detail" in data


@pytest.mark.parametrize("bad_id", INVALID_VIDEO_IDS)
def test_index_invalid_video_id_returns_422(client, bad_id):
    """Invalid video_ids should return 422 for /index endpoint."""
    response = client.post("/index", json={"video_id": bad_id})
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_get_cache_key_valid_id():
    """get_cache_key should accept valid YouTube IDs."""
    key = get_cache_key("dQw4w9WgXcQ")
    assert key.startswith("dQw4w9WgXcQ_")


@pytest.mark.parametrize("bad_id", INVALID_VIDEO_IDS)
def test_get_cache_key_invalid_id_raises(bad_id):
    """get_cache_key should reject invalid IDs to prevent path traversal."""
    with pytest.raises(ValueError) as exc_info:
        get_cache_key(bad_id)
    assert "Invalid video_id format" in str(exc_info.value)


def test_invalid_chat_request_creates_no_cache_files(client, tmp_path, monkeypatch):
    """Invalid video_id should not create any cache files on disk."""
    import cache
    import os

    # Patch cache.CACHE_DIR to use tmp_path
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path))

    # Track if get_or_build_index is called
    index_called = {"called": False}

    def tracking_index_builder(vid):
        index_called["called"] = True
        return (None, "error")

    monkeypatch.setattr("main.get_or_build_index", tracking_index_builder)

    response = client.post("/chat", json={"video_id": "../x", "query": "test"})
    assert response.status_code == 422

    # Index builder should never be called for invalid video_id
    assert not index_called["called"], "get_or_build_index was called with invalid video_id"

    # No cache files should be created in tmp_path
    cache_files = list(tmp_path.iterdir())
    assert len(cache_files) == 0, f"Unexpected cache files created: {cache_files}"


def test_invalid_index_request_creates_no_cache_files(client, tmp_path, monkeypatch):
    """Invalid video_id on /index should not create any cache files on disk."""
    import cache
    import os

    # Patch cache.CACHE_DIR to use tmp_path
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path))

    # Track if get_or_build_index is called
    index_called = {"called": False}

    def tracking_index_builder(vid):
        index_called["called"] = True
        return (None, "error")

    monkeypatch.setattr("main.get_or_build_index", tracking_index_builder)

    response = client.post("/index", json={"video_id": "../x"})
    assert response.status_code == 422

    # Index builder should never be called for invalid video_id
    assert not index_called["called"], "get_or_build_index was called with invalid video_id"

    # No cache files should be created in tmp_path
    cache_files = list(tmp_path.iterdir())
    assert len(cache_files) == 0, f"Unexpected cache files created: {cache_files}"



def test_chat_accepts_client_transcript(client, monkeypatch):
    captured = {}
    def mock_get_or_build(vid, raw_transcript=None):
        captured["vid"] = vid
        captured["transcript"] = raw_transcript
        return fake_index_result(), None

    monkeypatch.setattr(main, "get_or_build_index", mock_get_or_build)
    monkeypatch.setattr(main, "retrieve", lambda query, chunks, idx, ct: chunks)
    monkeypatch.setattr(main, "call_groq_with_retry", lambda prompt: "Answer with client transcript.")

    client_transcript = [
        {"text": "First segment from client", "start": 0.0, "duration": 5.0},
        {"text": "Second segment from client", "start": 5.0, "duration": 5.0},
    ]

    response = client.post(
        "/chat",
        json={
            "video_id": VALID_ID_1,
            "query": "Tell me about this video",
            "transcript": client_transcript,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["answer"] == "Answer with client transcript."
    assert captured["vid"] == VALID_ID_1
    assert captured["transcript"] == client_transcript


def test_index_accepts_client_transcript(client, monkeypatch):
    captured = {}
    def mock_get_or_build(vid, raw_transcript=None):
        captured["vid"] = vid
        captured["transcript"] = raw_transcript
        return fake_index_result(), None

    monkeypatch.setattr(main, "get_or_build_index", mock_get_or_build)

    client_transcript = [
        {"text": "Indexing segment from browser", "start": 1.0, "duration": 3.0}
    ]

    response = client.post(
        "/index",
        json={
            "video_id": VALID_ID_1,
            "transcript": client_transcript,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["message"] == "Video indexed and ready."
    assert captured["vid"] == VALID_ID_1
    assert captured["transcript"] == client_transcript
