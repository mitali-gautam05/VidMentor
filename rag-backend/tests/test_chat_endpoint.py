import httpx
import pytest
from fastapi.testclient import TestClient
from groq import APITimeoutError, RateLimitError, BadRequestError

import main
from config import CACHE_DIR
from schemas import ChatRequest


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
        json={"video_id": "vid_succe11", "query": "What is Python?", "current_time": 10.0},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["answer"] == "Python is a high-level programming language."

    # History is updated with the user query and assistant answer
    history = main.chat_sessions.get(("vid_succe11", "vid_succe11"), [])
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
        json={"video_id": "vid_timeo11", "query": "Why did this time out?", "current_time": 0.0},
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
    assert ("vid_timeo11", "vid_timeo11") not in main.chat_sessions or len(main.chat_sessions[("vid_timeo11", "vid_timeo11")]) == 0


def test_chat_rate_limit_failure_returns_503_and_preserves_history(client, monkeypatch):
    monkeypatch.setattr(main, "get_or_build_index", lambda vid: (fake_index_result(), None))
    monkeypatch.setattr(main, "retrieve", lambda query, chunks, idx, ct: chunks)

    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    resp = httpx.Response(429, request=req)
    rate_limit_err = RateLimitError("Rate limit exceeded", response=resp, body=None)

    monkeypatch.setattr(main, "call_groq_with_retry", lambda prompt: (_ for _ in ()).throw(rate_limit_err))

    response = client.post(
        "/chat",
        json={"video_id": "vid_rl_1111", "query": "Rate limited query"},
    )

    assert response.status_code == 503
    data = response.json()
    assert data.get("success") is False
    assert "error" in data
    assert ("vid_rl_1111", "vid_rl_1111") not in main.chat_sessions


def test_chat_upstream_error_returns_502_and_preserves_history(client, monkeypatch):
    monkeypatch.setattr(main, "get_or_build_index", lambda vid: (fake_index_result(), None))
    monkeypatch.setattr(main, "retrieve", lambda query, chunks, idx, ct: chunks)

    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    resp = httpx.Response(400, request=req)
    bad_req_err = BadRequestError("Invalid parameter", response=resp, body=None)

    monkeypatch.setattr(main, "call_groq_with_retry", lambda prompt: (_ for _ in ()).throw(bad_req_err))

    response = client.post(
        "/chat",
        json={"video_id": "vid_upstre1", "query": "Bad request query"},
    )

    assert response.status_code == 502
    data = response.json()
    assert data.get("success") is False
    assert "error" in data
    assert ("vid_upstre1", "vid_upstre1") not in main.chat_sessions


def test_chat_subsequent_request_succeeds_after_failed_attempt(client, monkeypatch):
    """Verify that a previous failed request leaves no residue in session history for future calls."""
    monkeypatch.setattr(main, "get_or_build_index", lambda vid: (fake_index_result(), None))
    monkeypatch.setattr(main, "retrieve", lambda query, chunks, idx, ct: chunks)

    # First call fails
    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    monkeypatch.setattr(main, "call_groq_with_retry", lambda prompt: (_ for _ in ()).throw(APITimeoutError(request=req)))
    r1 = client.post("/chat", json={"video_id": "vid_retryh1", "query": "First question that fails"})
    assert r1.status_code == 504

    # Second call succeeds
    monkeypatch.setattr(main, "call_groq_with_retry", lambda prompt: "Good answer")
    r2 = client.post("/chat", json={"video_id": "vid_retryh1", "query": "Second question that works"})
    assert r2.status_code == 200
    assert r2.json() == {"success": True, "answer": "Good answer"}

    # History should contain ONLY the second turn, not the first
    history = main.chat_sessions[("vid_retryh1", "vid_retryh1")]
    assert len(history) == 2
    assert history[0]["content"] == "Second question that works"
    assert history[1]["content"] == "Good answer"


def test_chat_invalid_video_id_returns_422(client, tmp_path, monkeypatch):
    """Invalid video_id (path traversal) returns 422 and creates no cache files."""
    monkeypatch.setattr("config.CACHE_DIR", str(tmp_path))

    response = client.post(
        "/chat",
        json={"video_id": "../x", "query": "test"},
    )

    assert response.status_code == 422
    data = response.json()
    assert "detail" in data

    index_files = list(tmp_path.glob("*.index"))
    pkl_files = list(tmp_path.glob("*_chunks.pkl"))
    assert len(index_files) == 0
    assert len(pkl_files) == 0


def test_chat_too_short_video_id_returns_422(client):
    """Too short video_id returns 422."""
    response = client.post(
        "/chat",
        json={"video_id": "short", "query": "test"},
    )

    assert response.status_code == 422


def test_chat_empty_video_id_returns_422(client):
    """Empty video_id returns 422."""
    response = client.post(
        "/chat",
        json={"video_id": "", "query": "test"},
    )

    assert response.status_code == 422


def test_index_invalid_video_id_returns_422(client, tmp_path, monkeypatch):
    """Invalid video_id on /index returns 422 and creates no cache files."""
    monkeypatch.setattr("config.CACHE_DIR", str(tmp_path))

    response = client.post(
        "/index",
        json={"video_id": "../x"},
    )

    assert response.status_code == 422
    data = response.json()
    assert "detail" in data

    index_files = list(tmp_path.glob("*.index"))
    pkl_files = list(tmp_path.glob("*_chunks.pkl"))
    assert len(index_files) == 0
    assert len(pkl_files) == 0

