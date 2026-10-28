import httpx
import pytest
from fastapi.testclient import TestClient
from groq import APITimeoutError, RateLimitError, BadRequestError

import main
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
        json={"video_id": "vid_success_1", "query": "What is Python?", "current_time": 10.0},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["answer"] == "Python is a high-level programming language."

    # History is updated with the user query and assistant answer
    history = main.chat_sessions.get(("vid_success_1", "vid_success_1"), [])
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
        json={"video_id": "vid_timeout_1", "query": "Why did this time out?", "current_time": 0.0},
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
    assert ("vid_timeout_1", "vid_timeout_1") not in main.chat_sessions or len(main.chat_sessions[("vid_timeout_1", "vid_timeout_1")]) == 0


def test_chat_rate_limit_failure_returns_503_and_preserves_history(client, monkeypatch):
    monkeypatch.setattr(main, "get_or_build_index", lambda vid: (fake_index_result(), None))
    monkeypatch.setattr(main, "retrieve", lambda query, chunks, idx, ct: chunks)

    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    resp = httpx.Response(429, request=req)
    rate_limit_err = RateLimitError("Rate limit exceeded", response=resp, body=None)

    monkeypatch.setattr(main, "call_groq_with_retry", lambda prompt: (_ for _ in ()).throw(rate_limit_err))

    response = client.post(
        "/chat",
        json={"video_id": "vid_rl_1", "query": "Rate limited query"},
    )

    assert response.status_code == 503
    data = response.json()
    assert data.get("success") is False
    assert "error" in data
    assert ("vid_rl_1", "vid_rl_1") not in main.chat_sessions


def test_chat_upstream_error_returns_502_and_preserves_history(client, monkeypatch):
    monkeypatch.setattr(main, "get_or_build_index", lambda vid: (fake_index_result(), None))
    monkeypatch.setattr(main, "retrieve", lambda query, chunks, idx, ct: chunks)

    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    resp = httpx.Response(400, request=req)
    bad_req_err = BadRequestError("Invalid parameter", response=resp, body=None)

    monkeypatch.setattr(main, "call_groq_with_retry", lambda prompt: (_ for _ in ()).throw(bad_req_err))

    response = client.post(
        "/chat",
        json={"video_id": "vid_upstream_1", "query": "Bad request query"},
    )

    assert response.status_code == 502
    data = response.json()
    assert data.get("success") is False
    assert "error" in data
    assert ("vid_upstream_1", "vid_upstream_1") not in main.chat_sessions


def test_chat_subsequent_request_succeeds_after_failed_attempt(client, monkeypatch):
    """Verify that a previous failed request leaves no residue in session history for future calls."""
    monkeypatch.setattr(main, "get_or_build_index", lambda vid: (fake_index_result(), None))
    monkeypatch.setattr(main, "retrieve", lambda query, chunks, idx, ct: chunks)

    # First call fails
    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    monkeypatch.setattr(main, "call_groq_with_retry", lambda prompt: (_ for _ in ()).throw(APITimeoutError(request=req)))
    r1 = client.post("/chat", json={"video_id": "vid_retry_hist", "query": "First question that fails"})
    assert r1.status_code == 504

    # Second call succeeds
    monkeypatch.setattr(main, "call_groq_with_retry", lambda prompt: "Good answer")
    r2 = client.post("/chat", json={"video_id": "vid_retry_hist", "query": "Second question that works"})
    assert r2.status_code == 200
    assert r2.json() == {"success": True, "answer": "Good answer"}

    # History should contain ONLY the second turn, not the first
    history = main.chat_sessions[("vid_retry_hist", "vid_retry_hist")]
    assert len(history) == 2
    assert history[0]["content"] == "Second question that works"
    assert history[1]["content"] == "Good answer"



def test_chat_with_include_sources_returns_sources(client, monkeypatch):
    monkeypatch.setattr(main, "get_or_build_index", lambda vid: (fake_index_result(), None))
    monkeypatch.setattr(main, "retrieve", lambda query, chunks, idx, ct: chunks)
    monkeypatch.setattr(main, "call_groq_with_retry", lambda prompt: "Good answer")

    response = client.post(
        "/chat",
        json={"video_id": "vid_src_1", "query": "Test query", "include_sources": True},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["answer"] == "Good answer"
    assert "sources" in data
    assert len(data["sources"]) == 1
    assert data["sources"][0]["start"] == 0
    assert data["sources"][0]["end"] == 30
