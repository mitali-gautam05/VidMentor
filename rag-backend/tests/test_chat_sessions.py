import os
import pytest

# llm.py builds a Groq client at import time; a placeholder key is enough here
os.environ.setdefault("GROQ_API_KEY", "test-placeholder")

from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402


@pytest.fixture(autouse=True)
def client(monkeypatch):
    main.chat_sessions.clear()
    monkeypatch.setattr(main, "get_or_build_index", lambda vid: (([], None), None))
    monkeypatch.setattr(main, "retrieve", lambda *a, **k: [])
    monkeypatch.setattr(main, "call_groq_with_retry", lambda prompt: "an answer")
    yield TestClient(main.app)
    main.chat_sessions.clear()


def post(client, query, video_id="vid11111111", session_id=None):
    body = {"video_id": video_id, "query": query}
    if session_id is not None:
        body["session_id"] = session_id
    res = client.post("/chat", json=body)
    assert res.status_code == 200
    assert res.json()["success"] is True


def test_sessions_on_the_same_video_keep_separate_history(client):
    post(client, "from a", session_id="a")
    post(client, "from a again", session_id="a")
    post(client, "from b", session_id="b")

    assert len(main.chat_sessions[("a", "vid11111111")]) == 4
    assert len(main.chat_sessions[("b", "vid11111111")]) == 2
    assert main.chat_sessions[("b", "vid11111111")][0]["content"] == "from b"


def test_same_session_on_different_videos_is_separate(client):
    post(client, "q1", video_id="vid11111111", session_id="a")
    post(client, "q2", video_id="vid22222222", session_id="a")
    assert set(main.chat_sessions) == {("a", "vid11111111"), ("a", "vid22222222")}


def test_request_without_session_id_falls_back_to_video_id(client):
    post(client, "first")
    post(client, "second")
    assert list(main.chat_sessions) == [("vid11111111", "vid11111111")]
    assert len(main.chat_sessions[("vid11111111", "vid11111111")]) == 4


def test_overlong_session_id_is_rejected(client):
    res = client.post(
        "/chat", json={"video_id": "vid11111111", "query": "q", "session_id": "x" * 65}
    )
    assert res.status_code == 422
    assert main.chat_sessions == {}


def test_session_id_at_max_length_is_accepted(client):
    post(client, "q", session_id="x" * 64)
