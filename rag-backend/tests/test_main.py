import pytest
from collections import OrderedDict

import main
from config import MAX_CHAT_SESSIONS, MAX_HISTORY_MESSAGES


@pytest.fixture(autouse=True)
def empty_store():
    main.chat_sessions.clear()
    yield
    main.chat_sessions.clear()


def test_history_is_capped_per_session():
    for i in range(MAX_HISTORY_MESSAGES // 2 + 3):
        main._remember("video", f"q{i}", f"a{i}")
    history = main.chat_sessions["video"]
    assert len(history) <= MAX_HISTORY_MESSAGES
    # the most recent exchange is kept and the oldest one is dropped
    assert history[-1] == {"role": "assistant", "content": f"a{MAX_HISTORY_MESSAGES // 2 + 2}"}
    # whole exchanges, never a dangling half
    assert len(history) % 2 == 0
    # the oldest exchange was really dropped, not merely trimmed to parity
    assert history[0]["content"] != "q0"


def test_a_session_is_evicted_only_when_the_store_is_full():
    ids = [f"video-{i}" for i in range(MAX_CHAT_SESSIONS + 1)]
    for vid in ids:
        main._remember(vid, "q", "a")
    assert len(main.chat_sessions) == MAX_CHAT_SESSIONS
    assert "video-0" not in main.chat_sessions  # the oldest session is evicted
    assert "video-1" in main.chat_sessions


def test_an_old_session_is_evicted_not_the_recently_used_one():
    for i in range(MAX_CHAT_SESSIONS):
        main._remember(f"video-{i}", "q", "a")
    main._remember("video-0", "q", "a")  # video-0 becomes the most recently used
    main._remember("video-new", "q", "a")
    assert "video-0" in main.chat_sessions  # it survived the eviction
    assert "video-1" not in main.chat_sessions  # the next-oldest went instead
    assert isinstance(main.chat_sessions, OrderedDict)  # sanity: LRU by recency


def test_video_status_endpoint(monkeypatch):
    from fastapi.testclient import TestClient
    client = TestClient(main.app)

    fake_chunks = [{"text": "Sample text", "start": 0.0, "end": 45.0}]
    fake_idx = object()
    monkeypatch.setattr(main, "get_or_build_index", lambda vid: ((fake_chunks, fake_idx), None) if vid == "valid_vid" else (None, "Video not found"))

    r_ok = client.get("/video/valid_vid/status")
    assert r_ok.status_code == 200
    data = r_ok.json()
    assert data["success"] is True
    assert data["video_id"] == "valid_vid"
    assert data["chunks_count"] == 1
    assert data["duration_sec"] == 45.0

    r_err = client.get("/video/invalid_vid/status")
    assert r_err.status_code == 200
    assert r_err.json() == {"success": False, "error": "Video not found"}
