import numpy as np
import indexer


def test_empty_transcript_returns_friendly_error(monkeypatch):
    monkeypatch.setattr(indexer, "get_transcript", lambda video_id: ([], None))
    result, error = indexer.build_index_for_video("abc12345678")
    assert result is None
    assert error


def test_client_supplied_transcript_builds_index_without_get_transcript(monkeypatch):
    called = False

    def fail_get_transcript(vid):
        nonlocal called
        called = True
        return None, "Should not be called"

    dummy_embedder = type("DummyEmbedder", (), {
        "encode": lambda self, texts, **kwargs: np.ones((len(texts), 384), dtype=np.float32)
    })()
    monkeypatch.setattr(indexer, "embedder", dummy_embedder)
    monkeypatch.setattr(indexer, "get_transcript", fail_get_transcript)

    raw = [
        {"text": "Python is an interpreted programming language.", "start": 0.0, "duration": 5.0},
        {"text": "It supports multiple programming paradigms.", "start": 5.0, "duration": 5.0},
    ]
    result, error = indexer.get_or_build_index("client12345", raw)
    assert not called, "get_transcript should NOT be called when client supplies transcript"
    assert error is None
    assert result is not None
    chunks, idx = result
    assert len(chunks) > 0
    assert idx.ntotal > 0


def test_client_supplied_empty_transcript_returns_error(monkeypatch):
    called = False

    def fail_get_transcript(vid):
        nonlocal called
        called = True
        return None, "Should not be called"

    monkeypatch.setattr(indexer, "get_transcript", fail_get_transcript)
    result, error = indexer.get_or_build_index("clientempty", [])
    assert not called
    assert result is None
    assert "empty" in error.lower()
