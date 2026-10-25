import faiss
import numpy as np
import pickle
import cache


def test_save_then_load_returns_same_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path))

    video_id = "testvideo11"
    chunks = [
        {"text": "Hello world", "start": 0},
        {"text": "Testing cache", "start": 10},
    ]

    index = faiss.IndexFlatIP(4)
    index.add(np.array([[1.0, 0.0, 0.0, 0.0]], dtype="float32"))

    cache.save_to_disk(video_id, chunks, index)
    result = cache.load_from_disk(video_id)

    assert result is not None

    loaded_chunks, loaded_index = result
    assert loaded_chunks == chunks
    assert loaded_index.d == 4
    assert loaded_index.ntotal == 1


def test_load_unsaved_video_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path))

    result = cache.load_from_disk("neversaved1")

    assert result is None


def test_get_cache_key_rejects_invalid_video_id():
    # Test that get_cache_key rejects invalid video IDs
    import pytest
    with pytest.raises(ValueError, match="Invalid video_id"):
        cache.get_cache_key("../x")
    with pytest.raises(ValueError, match="Invalid video_id"):
        cache.get_cache_key("short")
    with pytest.raises(ValueError, match="Invalid video_id"):
        cache.get_cache_key("a" * 12)
    with pytest.raises(ValueError, match="Invalid video_id"):
        cache.get_cache_key("")
    with pytest.raises(ValueError, match="Invalid video_id"):
        cache.get_cache_key("abc/def")
    with pytest.raises(ValueError, match="Invalid video_id"):
        cache.get_cache_key("abc def")


def test_cache_does_not_reuse_old_model_index(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path))

    video_id = "testvidmod1"
    chunks = [{"text": "Hello world", "start": 0}]
    index = faiss.IndexFlatIP(4)
    index.add(np.array([[1.0, 0.0, 0.0, 0.0]], dtype="float32"))

    cache.save_to_disk(video_id, chunks, index, model_name="all-MiniLM-L6-v2")

    result = cache.load_from_disk(video_id, model_name="paraphrase-multilingual-MiniLM-L12-v2")
    assert result is None

    old_result = cache.load_from_disk(video_id, model_name="all-MiniLM-L6-v2")
    assert old_result is not None


def test_cache_key_includes_chunk_settings(monkeypatch):
    monkeypatch.setattr(cache, "WINDOW_SEC", 40)
    monkeypatch.setattr(cache, "OVERLAP_SEC", 10)
    assert cache.get_cache_key("vid11111111", model_name="m").endswith("_w40_o10")


def test_changing_chunk_settings_does_not_reuse_old_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(cache, "WINDOW_SEC", 40)
    monkeypatch.setattr(cache, "OVERLAP_SEC", 0)

    chunks = [{"text": "Hello world", "start": 0}]
    index = faiss.IndexFlatIP(4)
    index.add(np.array([[1.0, 0.0, 0.0, 0.0]], dtype="float32"))
    cache.save_to_disk("vid11111111", chunks, index)

    monkeypatch.setattr(cache, "OVERLAP_SEC", 10)
    assert cache.load_from_disk("vid11111111") is None

    monkeypatch.setattr(cache, "OVERLAP_SEC", 0)
    monkeypatch.setattr(cache, "WINDOW_SEC", 60)
    assert cache.load_from_disk("vid11111111") is None

    monkeypatch.setattr(cache, "WINDOW_SEC", 40)
    assert cache.load_from_disk("vid11111111") is not None