import faiss
import numpy as np
import cache
import os


def test_save_then_load_returns_same_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path))

    video_id = "test_video"
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

    result = cache.load_from_disk("never_saved")

    assert result is None


def test_cache_does_not_reuse_old_model_index(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path))

    video_id = "test_video_model"
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
    assert cache.get_cache_key("vid", model_name="m").endswith("_w40_o10")


def test_changing_chunk_settings_does_not_reuse_old_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(cache, "WINDOW_SEC", 40)
    monkeypatch.setattr(cache, "OVERLAP_SEC", 0)

    chunks = [{"text": "Hello world", "start": 0}]
    index = faiss.IndexFlatIP(4)
    index.add(np.array([[1.0, 0.0, 0.0, 0.0]], dtype="float32"))
    cache.save_to_disk("vid", chunks, index)

    monkeypatch.setattr(cache, "OVERLAP_SEC", 10)
    assert cache.load_from_disk("vid") is None

    monkeypatch.setattr(cache, "OVERLAP_SEC", 0)
    monkeypatch.setattr(cache, "WINDOW_SEC", 60)
    assert cache.load_from_disk("vid") is None

    monkeypatch.setattr(cache, "WINDOW_SEC", 40)
    assert cache.load_from_disk("vid") is not None


def test_corrupt_cache_files_are_purged_and_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path))
    video_id = "corrupt_video"
    key = cache.get_cache_key(video_id)
    idx_path = tmp_path / f"{key}.index"
    chunks_path = tmp_path / f"{key}_chunks.json"

    idx_path.write_bytes(b"invalid faiss header data")
    chunks_path.write_bytes(b"corrupted json payload")

    result = cache.load_from_disk(video_id)
    assert result is None
    assert not idx_path.exists()
    assert not chunks_path.exists()


def test_hindi_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path))

    video_id = "test_hindi"
    chunks = [
        {"text": "नमस्ते दुनिया", "start": 0},
        {"text": "यह एक परीक्षण है", "start": 10},
    ]

    index = faiss.IndexFlatIP(4)
    index.add(np.array([[1.0, 0.0, 0.0, 0.0]], dtype="float32"))

    cache.save_to_disk(video_id, chunks, index)

    key = cache.get_cache_key(video_id)
    json_path = tmp_path / f"{key}_chunks.json"
    content = json_path.read_text(encoding="utf-8")
    assert "नमस्ते दुनिया" in content
    assert "यह एक परीक्षण है" in content
    assert "\\u" not in content

    result = cache.load_from_disk(video_id)
    assert result is not None
    loaded_chunks, _ = result
    assert loaded_chunks == chunks


def test_load_invalid_json_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path))

    key = cache.get_cache_key("test_bad_json")
    index = faiss.IndexFlatIP(4)
    index.add(np.array([[1.0, 0.0, 0.0, 0.0]], dtype="float32"))
    faiss.write_index(index, f"{tmp_path}/{key}.index")
    (tmp_path / f"{key}_chunks.json").write_text("not valid json", encoding="utf-8")

    assert cache.load_from_disk("test_bad_json") is None


def test_load_non_utf8_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path))

    key = cache.get_cache_key("test_bad_bytes")
    index = faiss.IndexFlatIP(4)
    index.add(np.array([[1.0, 0.0, 0.0, 0.0]], dtype="float32"))
    faiss.write_index(index, f"{tmp_path}/{key}.index")
    (tmp_path / f"{key}_chunks.json").write_bytes(b"\xff\xfe not utf-8")

    assert cache.load_from_disk("test_bad_bytes") is None


def test_load_missing_chunks_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path))

    key = cache.get_cache_key("test_missing_chunks")
    index = faiss.IndexFlatIP(4)
    index.add(np.array([[1.0, 0.0, 0.0, 0.0]], dtype="float32"))
    faiss.write_index(index, f"{tmp_path}/{key}.index")

    assert cache.load_from_disk("test_missing_chunks") is None


def test_load_missing_index_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path))

    key = cache.get_cache_key("test_missing_index")
    (tmp_path / f"{key}_chunks.json").write_text('[{"text": "test", "start": 0}]', encoding="utf-8")

    assert cache.load_from_disk("test_missing_index") is None


def test_load_invalid_chunks_structure_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path))

    key = cache.get_cache_key("test_bad_structure")
    index = faiss.IndexFlatIP(4)
    index.add(np.array([[1.0, 0.0, 0.0, 0.0]], dtype="float32"))
    faiss.write_index(index, f"{tmp_path}/{key}.index")
    (tmp_path / f"{key}_chunks.json").write_text('[{"text": "missing start"}]', encoding="utf-8")

    assert cache.load_from_disk("test_bad_structure") is None


def test_legacy_pickle_file_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path))

    video_id = "test_legacy"
    chunks = [{"text": "old pickle data", "start": 0}]
    index = faiss.IndexFlatIP(4)
    index.add(np.array([[1.0, 0.0, 0.0, 0.0]], dtype="float32"))

    key = cache.get_cache_key(video_id)
    faiss.write_index(index, f"{tmp_path}/{key}.index")
    import pickle
    with open(f"{tmp_path}/{key}_chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    result = cache.load_from_disk(video_id)
    assert result is None

    cache.save_to_disk(video_id, [{"text": "new json data", "start": 5}], index)
    result = cache.load_from_disk(video_id)
    assert result is not None
    loaded_chunks, _ = result
    assert loaded_chunks == [{"text": "new json data", "start": 5}]
