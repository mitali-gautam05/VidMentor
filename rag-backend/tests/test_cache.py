import faiss
import numpy as np
import cache
import os
import time

VALID_ID = "testvid1234"
VALID_ID_2 = "testvid1235"
VALID_ID_3 = "testvid1236"
VALID_ID_4 = "testvid1237"
VALID_ID_5 = "testvid1238"
VALID_ID_6 = "testvid1239"
VALID_ID_7 = "testvid1240"
VALID_ID_8 = "testvid1241"
VALID_ID_9 = "tstmisschnk"
VALID_ID_10 = "tstmissndx1"
VALID_ID_11 = "tstbdstrct1"
VALID_ID_12 = "tstlgcy1234"
VALID_ID_13 = "atmcchk1234"
VALID_ID_14 = "atmcflr1234"
VALID_ID_15 = "atmctrn1234"
VALID_ID_16 = "atmcmis1234"
VALID_ID_17 = "clntmp12345"


def test_save_then_load_returns_same_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path))

    video_id = VALID_ID
    chunks = [
        {"text": "Hello world", "start": 0},
        {"text": "Testing cache", "start": 10},
    ]

    index = faiss.IndexFlatIP(4)
    index.add(np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype="float32"))

    cache.save_to_disk(video_id, chunks, index)
    result = cache.load_from_disk(video_id)

    assert result is not None

    loaded_chunks, loaded_index = result
    assert loaded_chunks == chunks
    assert loaded_index.d == 4
    assert loaded_index.ntotal == 2


def test_load_unsaved_video_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path))

    result = cache.load_from_disk(VALID_ID_2)

    assert result is None


def test_cache_does_not_reuse_old_model_index(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path))

    video_id = VALID_ID_3
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
    assert cache.get_cache_key(VALID_ID_4, model_name="m").endswith("_w40_o10")


def test_changing_chunk_settings_does_not_reuse_old_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(cache, "WINDOW_SEC", 40)
    monkeypatch.setattr(cache, "OVERLAP_SEC", 0)

    chunks = [{"text": "Hello world", "start": 0}]
    index = faiss.IndexFlatIP(4)
    index.add(np.array([[1.0, 0.0, 0.0, 0.0]], dtype="float32"))
    cache.save_to_disk(VALID_ID_5, chunks, index)

    monkeypatch.setattr(cache, "OVERLAP_SEC", 10)
    assert cache.load_from_disk(VALID_ID_5) is None

    monkeypatch.setattr(cache, "OVERLAP_SEC", 0)
    monkeypatch.setattr(cache, "WINDOW_SEC", 60)
    assert cache.load_from_disk(VALID_ID_5) is None

    monkeypatch.setattr(cache, "WINDOW_SEC", 40)
    assert cache.load_from_disk(VALID_ID_5) is not None


def test_hindi_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path))

    video_id = VALID_ID_6
    chunks = [
        {"text": "नमस्ते दुनिया", "start": 0},
        {"text": "यह एक परीक्षण है", "start": 10},
    ]

    index = faiss.IndexFlatIP(4)
    index.add(np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype="float32"))

    cache.save_to_disk(video_id, chunks, index)

    key = cache.get_cache_key(video_id)
    json_path = tmp_path / f"{key}_chunks.json"
    content = json_path.read_text(encoding="utf-8")
    assert "नमस्ते दुनिया" in content
    assert "यह एक परीक्षण है" in content
    assert "\\u" not in content

    result = cache.load_from_disk(video_id)
    assert result is not None
    loaded_chunks, loaded_index = result
    assert loaded_chunks == chunks
    assert loaded_index.ntotal == 2


def test_load_invalid_json_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path))

    key = cache.get_cache_key(VALID_ID_7)
    index = faiss.IndexFlatIP(4)
    index.add(np.array([[1.0, 0.0, 0.0, 0.0]], dtype="float32"))
    faiss.write_index(index, f"{tmp_path}/{key}.index")
    (tmp_path / f"{key}_chunks.json").write_text("not valid json", encoding="utf-8")

    assert cache.load_from_disk(VALID_ID_7) is None


def test_load_non_utf8_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path))

    key = cache.get_cache_key(VALID_ID_8)
    index = faiss.IndexFlatIP(4)
    index.add(np.array([[1.0, 0.0, 0.0, 0.0]], dtype="float32"))
    faiss.write_index(index, f"{tmp_path}/{key}.index")
    (tmp_path / f"{key}_chunks.json").write_bytes(b"\xff\xfe not utf-8")

    assert cache.load_from_disk(VALID_ID_8) is None


def test_load_missing_chunks_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path))

    key = cache.get_cache_key(VALID_ID_9)
    index = faiss.IndexFlatIP(4)
    index.add(np.array([[1.0, 0.0, 0.0, 0.0]], dtype="float32"))
    faiss.write_index(index, f"{tmp_path}/{key}.index")

    assert cache.load_from_disk(VALID_ID_9) is None


def test_load_missing_index_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path))

    key = cache.get_cache_key(VALID_ID_10)
    (tmp_path / f"{key}_chunks.json").write_text('[{"text": "test", "start": 0}]', encoding="utf-8")

    assert cache.load_from_disk(VALID_ID_10) is None


def test_load_invalid_chunks_structure_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path))

    key = cache.get_cache_key(VALID_ID_11)
    index = faiss.IndexFlatIP(4)
    index.add(np.array([[1.0, 0.0, 0.0, 0.0]], dtype="float32"))
    faiss.write_index(index, f"{tmp_path}/{key}.index")
    (tmp_path / f"{key}_chunks.json").write_text('[{"text": "missing start"}]', encoding="utf-8")

    assert cache.load_from_disk(VALID_ID_11) is None


def test_legacy_pickle_file_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path))

    video_id = VALID_ID_12
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


def test_atomic_save_and_mismatched_pair_detection(tmp_path, monkeypatch):
    """Test that a stale or mismatched file pair (from an interrupted write) is treated as a cache miss."""
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path))

    video_id = VALID_ID_13
    chunks = [{"text": "Hello world", "start": 0}]
    index = faiss.IndexFlatIP(4)
    index.add(np.array([[1.0, 0.0, 0.0, 0.0]], dtype="float32"))

    cache.save_to_disk(video_id, chunks, index)
    assert cache.load_from_disk(video_id) is not None

    key = cache.get_cache_key(video_id)
    chunks_path = tmp_path / f"{key}_chunks.json"

    # Simulate an interrupted write by artificially offsetting chunks timestamp by 100 seconds
    stale_time = time.time() - 100
    os.utime(str(chunks_path), (stale_time, stale_time))

    # Mismatched timestamps must be safely ignored as a cache miss
    assert cache.load_from_disk(video_id) is None


def test_load_mismatched_vector_and_chunk_count_returns_none(tmp_path, monkeypatch):
    """Test that if index.ntotal does not match len(chunks), it is treated as a cache miss."""
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path))

    video_id = VALID_ID_16
    key = cache.get_cache_key(video_id)

    # chunks has 2 entries
    chunks = [
        {"text": "Chunk 1", "start": 0},
        {"text": "Chunk 2", "start": 10},
    ]
    # index has only 1 vector
    index = faiss.IndexFlatIP(4)
    index.add(np.array([[1.0, 0.0, 0.0, 0.0]], dtype="float32"))

    # Write files manually to simulate a mismatched pair
    import json
    chunks_path = tmp_path / f"{key}_chunks.json"
    idx_path = tmp_path / f"{key}.index"

    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f)
    faiss.write_index(index, str(idx_path))

    # Synchronize timestamps so mtime check passes
    now = time.time()
    os.utime(str(chunks_path), (now, now))
    os.utime(str(idx_path), (now, now))

    # Should detect count mismatch (index.ntotal=1 != len(chunks)=2) and return None
    assert cache.load_from_disk(video_id) is None


def test_cleanup_stale_tmp_files(tmp_path, monkeypatch):
    """Verify that cleanup_stale_tmp_files deletes old .tmp files and preserves valid/fresh files."""
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path))

    old_tmp_1 = tmp_path / "cache_entry_1.tmp.1234.uuid1"
    old_tmp_2 = tmp_path / "cache_entry_2.tmp"
    fresh_tmp = tmp_path / "cache_entry_3.tmp.5678.uuid2"
    valid_file = tmp_path / "cache_entry_valid.index"

    old_tmp_1.write_text("old temp data 1")
    old_tmp_2.write_text("old temp data 2")
    fresh_tmp.write_text("fresh temp data")
    valid_file.write_text("valid index data")

    # Mark old files as 1000 seconds old
    old_time = time.time() - 1000
    os.utime(str(old_tmp_1), (old_time, old_time))
    os.utime(str(old_tmp_2), (old_time, old_time))

    cleaned = cache.cleanup_stale_tmp_files(str(tmp_path), stale_age_sec=300.0)
    assert cleaned == 2
    assert not old_tmp_1.exists()
    assert not old_tmp_2.exists()
    assert fresh_tmp.exists()
    assert valid_file.exists()

    # With stale_age_sec=0, fresh_tmp is also cleaned, but valid_file remains untouched
    cleaned_all = cache.cleanup_stale_tmp_files(str(tmp_path), stale_age_sec=0)
    assert cleaned_all == 1
    assert not fresh_tmp.exists()
    assert valid_file.exists()


def test_atomic_save_cleans_up_temp_files_on_failure(tmp_path, monkeypatch):
    """Verify that if writing fails midway, temporary staging files are removed."""
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path))

    video_id = VALID_ID_14
    chunks = [{"text": "Test chunk", "start": 0}]
    index = faiss.IndexFlatIP(4)

    def broken_write_index(*args, **kwargs):
        raise IOError("Disk write simulation failure")

    monkeypatch.setattr(faiss, "write_index", broken_write_index)

    import pytest
    with pytest.raises(IOError, match="Disk write simulation failure"):
        cache.save_to_disk(video_id, chunks, index)

    tmp_files = list(tmp_path.glob("*.tmp*"))
    assert len(tmp_files) == 0


def test_atomic_save_torn_read_prevention_under_concurrency(tmp_path, monkeypatch):
    """Run concurrent readers and writers to verify that readers never encounter torn reads,
    partial JSON, mismatched vector/chunk counts, or None misses after initial cache save."""
    import threading

    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path))

    video_id = VALID_ID_15
    num_chunks = 20
    chunks = [{"text": f"Chunk payload line {i}", "start": i * 10} for i in range(num_chunks)]
    index = faiss.IndexFlatIP(4)
    vectors = np.ones((num_chunks, 4), dtype="float32")
    index.add(vectors)

    # Ensure cache is primed before reader threads start
    cache.save_to_disk(video_id, chunks, index)

    errors = []
    stop_event = threading.Event()

    def continuous_writer():
        while not stop_event.is_set():
            try:
                cache.save_to_disk(video_id, chunks, index)
            except Exception as e:
                errors.append(f"Writer error: {e}")
            time.sleep(0.005)

    def continuous_reader():
        for _ in range(50):
            try:
                result = cache.load_from_disk(video_id)
                assert result is not None, "load_from_disk returned None (torn read or missing cache)"
                loaded_chunks, loaded_index = result
                assert len(loaded_chunks) == loaded_index.ntotal, (
                    f"Count mismatch: chunks={len(loaded_chunks)}, index={loaded_index.ntotal}"
                )
                assert len(loaded_chunks) == num_chunks, (
                    f"Expected {num_chunks} chunks, got {len(loaded_chunks)}"
                )
                assert loaded_chunks[0]["text"] == chunks[0]["text"]
            except Exception as e:
                errors.append(f"Reader error: {e}")
            time.sleep(0.002)

    writer_thread = threading.Thread(target=continuous_writer)
    reader_threads = [threading.Thread(target=continuous_reader) for _ in range(3)]

    writer_thread.start()
    for t in reader_threads:
        t.start()

    for t in reader_threads:
        t.join()

    stop_event.set()
    writer_thread.join()

    assert len(errors) == 0, f"Concurrency errors detected: {errors}"
