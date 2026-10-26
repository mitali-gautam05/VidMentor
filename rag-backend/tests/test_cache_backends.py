import sys

import faiss
import numpy as np
import pytest
from fastapi.testclient import TestClient

import cache
import indexer
import main


class FakeRedis:
    """Tiny in-memory stand-in for redis.Redis (get/set with ex)."""

    def __init__(self):
        self.data = {}
        self.ttls = {}
        self.fail_get = False
        self.fail_set = False

    def get(self, key):
        if self.fail_get:
            raise ConnectionError("redis is down")
        return self.data.get(key)

    def set(self, key, value, ex=None):
        if self.fail_set:
            raise ConnectionError("redis is down")
        self.data[key] = value
        self.ttls[key] = ex


def make_index():
    index = faiss.IndexFlatIP(4)
    index.add(np.array([[1.0, 0.0, 0.0, 0.0]], dtype="float32"))
    return index


CHUNKS = [{"text": "Hello world", "start": 0}, {"text": "Second chunk", "start": 10}]


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    cache.video_cache.clear()
    monkeypatch.setattr(cache, "_redis_store", None)
    yield
    cache.video_cache.clear()


def use_redis(monkeypatch, client, ttl=0):
    store = cache.RedisStore(client=client, ttl=ttl)
    monkeypatch.setattr(cache, "CACHE_BACKEND", "redis")
    monkeypatch.setattr(cache, "_redis_store", store)
    return store


def test_default_backend_is_disk(tmp_path, monkeypatch):
    assert cache.CACHE_BACKEND == "disk"
    assert isinstance(cache.get_store(), cache.DiskStore)
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path))
    cache.save_to_disk("vid", CHUNKS, make_index())
    assert (tmp_path / f"{cache.get_cache_key('vid')}.index").exists()
    assert (tmp_path / f"{cache.get_cache_key('vid')}_chunks.json").exists()


def test_redis_round_trip_through_wrappers(monkeypatch):
    use_redis(monkeypatch, FakeRedis())
    cache.save_to_disk("vid", CHUNKS, make_index())
    loaded_chunks, loaded_index = cache.load_from_disk("vid")
    assert loaded_chunks == CHUNKS
    assert loaded_index.d == 4
    assert loaded_index.ntotal == 1


def test_redis_miss_for_unknown_video(monkeypatch):
    use_redis(monkeypatch, FakeRedis())
    assert cache.load_from_disk("never_saved") is None


def test_redis_keys_are_namespaced_and_per_video(monkeypatch):
    client = FakeRedis()
    use_redis(monkeypatch, client)
    cache.save_to_disk("videoA", CHUNKS, make_index())
    cache.save_to_disk("videoB", [{"text": "other", "start": 1}], make_index())
    assert all(k.startswith(cache.REDIS_KEY_PREFIX) for k in client.data)
    assert len(client.data) == 4
    assert cache.load_from_disk("videoA")[0] == CHUNKS
    assert cache.load_from_disk("videoB")[0] == [{"text": "other", "start": 1}]
    assert cache.load_from_disk("video") is None


def test_redis_model_change_gives_different_key(monkeypatch):
    use_redis(monkeypatch, FakeRedis())
    cache.save_to_disk("vid", CHUNKS, make_index(), model_name="model-a")
    assert cache.load_from_disk("vid", model_name="model-a") is not None
    assert cache.load_from_disk("vid", model_name="model-b") is None


def test_redis_chunk_settings_change_gives_different_key(monkeypatch):
    use_redis(monkeypatch, FakeRedis())
    cache.save_to_disk("vid", CHUNKS, make_index())
    monkeypatch.setattr(cache, "WINDOW_SEC", cache.WINDOW_SEC + 5)
    assert cache.load_from_disk("vid") is None


def test_redis_ttl_is_passed_on_set(monkeypatch):
    client = FakeRedis()
    use_redis(monkeypatch, client, ttl=3600)
    cache.save_to_disk("vid", CHUNKS, make_index())
    assert set(client.ttls.values()) == {3600}


def test_redis_no_ttl_means_no_expiry(monkeypatch):
    client = FakeRedis()
    use_redis(monkeypatch, client, ttl=0)
    cache.save_to_disk("vid", CHUNKS, make_index())
    assert set(client.ttls.values()) == {None}


def test_redis_expired_entry_is_a_miss(monkeypatch):
    client = FakeRedis()
    use_redis(monkeypatch, client, ttl=60)
    cache.save_to_disk("vid", CHUNKS, make_index())
    client.data.clear()  # what Redis does once the TTL passes
    assert cache.load_from_disk("vid") is None


@pytest.mark.parametrize("bad_chunks", [b"not json{", b'{"a": 1}', b'[{"text": "no start"}]'])
def test_redis_corrupt_chunks_is_a_miss(monkeypatch, bad_chunks):
    client = FakeRedis()
    use_redis(monkeypatch, client)
    cache.save_to_disk("vid", CHUNKS, make_index())
    chunks_key = [k for k in client.data if k.endswith(":chunks")][0]
    client.data[chunks_key] = bad_chunks
    assert cache.load_from_disk("vid") is None


def test_redis_corrupt_index_is_a_miss(monkeypatch):
    client = FakeRedis()
    use_redis(monkeypatch, client)
    cache.save_to_disk("vid", CHUNKS, make_index())
    index_key = [k for k in client.data if k.endswith(":index")][0]
    client.data[index_key] = b"garbage bytes"
    assert cache.load_from_disk("vid") is None


@pytest.mark.parametrize("missing", [":index", ":chunks"])
def test_redis_missing_half_is_a_miss(monkeypatch, missing):
    client = FakeRedis()
    use_redis(monkeypatch, client)
    cache.save_to_disk("vid", CHUNKS, make_index())
    for k in [k for k in client.data if k.endswith(missing)]:
        del client.data[k]
    assert cache.load_from_disk("vid") is None


def test_redis_get_error_is_a_miss(monkeypatch):
    client = FakeRedis()
    use_redis(monkeypatch, client)
    cache.save_to_disk("vid", CHUNKS, make_index())
    client.fail_get = True
    assert cache.load_from_disk("vid") is None


def test_redis_set_error_does_not_raise(monkeypatch):
    client = FakeRedis()
    client.fail_set = True
    use_redis(monkeypatch, client)
    cache.save_to_disk("vid", CHUNKS, make_index())  # must not raise
    assert client.data == {}


def test_bad_redis_url_is_a_miss_and_does_not_raise(monkeypatch):
    store = cache.RedisStore(url="not-a-valid-url")
    monkeypatch.setattr(cache, "CACHE_BACKEND", "redis")
    monkeypatch.setattr(cache, "_redis_store", store)
    cache.save_to_disk("vid", CHUNKS, make_index())
    assert cache.load_from_disk("vid") is None


def test_missing_redis_package_is_a_miss(monkeypatch):
    monkeypatch.setitem(sys.modules, "redis", None)  # makes "import redis" fail
    store = cache.RedisStore(url="redis://localhost:6379/0")
    assert store.load("k") is None
    store.save("k", CHUNKS, make_index())


def test_redis_not_imported_with_default_backend(tmp_path, monkeypatch):
    monkeypatch.delitem(sys.modules, "redis", raising=False)
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path))
    cache.save_to_disk("vid", CHUNKS, make_index())
    cache.load_from_disk("vid")
    assert "redis" not in sys.modules


def test_get_or_build_index_survives_failing_redis(monkeypatch):
    client = FakeRedis()
    client.fail_get = True
    client.fail_set = True
    use_redis(monkeypatch, client)
    built = []

    def fake_build(video_id):
        built.append(video_id)
        return (CHUNKS, make_index()), None

    monkeypatch.setattr(indexer, "build_index_for_video", fake_build)
    result, error = indexer.get_or_build_index("vid")
    assert error is None
    assert result[0] == CHUNKS
    assert built == ["vid"]


def test_get_or_build_index_loads_from_redis_without_rebuilding(monkeypatch):
    use_redis(monkeypatch, FakeRedis())
    cache.save_to_disk("vid", CHUNKS, make_index())

    def fail_build(video_id):
        raise AssertionError("should not rebuild")

    monkeypatch.setattr(indexer, "build_index_for_video", fail_build)
    result, error = indexer.get_or_build_index("vid")
    assert error is None
    assert result[0] == CHUNKS


def test_chat_still_works_when_redis_is_down(monkeypatch):
    client = FakeRedis()
    client.fail_get = True
    client.fail_set = True
    use_redis(monkeypatch, client)
    monkeypatch.setattr(indexer, "build_index_for_video", lambda vid: ((CHUNKS, make_index()), None))
    monkeypatch.setattr(main, "retrieve", lambda query, chunks, idx, ct: chunks)
    monkeypatch.setattr(main, "call_groq_with_retry", lambda prompt: "An answer")
    main.chat_sessions.clear()

    response = TestClient(main.app).post(
        "/chat", json={"video_id": "vid_redis_down", "query": "What?", "current_time": 5.0}
    )

    assert response.status_code == 200
    assert response.json()["success"] is True
    assert response.json()["answer"] == "An answer"
    main.chat_sessions.clear()
