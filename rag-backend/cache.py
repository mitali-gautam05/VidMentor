import os
import json
import logging
import faiss
import numpy as np
from config import (
    CACHE_DIR, CACHE_BACKEND, REDIS_URL, CACHE_TTL_SEC,
    EMBEDDING_MODEL, WINDOW_SEC, OVERLAP_SEC,
)

logger = logging.getLogger(__name__)

# Prefix for every Redis key, so VidMentor never touches other keys on a shared Redis.
REDIS_KEY_PREFIX = "vidmentor:cache:v1:"

if CACHE_BACKEND == "disk":
    os.makedirs(CACHE_DIR, exist_ok=True)

# In-memory cache: cache_key -> (chunks, index)
video_cache = {}

def get_cache_key(video_id: str, model_name: str | None = None) -> str:
    if model_name is None:
        model_name = EMBEDDING_MODEL
    safe_model = str(model_name).replace("/", "_").replace("\\", "_")
    # Chunk settings are part of the key, so changing them rebuilds the index
    # instead of reusing chunks made with the old settings.
    return f"{video_id}_{safe_model}_w{WINDOW_SEC}_o{OVERLAP_SEC}"


def _valid_chunks(chunks):
    if not isinstance(chunks, list):
        return False
    return all(isinstance(c, dict) and "text" in c and "start" in c for c in chunks)


class DiskStore:
    """Cache on local disk (or a mounted persistent volume). Reads cache.CACHE_DIR on every call."""

    def save(self, key, chunks, index):
        os.makedirs(CACHE_DIR, exist_ok=True)
        faiss.write_index(index, f"{CACHE_DIR}/{key}.index")
        with open(f"{CACHE_DIR}/{key}_chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False)

    def load(self, key):
        idx_path = f"{CACHE_DIR}/{key}.index"
        chunks_path = f"{CACHE_DIR}/{key}_chunks.json"
        if os.path.exists(idx_path) and os.path.exists(chunks_path):
            try:
                with open(chunks_path, "r", encoding="utf-8") as f:
                    chunks = json.load(f)
                if not _valid_chunks(chunks):
                    return None
                return chunks, faiss.read_index(idx_path)
            except (json.JSONDecodeError, ValueError, OSError):
                return None
        return None


class RedisStore:
    """Cache in Redis. Any Redis problem is logged and treated as a miss; nothing is raised."""

    def __init__(self, client=None, url=None, ttl=None):
        self._client = client
        self._url = url if url is not None else REDIS_URL
        self.ttl = CACHE_TTL_SEC if ttl is None else ttl

    def _get_client(self):
        if self._client is None:
            try:
                import redis  # imported only when the Redis backend is used
                self._client = redis.Redis.from_url(
                    self._url, socket_connect_timeout=2, socket_timeout=2
                )
            except Exception as e:
                logger.warning("Redis cache unavailable, treating as miss: %s", e)
                return None
        return self._client

    @staticmethod
    def _keys(key):
        return f"{REDIS_KEY_PREFIX}{key}:index", f"{REDIS_KEY_PREFIX}{key}:chunks"

    def save(self, key, chunks, index):
        try:
            client = self._get_client()
            if client is None:
                return
            index_key, chunks_key = self._keys(key)
            index_bytes = faiss.serialize_index(index).tobytes()
            chunks_bytes = json.dumps(chunks, ensure_ascii=False).encode("utf-8")
            ex = self.ttl if self.ttl and self.ttl > 0 else None
            # Chunks last: a reader that sees chunks also finds the index.
            client.set(index_key, index_bytes, ex=ex)
            client.set(chunks_key, chunks_bytes, ex=ex)
        except Exception as e:
            logger.warning("Redis cache save failed for %s: %s", key, e)

    def load(self, key):
        try:
            client = self._get_client()
            if client is None:
                return None
            index_key, chunks_key = self._keys(key)
            chunks_raw = client.get(chunks_key)
            index_raw = client.get(index_key)
            if chunks_raw is None or index_raw is None:
                return None
            chunks = json.loads(chunks_raw)
            if not _valid_chunks(chunks):
                logger.warning("Redis cache entry for %s has invalid chunks, ignoring", key)
                return None
            index = faiss.deserialize_index(np.frombuffer(index_raw, dtype="uint8"))
            return chunks, index
        except Exception as e:
            logger.warning("Redis cache load failed for %s: %s", key, e)
            return None


_redis_store = None

def get_store():
    """Return the store chosen by CACHE_BACKEND (disk unless set to redis)."""
    global _redis_store
    if CACHE_BACKEND == "redis":
        if _redis_store is None:
            _redis_store = RedisStore()
        return _redis_store
    return DiskStore()


# Kept under the old names so indexer.py and existing callers keep working.
def save_to_disk(video_id, chunks, index, model_name=None):
    get_store().save(get_cache_key(video_id, model_name), chunks, index)

def load_from_disk(video_id, model_name=None):
    return get_store().load(get_cache_key(video_id, model_name))
