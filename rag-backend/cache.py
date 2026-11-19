import os
import json
import time
import uuid
import logging
import faiss
import re
from config import CACHE_DIR, EMBEDDING_MODEL, WINDOW_SEC, OVERLAP_SEC

logger = logging.getLogger(__name__)

os.makedirs(CACHE_DIR, exist_ok=True)

# In-memory cache: cache_key -> (chunks, index)
video_cache = {}

# Maximum allowed modification time difference between .index and _chunks.json.
# Files written together have identical timestamps; drift indicates an interrupted write.
MAX_PAIR_DRIFT_SEC = 5.0

# YouTube ID pattern: 11 characters from A-Z, a-z, 0-9, _, -
VIDEO_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{11}")


def _validate_video_id(video_id: str) -> str:
    """Validate video_id format and reject path traversal attempts."""
    if not VIDEO_ID_PATTERN.fullmatch(video_id):
        raise ValueError("Invalid video_id format. Must be 11 characters from A-Z, a-z, 0-9, _, -")
    return video_id


def get_cache_key(video_id: str, model_name: str | None = None) -> str:
    _validate_video_id(video_id)
    if model_name is None:
        model_name = EMBEDDING_MODEL
    safe_model = str(model_name).replace("/", "_").replace("\\", "_")
    # Chunk settings are part of the key, so changing them rebuilds the index
    # instead of reusing chunks made with the old settings.
    return f"{video_id}_{safe_model}_w{WINDOW_SEC}_o{OVERLAP_SEC}"


def _cleanup_tmp_file(path: str) -> None:
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def _atomic_replace(src: str, dst: str, retries: int = 5, delay: float = 0.01) -> None:
    """Atomically replace dst with src, retrying briefly on Windows sharing violations."""
    for attempt in range(retries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(delay)


def save_to_disk(video_id, chunks, index, model_name=None):
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = get_cache_key(video_id, model_name)
    idx_path = f"{CACHE_DIR}/{key}.index"
    chunks_path = f"{CACHE_DIR}/{key}_chunks.json"

    # Stage to unique temporary files in the same directory (guarantees same filesystem for os.replace)
    tmp_chunks = f"{chunks_path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    tmp_idx = f"{idx_path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"

    try:
        with open(tmp_chunks, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

        faiss.write_index(index, tmp_idx)

        # Synchronize modification timestamps so load_from_disk can verify the pair belongs together
        write_time = time.time()
        os.utime(tmp_idx, (write_time, write_time))
        os.utime(tmp_chunks, (write_time, write_time))

        _atomic_replace(tmp_idx, idx_path)
        _atomic_replace(tmp_chunks, chunks_path)
    except Exception as e:
        _cleanup_tmp_file(tmp_chunks)
        _cleanup_tmp_file(tmp_idx)
        logger.error("Failed to save cache for %s: %s", video_id, e)
        raise


def load_from_disk(video_id, model_name=None):
    key = get_cache_key(video_id, model_name)
    idx_path = f"{CACHE_DIR}/{key}.index"
    chunks_path = f"{CACHE_DIR}/{key}_chunks.json"

    # Both files must exist
    if not (os.path.exists(idx_path) and os.path.exists(chunks_path)):
        return None

    try:
        # Detect mismatched or stale pairs: if an interrupted write replaced only one file,
        # modification timestamps will drift significantly.
        mtime_idx = os.path.getmtime(idx_path)
        mtime_chunks = os.path.getmtime(chunks_path)
        if abs(mtime_idx - mtime_chunks) > MAX_PAIR_DRIFT_SEC:
            logger.warning(
                "Cache files for %s have mismatched timestamps (drift=%.2fs), ignoring as stale",
                key,
                abs(mtime_idx - mtime_chunks),
            )
            return None

        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        if not isinstance(chunks, list):
            return None
        for chunk in chunks:
            if not isinstance(chunk, dict) or "text" not in chunk or "start" not in chunk:
                return None

        index = faiss.read_index(idx_path)
        return chunks, index
    except (json.JSONDecodeError, ValueError, OSError, RuntimeError) as e:
        logger.warning("Failed to load disk cache for %s, treating as miss: %s", key, e)
        return None
