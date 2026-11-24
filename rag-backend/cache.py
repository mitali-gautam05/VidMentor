import os
import json
import time
import uuid
import logging
import faiss
import re
from config import CACHE_DIR, EMBEDDING_MODEL, WINDOW_SEC, OVERLAP_SEC

logger = logging.getLogger(__name__)

# Maximum allowed modification time difference between .index and _chunks.json.
# Files written together have identical timestamps; drift indicates an interrupted write.
MAX_PAIR_DRIFT_SEC = 5.0

# YouTube ID pattern: 11 characters from A-Z, a-z, 0-9, _, -
VIDEO_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{11}")

# Retry parameters for Windows sharing violations during atomic replace
MAX_REPLACE_RETRIES = 15
INITIAL_REPLACE_DELAY_SEC = 0.01
MAX_REPLACE_DELAY_SEC = 0.05

# In-memory cache: cache_key -> (chunks, index)
video_cache = {}


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


def cleanup_stale_tmp_files(cache_dir: str | None = None, stale_age_sec: float = 300.0) -> int:
    """Remove orphaned temporary files left behind by crashed or killed processes.

    Files in cache_dir matching '*.tmp.*' or ending with '.tmp' older than
    stale_age_sec (default 300s) are deleted. If stale_age_sec is 0, all temporary
    files are removed immediately.
    """
    target_dir = cache_dir if cache_dir is not None else CACHE_DIR
    if not os.path.isdir(target_dir):
        return 0

    now = time.time()
    cleaned = 0
    try:
        for entry in os.scandir(target_dir):
            if entry.is_file() and (".tmp." in entry.name or entry.name.endswith(".tmp")):
                try:
                    mtime = entry.stat().st_mtime
                    if (now - mtime) >= stale_age_sec:
                        os.remove(entry.path)
                        cleaned += 1
                except OSError:
                    pass
    except OSError:
        pass
    return cleaned


def _atomic_replace(
    src: str,
    dst: str,
    retries: int = MAX_REPLACE_RETRIES,
    delay: float = INITIAL_REPLACE_DELAY_SEC,
) -> None:
    """Atomically replace dst with src, retrying briefly on Windows sharing violations.

    Bounded with a strictly clamped maximum retry count and capped delay to prevent hangs.
    """
    max_attempts = max(1, min(retries, 30))
    current_delay = delay
    for attempt in range(max_attempts):
        try:
            os.replace(src, dst)
            return
        except (PermissionError, OSError) as e:
            is_sharing_violation = (
                isinstance(e, PermissionError)
                or getattr(e, "winerror", None) in (5, 32)
            )
            if not is_sharing_violation or attempt == max_attempts - 1:
                raise
            time.sleep(current_delay)
            current_delay = min(current_delay * 1.5, MAX_REPLACE_DELAY_SEC)


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
        with open(tmp_idx, "a+b") as f:
            f.flush()
            os.fsync(f.fileno())

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


def load_from_disk(video_id, model_name=None, retries: int = 5, retry_delay: float = 0.005):
    key = get_cache_key(video_id, model_name)
    idx_path = f"{CACHE_DIR}/{key}.index"
    chunks_path = f"{CACHE_DIR}/{key}_chunks.json"

    max_attempts = max(1, min(retries, 10))
    for attempt in range(max_attempts):
        # Both files must exist
        if not (os.path.exists(idx_path) and os.path.exists(chunks_path)):
            if attempt < max_attempts - 1:
                time.sleep(retry_delay)
                continue
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
                if attempt < max_attempts - 1:
                    time.sleep(retry_delay)
                    continue
                return None

            with open(chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            if not isinstance(chunks, list):
                return None
            for chunk in chunks:
                if not isinstance(chunk, dict) or "text" not in chunk or "start" not in chunk:
                    return None

            index = faiss.read_index(idx_path)
            if index.ntotal != len(chunks):
                logger.warning(
                    "Cache files for %s have mismatched counts (index.ntotal=%d, len(chunks)=%d), ignoring as stale",
                    key,
                    index.ntotal,
                    len(chunks),
                )
                if attempt < max_attempts - 1:
                    time.sleep(retry_delay)
                    continue
                return None

            return chunks, index
        except (PermissionError, OSError, RuntimeError) as e:
            # On Windows, os.replace can briefly cause transient sharing violations for readers
            is_transient = (
                isinstance(e, PermissionError)
                or getattr(e, "winerror", None) in (5, 32)
                or "Permission denied" in str(e)
            )
            if is_transient and attempt < max_attempts - 1:
                time.sleep(retry_delay)
                continue
            logger.warning("Failed to load disk cache for %s, treating as miss: %s", key, e)
            return None
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Failed to load disk cache for %s, treating as miss: %s", key, e)
            return None

    return None


os.makedirs(CACHE_DIR, exist_ok=True)
cleanup_stale_tmp_files()
