import os
import json
import faiss
import logging
from config import CACHE_DIR, EMBEDDING_MODEL, WINDOW_SEC, OVERLAP_SEC

logger = logging.getLogger(__name__)

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

def save_to_disk(video_id, chunks, index, model_name=None):
    key = get_cache_key(video_id, model_name)
    faiss.write_index(index, f"{CACHE_DIR}/{key}.index")
    with open(f"{CACHE_DIR}/{key}_chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

def load_from_disk(video_id, model_name=None):
    key = get_cache_key(video_id, model_name)
    idx_path = f"{CACHE_DIR}/{key}.index"
    chunks_path = f"{CACHE_DIR}/{key}_chunks.json"
    if os.path.exists(idx_path) and os.path.exists(chunks_path):
        try:
            with open(chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            if not isinstance(chunks, list):
                raise ValueError("Chunks must be a list")
            for chunk in chunks:
                if not isinstance(chunk, dict) or "text" not in chunk or "start" not in chunk:
                    raise ValueError("Malformed chunk structure")
            return chunks, faiss.read_index(idx_path)
        except Exception as e:
            logger.warning("Corrupt disk cache for %s (%s); purging", video_id, e)
            for path in (idx_path, chunks_path):
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except OSError:
                    pass
            return None
    return None
