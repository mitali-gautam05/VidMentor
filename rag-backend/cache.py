import os
import pickle
import re
import faiss
from config import CACHE_DIR, EMBEDDING_MODEL, WINDOW_SEC, OVERLAP_SEC

os.makedirs(CACHE_DIR, exist_ok=True)

# In-memory cache: cache_key -> (chunks, index)
video_cache = {}

VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")


def _validate_video_id(video_id: str) -> None:
    if not VIDEO_ID_PATTERN.match(video_id):
        raise ValueError(f"Invalid video_id: {video_id!r}. Must be 11 chars A-Z a-z 0-9 _ -")


def get_cache_key(video_id: str, model_name: str | None = None) -> str:
    _validate_video_id(video_id)
    if model_name is None:
        model_name = EMBEDDING_MODEL
    safe_model = str(model_name).replace("/", "_").replace("\\", "_")
    # Chunk settings are part of the key, so changing them rebuilds the index
    # instead of reusing chunks made with the old settings.
    return f"{video_id}_{safe_model}_w{WINDOW_SEC}_o{OVERLAP_SEC}"


def save_to_disk(video_id, chunks, index, model_name=None):
    key = get_cache_key(video_id, model_name)
    faiss.write_index(index, f"{CACHE_DIR}/{key}.index")
    with open(f"{CACHE_DIR}/{key}_chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)


def load_from_disk(video_id, model_name=None):
    key = get_cache_key(video_id, model_name)
    idx_path, chunks_path = f"{CACHE_DIR}/{key}.index", f"{CACHE_DIR}/{key}_chunks.pkl"
    if os.path.exists(idx_path) and os.path.exists(chunks_path):
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)
        return chunks, faiss.read_index(idx_path)
    return None