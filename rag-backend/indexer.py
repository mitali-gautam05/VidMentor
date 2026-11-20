import faiss
from embeddings import embedder
from chunking import chunk_transcript
from transcript import get_transcript
from cache import video_cache, save_to_disk, load_from_disk, get_cache_key

def build_index_for_video(video_id, raw_transcript=None):
    if raw_transcript is not None:
        transcript = raw_transcript
        error = None
    else:
        transcript, error = get_transcript(video_id)
    if transcript is None:
        return None, error
    chunks = chunk_transcript(transcript)
    if not chunks:
        return None, "This video's transcript is empty."
    embeddings = embedder.encode([c["text"] for c in chunks], convert_to_numpy=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return (chunks, index), None


def get_or_build_index(video_id, raw_transcript=None):
    cache_key = get_cache_key(video_id)
    if cache_key in video_cache:
        return video_cache[cache_key], None
    if raw_transcript is None:
        disk_result = load_from_disk(video_id)
        if disk_result:
            video_cache[cache_key] = disk_result
            return disk_result, None
    result, error = build_index_for_video(video_id, raw_transcript=raw_transcript)
    if result is None:
        return None, error
    chunks, index = result
    if raw_transcript is None:
        save_to_disk(video_id, chunks, index)
    video_cache[cache_key] = (chunks, index)
    return (chunks, index), None
