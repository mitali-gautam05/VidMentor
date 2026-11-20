from embeddings import embedder
from config import TOP_K, TIME_WINDOW_SEC, TIME_PENALTY_PER_SEC


# Semantic search, re-ranked with a soft penalty for chunks far from the
# viewer's current playback time (no penalty inside TIME_WINDOW_SEC).
def retrieve(query, chunks, index, current_time=None, top_k=TOP_K, time_window=TIME_WINDOW_SEC):
    if not chunks:
        return []
    ntotal = getattr(index, "ntotal", len(chunks))
    if ntotal == 0:
        return []
    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    k = min(len(chunks), ntotal)
    scores, idxs = index.search(q_emb, k)
    results = []
    for score, i in zip(scores[0], idxs[0]):
        if i < 0 or i >= len(chunks):
            continue
        c = chunks[i]
        penalty = 0
        if current_time is not None:
            dist = abs(c["start"] - current_time)
            penalty = max(0, dist - time_window) * TIME_PENALTY_PER_SEC
        results.append((score - penalty, c))
    results.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in results[:top_k]]