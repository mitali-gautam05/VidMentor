import numpy as np
import retrieval


class FakeEmbedder:
    def encode(self, texts, **kwargs):
        return np.zeros((len(texts), 2), dtype="float32")


class FakeIndex:
    """Returns chunk positions ranked by the given similarity scores."""
    def __init__(self, scores):
        self.scores = np.array(scores, dtype="float32")

    def search(self, q_emb, k):
        order = np.argsort(-self.scores)
        return np.array([self.scores[order]]), np.array([order])


def make_chunks(*starts):
    return [{"text": f"chunk at {s}", "start": s, "end": s + 40} for s in starts]


def test_nearby_chunk_wins_when_similarity_ties(monkeypatch):
    monkeypatch.setattr(retrieval, "embedder", FakeEmbedder())
    chunks = make_chunks(0, 1000)
    index = FakeIndex([0.9, 0.9])
    result = retrieval.retrieve("q", chunks, index, current_time=1000, top_k=1)
    assert result[0]["start"] == 1000


def test_without_current_time_best_similarity_wins(monkeypatch):
    monkeypatch.setattr(retrieval, "embedder", FakeEmbedder())
    chunks = make_chunks(0, 1000)
    index = FakeIndex([0.95, 0.5])
    result = retrieval.retrieve("q", chunks, index, current_time=None, top_k=1)
    assert result[0]["start"] == 0


def test_no_penalty_inside_the_time_window(monkeypatch):
    monkeypatch.setattr(retrieval, "embedder", FakeEmbedder())
    chunks = make_chunks(0, 100)
    index = FakeIndex([0.9, 0.8])
    result = retrieval.retrieve("q", chunks, index, current_time=100, top_k=1)
    assert result[0]["start"] == 0  # 100s away is inside the 180s window


def test_returns_at_most_top_k(monkeypatch):
    monkeypatch.setattr(retrieval, "embedder", FakeEmbedder())
    chunks = make_chunks(0, 50, 100, 150, 200)
    index = FakeIndex([0.9, 0.8, 0.7, 0.6, 0.5])
    assert len(retrieval.retrieve("q", chunks, index, top_k=3)) == 3

def test_empty_chunks_returns_empty_list(monkeypatch):
    monkeypatch.setattr(retrieval, "embedder", FakeEmbedder())
    index = FakeIndex([])
    assert retrieval.retrieve("q", [], index) == []


def test_negative_or_out_of_bounds_faiss_indices_are_ignored(monkeypatch):
    monkeypatch.setattr(retrieval, "embedder", FakeEmbedder())
    chunks = make_chunks(0, 100)

    class IndexWithSentinels:
        ntotal = 2
        def search(self, q_emb, k):
            return np.array([[0.9, 0.1]]), np.array([[0, -1]])

    result = retrieval.retrieve("q", chunks, IndexWithSentinels(), top_k=5)
    assert len(result) == 1
    assert result[0]["start"] == 0
