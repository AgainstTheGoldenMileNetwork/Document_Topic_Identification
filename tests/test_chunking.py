import numpy as np
import pytest

from data_loader import ChunkIndex, chunk_spans, load_or_embed, pool_chunks
from tests.conftest import FakeEmbedder


def offsets(n, width=3):
    # token i covers characters [4i, 4i+3)
    return [(4 * i, 4 * i + width) for i in range(n)]


def test_chunk_spans_cover_text_with_overlap():
    spans = chunk_spans(offsets(10), window=4, overlap=1)
    assert spans == [(0, 15), (12, 27), (24, 39)]


def test_short_text_is_one_chunk():
    assert chunk_spans(offsets(3), window=4, overlap=1) == [(0, 11)]


def test_empty_and_invalid():
    assert chunk_spans([], window=4, overlap=1) == []
    with pytest.raises(ValueError):
        chunk_spans(offsets(5), window=4, overlap=4)


def test_pool_chunks_weights_by_length_and_normalises():
    chunks = ChunkIndex(
        embeddings=np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float32),
        doc_idx=np.array([0, 0, 1]),
        spans=np.array([[0, 30], [30, 40], [0, 5]]),
    )
    pooled = pool_chunks(chunks, 2)
    assert np.allclose(np.linalg.norm(pooled, axis=1), 1.0)
    assert pooled[0, 0] > pooled[0, 1]  # the longer chunk dominates
    assert np.allclose(pooled[1], [0.0, 1.0])


def test_chunked_embedding_reads_the_whole_document(tmp_path):
    # The distinguishing word only appears after the first window.
    head = "filler " * 60
    texts = [head + "patent patent patent", head + "vessel vessel vessel"]
    chunked, chunks = FakeEmbedder("chunked").embed_corpus(texts)
    assert len(chunks.doc_idx) > 2
    assert chunked[0] @ chunked[1] < 0.999
    # Chunk spans slice the original text exactly.
    start, end = chunks.spans[-1]
    assert texts[1][start:end].endswith("vessel vessel vessel")


def test_embedding_cache_round_trip(tmp_path):
    embedder = FakeEmbedder()
    texts = ["one two three " * 20, "four five six " * 20]
    first, chunks = load_or_embed(embedder, texts, tmp_path)
    assert len(list(tmp_path.glob("emb-*.npz"))) == 1
    embedder._model = None  # a cache hit must not need the model
    second, chunks2 = load_or_embed(embedder, texts, tmp_path)
    assert np.allclose(first, second)
    assert np.array_equal(chunks.spans, chunks2.spans)
