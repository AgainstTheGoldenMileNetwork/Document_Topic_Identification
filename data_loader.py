import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

DATA_PATH = Path("data/documents.jsonl")

Span = Tuple[int, int]


def load_docs(path: Path = DATA_PATH) -> List[Dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(f"{path} does not exist. Run download_docs.py first.")
    docs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    return docs


def chunk_spans(offsets: Sequence[Span], window: int, overlap: int) -> List[Span]:
    """
    Group token character offsets into overlapping windows of `window` tokens
    and return the character span each window covers in the original text.
    Slicing the original text (rather than decoding tokens) keeps casing and
    punctuation intact for search snippets.
    """
    if not offsets:
        return []
    if overlap >= window:
        raise ValueError("overlap must be smaller than window")
    step = window - overlap
    spans = []
    for start in range(0, len(offsets), step):
        group = offsets[start:start + window]
        spans.append((group[0][0], group[-1][1]))
        if start + window >= len(offsets):
            break
    return spans


@dataclass
class ChunkIndex:
    """Chunk-level embeddings, used for passage search and doc pooling."""

    embeddings: np.ndarray  # (num_chunks, dim), L2-normalised
    doc_idx: np.ndarray     # (num_chunks,) index into the docs list
    spans: np.ndarray       # (num_chunks, 2) character span in the doc text


class Embedder:
    """
    Wraps a SentenceTransformer. The model only reads `max_seq_length` tokens
    (256 for all-MiniLM-L6-v2) and silently drops the rest, so long opinions
    are split into overlapping windows and the window vectors are averaged,
    weighted by length.
    """

    def __init__(self, model_name: str, strategy: str = "chunked", overlap: int = 32):
        if strategy not in {"chunked", "truncated"}:
            raise ValueError(f"Unknown embed strategy: {strategy}")
        self.model_name = model_name
        self.strategy = strategy
        self.overlap = overlap
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def cache_key(self, texts: List[str]) -> str:
        h = hashlib.sha1()
        h.update(f"{self.model_name}|{self.strategy}|{self.overlap}|v1".encode())
        for t in texts:
            h.update(hashlib.sha1(t.encode("utf-8")).digest())
        return h.hexdigest()[:16]

    def _spans_for(self, text: str) -> List[Span]:
        if self.strategy == "truncated":
            return [(0, len(text))]
        # Leave headroom for special tokens and re-tokenisation drift.
        window = self.model.max_seq_length - 16
        enc = self.model.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)
        spans = chunk_spans(enc["offset_mapping"], window, self.overlap)
        return spans or [(0, len(text))]

    def embed_corpus(self, texts: List[str]) -> Tuple[np.ndarray, ChunkIndex]:
        chunk_texts: List[str] = []
        doc_idx: List[int] = []
        spans: List[Span] = []
        for i, text in enumerate(texts):
            for start, end in self._spans_for(text):
                chunk_texts.append(text[start:end])
                doc_idx.append(i)
                spans.append((start, end))

        chunk_emb = self.model.encode(
            chunk_texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True
        )
        chunks = ChunkIndex(
            embeddings=np.asarray(chunk_emb, dtype=np.float32),
            doc_idx=np.asarray(doc_idx),
            spans=np.asarray(spans),
        )
        return pool_chunks(chunks, len(texts)), chunks

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(self.model.encode([text], normalize_embeddings=True)[0], dtype=np.float32)


def pool_chunks(chunks: ChunkIndex, num_docs: int) -> np.ndarray:
    """Length-weighted mean of each doc's chunk vectors, re-normalised."""
    weights = (chunks.spans[:, 1] - chunks.spans[:, 0]).astype(np.float32)
    dim = chunks.embeddings.shape[1]
    pooled = np.zeros((num_docs, dim), dtype=np.float32)
    np.add.at(pooled, chunks.doc_idx, chunks.embeddings * weights[:, None])
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    return pooled / np.clip(norms, 1e-12, None)


def load_or_embed(embedder: Embedder, texts: List[str], cache_dir: Path) -> Tuple[np.ndarray, ChunkIndex]:
    """Embed the corpus, reusing a cached result when texts and model match."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"emb-{embedder.cache_key(texts)}.npz"
    if path.exists():
        data = np.load(path)
        chunks = ChunkIndex(data["chunk_embeddings"], data["chunk_doc_idx"], data["chunk_spans"])
        return data["doc_embeddings"], chunks

    doc_emb, chunks = embedder.embed_corpus(texts)
    np.savez(
        path,
        doc_embeddings=doc_emb,
        chunk_embeddings=chunks.embeddings,
        chunk_doc_idx=chunks.doc_idx,
        chunk_spans=chunks.spans,
    )
    return doc_emb, chunks
