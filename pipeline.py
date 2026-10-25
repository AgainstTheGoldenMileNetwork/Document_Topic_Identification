"""
Builds everything the API serves — embeddings, BERTopic topics, the
similarity graph, the 2-D map, the comparison between the two groupings and
an explanation of each case where they disagree —
in one place, so the API, evaluate.py and the tests share the same code path.

Embeddings and topic results are cached under settings.cache_dir, so only the
first start after the corpus or model changes is slow.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from config import Settings
from data_loader import ChunkIndex, Embedder, load_docs, load_or_embed
from evaluation import compare_partitions, explain_disagreements, passage_topic_mix
from labeling import generate_labels
from similarity import SimilarityGraph
from topic_modeling import project_2d, run_topic_modeling

logger = logging.getLogger(__name__)

ANALYSIS_VERSION = "v1"


@dataclass
class CorpusIndex:
    docs: List[Dict[str, Any]]
    embeddings: np.ndarray
    chunks: ChunkIndex
    topics: Dict[int, Dict[str, Any]]
    graph: SimilarityGraph
    coords: np.ndarray
    comparison: Dict[str, Any]
    disagreements: Dict[str, Any]
    outlier_fraction: float
    embedder: Embedder
    settings: Settings
    docs_by_id: Dict[str, Dict[str, Any]] = field(init=False)

    def __post_init__(self):
        self.docs_by_id = {str(d["id"]): d for d in self.docs}

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Semantic search over chunks: each doc scores by its best-matching
        passage, which is returned as the snippet.
        """
        q = self.embedder.embed_query(query)
        sims = self.chunks.embeddings @ q
        best: Dict[int, int] = {}
        for ci in np.argsort(-sims):
            di = int(self.chunks.doc_idx[ci])
            if di not in best:
                best[di] = int(ci)
                if len(best) >= top_k:
                    break
        results = []
        for di, ci in best.items():
            doc = self.docs[di]
            start, end = (int(x) for x in self.chunks.spans[ci])
            results.append({
                "id": str(doc["id"]),
                "title": doc.get("title"),
                "score": float(sims[ci]),
                "cluster_id": doc.get("cluster_id"),
                "snippet": " ".join(doc["text"][start:end].split())[:400],
            })
        return results


def _analysis_cache_key(embedding_key: str, settings: Settings) -> str:
    raw = f"{ANALYSIS_VERSION}|{embedding_key}|{settings.random_state}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def build_index(settings: Settings, embedder: Embedder | None = None) -> CorpusIndex:
    docs = load_docs(settings.docs_path)
    texts = [d["text"] for d in docs]
    embedder = embedder or Embedder(settings.embed_model, settings.embed_strategy, settings.chunk_overlap)

    logger.info("Embedding %d documents (%s, %s)...", len(docs), settings.embed_model, settings.embed_strategy)
    embeddings, chunks = load_or_embed(embedder, texts, settings.cache_dir)

    cache_path = settings.cache_dir / f"analysis-{_analysis_cache_key(embedder.cache_key(texts), settings)}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        topic_ids = cached["topic_ids"]
        topics = {int(k): v for k, v in cached["topics"].items()}
        outlier_fraction = cached["outlier_fraction"]
        coords = np.asarray(cached["coords"])
    else:
        logger.info("Running BERTopic...")
        topic_ids, topics, outlier_fraction = run_topic_modeling(docs, embeddings, settings.random_state)
        logger.info("Projecting to 2-D...")
        coords = project_2d(embeddings, settings.random_state)
        cache_path.write_text(json.dumps({
            "topic_ids": topic_ids,
            "topics": {str(k): v for k, v in topics.items()},
            "outlier_fraction": outlier_fraction,
            "coords": np.asarray(coords).tolist(),
        }))

    for doc, tid in zip(docs, topic_ids):
        doc["cluster_id"] = int(tid)

    docs_by_id = {str(d["id"]): d for d in docs}
    llm_labels = (
        generate_labels(topics, docs_by_id, settings.llm_model, settings.cache_dir) if settings.llm_labels else {}
    )
    for tid, info in topics.items():
        info["label"] = llm_labels.get(tid, info["summary"])
        info["label_source"] = "llm" if tid in llm_labels else "keywords"

    logger.info("Building similarity graph...")
    doc_ids = [str(d["id"]) for d in docs]
    graph = SimilarityGraph(
        doc_ids, embeddings, threshold=settings.sim_threshold, percentile=settings.sim_percentile,
        knn=settings.sim_knn, seed=settings.random_state,
    )
    community_ids = graph.community_labels()
    for doc, cid in zip(docs, community_ids):
        doc["community_id"] = cid
    comparison = compare_partitions(doc_ids, topic_ids, community_ids)
    disagreements = explain_disagreements(doc_ids, topic_ids, community_ids, embeddings)
    for doc, mix in zip(docs, passage_topic_mix(embeddings, topic_ids, chunks.embeddings, chunks.doc_idx)):
        doc["passage_topics"] = mix

    return CorpusIndex(
        docs=docs,
        embeddings=embeddings,
        chunks=chunks,
        topics=topics,
        graph=graph,
        coords=np.asarray(coords),
        comparison=comparison,
        disagreements=disagreements,
        outlier_fraction=outlier_fraction,
        embedder=embedder,
        settings=settings,
    )
