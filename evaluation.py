"""
Label-free metrics for comparing embedding strategies and the two grouping
methods. The corpus has no ground-truth topic labels, so each metric measures
something checkable on its own:

- self-retrieval: each case's syllabus (an independent summary written by the
  Reporter of Decisions) should retrieve that case's opinion. This directly
  tests whether the opinion embedding captures what the case is about.
- NPMI coherence / diversity: do a topic's keywords co-occur in real
  documents, and do topics use distinct vocabularies?
- ARI / NMI: how closely BERTopic topics and Louvain communities agree.
- stability: do topics survive a change of UMAP random seed?
"""

from collections import Counter
from itertools import combinations, product
from typing import Any, Dict, List, Sequence

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from topic_modeling import LEGAL_STOP_WORDS


def self_retrieval(doc_embeddings: np.ndarray, query_embeddings: np.ndarray, valid: Sequence[bool]) -> Dict[str, float]:
    """Query i should rank doc i first. Rows where valid[i] is False are skipped."""
    valid_idx = np.flatnonzero(np.asarray(valid))
    sims = query_embeddings[valid_idx] @ doc_embeddings.T
    target = sims[np.arange(len(valid_idx)), valid_idx]
    ranks = 1 + (sims > target[:, None]).sum(axis=1)
    return {
        "queries": int(len(valid_idx)),
        "mrr": float(np.mean(1.0 / ranks)),
        "recall_at_1": float(np.mean(ranks <= 1)),
        "recall_at_5": float(np.mean(ranks <= 5)),
        "median_rank": float(np.median(ranks)),
    }


def npmi_coherence(topic_terms: List[List[str]], texts: List[str], top_n: int = 10) -> float:
    """
    Mean NPMI over keyword pairs, using whole-document co-occurrence.
    Ranges from -1 (never together) to 1 (always together).
    """
    vocab = sorted({t for terms in topic_terms for t in terms[:top_n]})
    if not vocab:
        return 0.0
    # Same tokenisation as the topic model, so bigram keywords line up.
    vectorizer = CountVectorizer(
        vocabulary=vocab,
        binary=True,
        ngram_range=(1, 2),
        stop_words=list(ENGLISH_STOP_WORDS.union(LEGAL_STOP_WORDS)),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )
    X = vectorizer.fit_transform(texts).tocsc()
    n_docs = X.shape[0]
    col = {w: i for i, w in enumerate(vocab)}
    doc_sets = {w: set(X[:, col[w]].nonzero()[0]) for w in vocab}

    scores = []
    for terms in topic_terms:
        pair_scores = []
        for a, b in combinations(terms[:top_n], 2):
            pa, pb = len(doc_sets[a]) / n_docs, len(doc_sets[b]) / n_docs
            pab = len(doc_sets[a] & doc_sets[b]) / n_docs
            if pab == 0 or pa == 0 or pb == 0:
                pair_scores.append(-1.0)
            elif pab == 1.0:
                pair_scores.append(1.0)
            else:
                pair_scores.append(np.log(pab / (pa * pb)) / -np.log(pab))
        if pair_scores:
            scores.append(float(np.mean(pair_scores)))
    return float(np.mean(scores)) if scores else 0.0


def topic_diversity(topic_terms: List[List[str]], top_n: int = 10) -> float:
    """Fraction of unique words across all topics' top-n keywords (1.0 = no overlap)."""
    words = [w for terms in topic_terms for w in terms[:top_n]]
    return len(set(words)) / len(words) if words else 0.0


def compare_partitions(doc_ids: List[str], topics: List[int], communities: List[int]) -> Dict[str, Any]:
    """
    Agreement between BERTopic topics and Louvain communities, plus the
    documents where they disagree: a doc disagrees when its topic is not the
    most common topic in its community.
    """
    contingency: Dict[int, Counter] = {}
    for t, c in zip(topics, communities):
        contingency.setdefault(c, Counter())[t] += 1
    majority = {c: counts.most_common(1)[0][0] for c, counts in contingency.items()}
    disagreements = [did for did, t, c in zip(doc_ids, topics, communities) if majority[c] != t]
    return {
        "ari": float(adjusted_rand_score(topics, communities)),
        "nmi": float(normalized_mutual_info_score(topics, communities)),
        "num_topics": len(set(topics)),
        "num_communities": len(set(communities)),
        "community_majority_topic": {str(c): int(t) for c, t in sorted(majority.items())},
        "contingency": {
            str(c): {str(t): n for t, n in sorted(counts.items())} for c, counts in sorted(contingency.items())
        },
        "disagreement_rate": len(disagreements) / len(doc_ids) if doc_ids else 0.0,
        "disagreements": disagreements,
    }


def threshold_sweep(
    doc_ids: List[str],
    embeddings: np.ndarray,
    topics: List[int],
    percentiles: Sequence[float] = (85, 90, 93, 95, 97, 99),
    knn_values: Sequence[int] = (0, 3, 5),
) -> List[Dict[str, Any]]:
    """Rebuild the similarity graph at several edge cut-offs and neighbour counts and record what changes."""
    from similarity import SimilarityGraph

    rows = []
    for k, p in product(knn_values, percentiles):
        g = SimilarityGraph(doc_ids, embeddings, percentile=p, knn=k)
        info = g.get_info()
        labels = g.community_labels()
        sizes = sorted((len(c) for c in g.communities), reverse=True)
        rows.append({
            "knn": k,
            "percentile": p,
            "threshold": round(g.threshold, 4),
            "edges": info["num_edges"],
            "isolated_docs": sum(1 for _, d in g.graph.degree() if d == 0),
            "communities": info["communities"],
            "largest_community_share": sizes[0] / len(doc_ids),
            "modularity": round(info["modularity"], 4),
            "ari_vs_topics": round(float(adjusted_rand_score(topics, labels)), 4),
            "nmi_vs_topics": round(float(normalized_mutual_info_score(topics, labels)), 4),
        })
    return rows


def topic_stability(docs: List[Dict[str, Any]], embeddings: np.ndarray, seeds: Sequence[int] = (1, 2, 3)) -> float:
    """Mean pairwise ARI between BERTopic runs that differ only in UMAP seed."""
    from topic_modeling import run_topic_modeling

    runs = [run_topic_modeling(docs, embeddings, random_state=s)[0] for s in seeds]
    return float(np.mean([adjusted_rand_score(a, b) for a, b in combinations(runs, 2)]))
