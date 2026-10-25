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


def _unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-12)


def explain_disagreements(
    doc_ids: List[str], topics: List[int], communities: List[int], embeddings: np.ndarray, k: int = 10
) -> Dict[str, Any]:
    """
    Sort the documents compare_partitions flags as disagreeing into three kinds,
    because "topic differs from its community's majority" lumps together very
    different situations:

    - lone: the only document of its topic in its community. These are the
      individual cases the two methods really place differently.
    - merged: most of the document's topic sits in this same community, which
      it shares with a larger topic. Louvain drew one group where BERTopic drew
      two; nothing about the document itself is unusual.
    - split: a sub-group of the topic (two or more documents) sits in a
      community dominated by another topic, apart from the rest of its topic.

    Each document also gets evidence for which method's placement it fits: how
    its k nearest neighbours divide between its topic and its community, and its
    similarity to the rest of its own topic versus the community's majority topic.
    """
    topics, communities = [int(t) for t in topics], [int(c) for c in communities]
    base = compare_partitions(doc_ids, topics, communities)
    majority = {int(c): t for c, t in base["community_majority_topic"].items()}
    topic_arr, comm_arr = np.asarray(topics), np.asarray(communities)
    pair_counts = Counter(zip(topics, communities))
    topic_sizes = Counter(topics)
    home = {t: Counter(comm_arr[topic_arr == t].tolist()).most_common(1)[0][0] for t in topic_sizes}
    sums = {t: embeddings[topic_arr == t].sum(axis=0) for t in topic_sizes}
    pos = {did: i for i, did in enumerate(doc_ids)}

    details: Dict[str, Dict[str, Any]] = {}
    groups: Dict[tuple, List[str]] = {}
    for did in base["disagreements"]:
        i = pos[did]
        t, c = topics[i], communities[i]
        m = majority[c]
        n_here = pair_counts[(t, c)]
        kind = "lone" if n_here == 1 else "merged" if home[t] == c else "split"

        sims = embeddings @ embeddings[i]
        sims[i] = -np.inf
        nn = np.argsort(-sims)[:k]
        # Leave the document out of its own topic's centroid so it isn't compared with itself.
        own = _unit(sums[t] - embeddings[i]) if topic_sizes[t] > 1 else np.zeros_like(embeddings[i])
        details[did] = {
            "kind": kind,
            "topic": t,
            "community": c,
            "majority_topic": m,
            "home_community": home[t],
            "topic_in_community": n_here,
            "topic_size": topic_sizes[t],
            "neighbors": {
                "k": int(len(nn)),
                "same_topic": int((topic_arr[nn] == t).sum()),
                "same_community": int((comm_arr[nn] == c).sum()),
                "majority_topic": int((topic_arr[nn] == m).sum()),
            },
            "similarity": {
                "own_topic": round(float(embeddings[i] @ own), 4),
                "majority_topic": round(float(embeddings[i] @ _unit(sums[m])), 4),
            },
        }
        groups.setdefault((kind, t, c), []).append(did)

    kind_order = {"lone": 0, "split": 1, "merged": 2}
    group_list = [
        {"kind": kind, "topic": t, "community": c, "majority_topic": majority[c], "home_community": home[t],
         "topic_size": topic_sizes[t], "docs": ids}
        for (kind, t, c), ids in sorted(groups.items(), key=lambda g: (kind_order[g[0][0]], -len(g[1]), g[0][1:]))
    ]
    return {
        "counts": {kind: sum(len(g["docs"]) for g in group_list if g["kind"] == kind) for kind in kind_order},
        "groups": group_list,
        "docs": details,
    }


def passage_topic_mix(
    embeddings: np.ndarray, topics: List[int], chunk_embeddings: np.ndarray, chunk_doc_idx: np.ndarray
) -> List[Dict[int, int]]:
    """
    For each document, how many of its passages (embedding chunks) sit closest
    to each topic's centroid. A case whose passages divide between topics
    covers more than one area of law.
    """
    topic_arr = np.asarray(topics)
    topic_list = sorted(set(topics))
    centroids = np.stack([_unit(embeddings[topic_arr == t].mean(axis=0)) for t in topic_list])
    nearest = np.asarray(topic_list)[np.argmax(chunk_embeddings @ centroids.T, axis=1)]
    mixes: List[Dict[int, int]] = [{} for _ in range(len(embeddings))]
    for di, t in zip(np.asarray(chunk_doc_idx).tolist(), nearest.tolist()):
        mixes[di][t] = mixes[di].get(t, 0) + 1
    return [dict(sorted(m.items(), key=lambda kv: -kv[1])) for m in mixes]


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
