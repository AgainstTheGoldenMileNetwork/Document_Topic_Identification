import numpy as np
import pytest

from similarity import SimilarityGraph


@pytest.fixture
def graph():
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(3, 16))
    emb = np.vstack([c + 0.3 * rng.normal(size=(10, 16)) for c in centers])
    ids = [f"d{i}" for i in range(len(emb))]
    return SimilarityGraph(ids, emb, percentile=70)


def test_percentile_threshold_keeps_expected_share_of_pairs(graph):
    n = len(graph.doc_ids)
    pairs = n * (n - 1) // 2
    assert graph.graph.number_of_edges() == pytest.approx(0.3 * pairs, abs=2)
    assert graph.get_sim_stats()["threshold_percentile"] == 70


def test_explicit_threshold_wins():
    emb = np.eye(4)
    g = SimilarityGraph(["a", "b", "c", "d"], emb, threshold=0.5, percentile=10, knn=0)
    assert g.threshold == 0.5
    assert g.graph.number_of_edges() == 0
    assert len(g.communities) == 4


def test_get_similar_excludes_self_and_is_sorted(graph):
    results = graph.get_similar("d0", top_k=5)
    assert len(results) == 5
    assert all(did != "d0" for did, _ in results)
    sims = [s for _, s in results]
    assert sims == sorted(sims, reverse=True)
    assert graph.get_similar("missing") == []


def test_communities_recover_planted_clusters(graph):
    labels = graph.community_labels()
    assert len(labels) == 30
    # Each planted cluster of 10 lands mostly in one community.
    for start in (0, 10, 20):
        block = labels[start:start + 10]
        assert max(block.count(c) for c in set(block)) >= 8
    assert graph.modularity() > 0.3


def test_stats_match_brute_force(graph):
    n = len(graph.doc_ids)
    brute = [graph.sim_matrix[i, j] for i in range(n) for j in range(i + 1, n)]
    stats = graph.get_sim_stats()
    assert stats["mean"] == pytest.approx(float(np.mean(brute)))
    assert stats["edges_at_threshold"] == sum(s >= graph.threshold for s in brute)


def test_edges_default_to_threshold_and_are_sorted(graph):
    edges = graph.get_edges(limit=1000)
    assert len(edges) == graph.graph.number_of_edges()
    weights = [e["weight"] for e in edges]
    assert weights == sorted(weights, reverse=True)
    assert min(weights) >= graph.threshold


def test_knn_edges_leave_no_isolated_docs():
    emb = np.eye(4) + 0.01
    g = SimilarityGraph(["a", "b", "c", "d"], emb, threshold=0.99, knn=1)
    assert all(d >= 1 for _, d in g.graph.degree())
    assert g.get_sim_stats()["edges_at_threshold"] == 0
