import numpy as np
import pytest

from evaluation import compare_partitions, npmi_coherence, self_retrieval, topic_diversity


def test_self_retrieval_perfect_and_skips_invalid():
    emb = np.eye(5)
    result = self_retrieval(emb, emb, [True, True, True, False, True])
    assert result["queries"] == 4
    assert result["mrr"] == 1.0 and result["recall_at_1"] == 1.0


def test_self_retrieval_rank():
    docs = np.array([[1.0, 0.0], [0.0, 1.0]])
    queries = np.array([[0.0, 1.0], [0.0, 1.0]])  # query 0 prefers doc 1
    result = self_retrieval(docs, queries, [True, True])
    assert result["recall_at_1"] == 0.5
    assert result["mrr"] == pytest.approx(0.75)


def test_compare_partitions():
    ids = ["a", "b", "c", "d"]
    same = compare_partitions(ids, [0, 0, 1, 1], [5, 5, 7, 7])
    assert same["ari"] == pytest.approx(1.0)
    assert same["disagreements"] == []
    mixed = compare_partitions(ids, [0, 0, 0, 1], [5, 5, 5, 5])
    assert mixed["disagreements"] == ["d"]
    assert mixed["community_majority_topic"] == {"5": 0}
    assert mixed["contingency"] == {"5": {"0": 3, "1": 1}}


def test_npmi_coherence_bounds():
    texts = ["vessel cargo", "vessel cargo", "patent claim", "patent claim"]
    assert npmi_coherence([["vessel", "cargo"]], texts) == pytest.approx(1.0)
    assert npmi_coherence([["vessel", "patent"]], texts) == -1.0


def test_topic_diversity():
    assert topic_diversity([["a", "b"], ["c", "d"]]) == 1.0
    assert topic_diversity([["a", "b"], ["a", "b"]]) == 0.5
