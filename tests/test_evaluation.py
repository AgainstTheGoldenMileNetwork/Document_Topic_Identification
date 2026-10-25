import numpy as np
import pytest

from evaluation import (
    compare_partitions,
    explain_disagreements,
    npmi_coherence,
    passage_topic_mix,
    self_retrieval,
    topic_diversity,
)


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


def test_explain_disagreements_kinds():
    # Community 5: topic 0 (4) + all but one of topic 1 (3)  -> topic 1 is merged in.
    # Community 6: topic 2 (4) + two of topic 3's five        -> those two are split off.
    # Community 7: rest of topic 3 (3) + the last topic-1 doc -> that doc is lone.
    topics = [0] * 4 + [1] * 3 + [2] * 4 + [3] * 2 + [3] * 3 + [1]
    communities = [5] * 7 + [6] * 6 + [7] * 4
    ids = [f"d{i}" for i in range(len(topics))]
    emb = np.random.default_rng(0).normal(size=(len(ids), 8))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)

    result = explain_disagreements(ids, topics, communities, emb, k=5)
    assert result["counts"] == {"lone": 1, "split": 2, "merged": 3}
    assert [g["kind"] for g in result["groups"]] == ["lone", "split", "merged"]
    assert result["groups"][0]["docs"] == ["d16"]
    assert result["groups"][1] == {
        "kind": "split", "topic": 3, "community": 6, "majority_topic": 2,
        "home_community": 7, "topic_size": 5, "docs": ["d11", "d12"],
    }
    lone = result["docs"]["d16"]
    assert lone["topic_in_community"] == 1 and lone["home_community"] == 5
    assert lone["neighbors"]["k"] == 5
    assert set(result["docs"]) == set(compare_partitions(ids, topics, communities)["disagreements"])


def test_passage_topic_mix():
    emb = np.array([[1.0, 0.0], [0.0, 1.0]])
    chunks = np.array([[1.0, 0.1], [0.9, 0.2], [0.1, 1.0], [0.2, 0.9]])
    mix = passage_topic_mix(emb, [0, 1], chunks, np.array([0, 0, 0, 1]))
    assert mix == [{0: 2, 1: 1}, {1: 1}]
