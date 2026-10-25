import pytest
from fastapi.testclient import TestClient

import app as app_module
from pipeline import build_index
from tests.conftest import FakeEmbedder


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import json
    from tests.conftest import make_corpus

    tmp = tmp_path_factory.mktemp("api")
    docs_path = tmp / "docs.jsonl"
    docs_path.write_text("\n".join(json.dumps(d) for d in make_corpus()) + "\n")

    mp = pytest.MonkeyPatch()
    mp.setenv("DOCS_PATH", str(docs_path))
    mp.setenv("CACHE_DIR", str(tmp / "cache"))
    mp.setenv("SIM_PERCENTILE", "85")
    mp.delenv("ENABLE_LLM_LABELS", raising=False)
    mp.setattr(app_module, "build_index", lambda settings: build_index(settings, embedder=FakeEmbedder()))
    with TestClient(app_module.app) as c:
        yield c
    mp.undo()


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["docs"] == 48 and body["clusters"] >= 2


def test_clusters_recover_themes(client):
    clusters = client.get("/clusters").json()
    assert sum(c["size"] for c in clusters) == 48
    for c in clusters:
        assert c["label"] == c["summary"] and c["label_source"] == "keywords"
        assert c["representative_docs"]
    # Every theme's distinctive vocabulary shows up in some topic's keywords.
    all_terms = {t for c in clusters for t in c["top_terms"]}
    assert {"patent", "vessel", "tax", "jury"} & all_terms
    detail = client.get(f"/clusters/{clusters[0]['cluster_id']}").json()
    assert detail["documents"]
    assert client.get("/clusters/999").status_code == 404


def test_document_and_similar(client):
    doc = client.get("/documents/patent-0").json()
    assert doc["syllabus"].startswith("A patent dispute")
    assert doc["community_id"] is not None
    similar = client.get("/documents/patent-0/similar?top_k=5").json()
    assert len(similar) == 5
    assert sum(s["id"].startswith("patent") for s in similar) >= 4
    assert client.get("/documents/nope").status_code == 404
    assert client.get("/documents/nope/similar").status_code == 404


def test_search_returns_passages(client):
    results = client.get("/search", params={"q": "ship cargo voyage collision", "top_k": 5}).json()
    assert len(results) == 5
    assert results[0]["id"].startswith("maritime")
    assert results[0]["snippet"]
    assert client.get("/search", params={"q": "x"}).status_code == 422


def test_graph_endpoints(client):
    info = client.get("/graph/info").json()
    assert info["num_nodes"] == 48 and info["num_edges"] > 0
    edges = client.get("/graph/edges?limit=5").json()
    assert len(edges) == 5 and edges[0]["weight"] >= edges[-1]["weight"]
    communities = client.get("/graph/communities").json()
    assert sum(len(c["document_ids"]) for c in communities) == 48
    assert client.get("/graph/communities/999").status_code == 404
    assert client.get("/graph/stats").json()["threshold_percentile"] == 85


def test_export_and_comparison(client):
    export = client.get("/graph/export").json()
    assert len(export["nodes"]) == 48
    node = export["nodes"][0]
    assert 0 <= node["x"] <= 1 and 0 <= node["y"] <= 1
    assert {"topic", "community", "disagrees", "kind"} <= node.keys()
    assert all(n["disagrees"] == (n["kind"] != "agree") for n in export["nodes"])
    comparison = client.get("/analysis/comparison").json()
    assert -1 <= comparison["ari"] <= 1
    assert len(comparison["disagreements"]) == round(comparison["disagreement_rate"] * 48)


def test_disagreement_analysis(client):
    analysis = client.get("/analysis/disagreements").json()
    flagged = client.get("/analysis/comparison").json()["disagreements"]
    assert sum(analysis["counts"].values()) == len(flagged)
    assert set(analysis["docs"]) == set(flagged)
    assert sum(len(g["docs"]) for g in analysis["groups"]) == len(flagged)
    doc = client.get("/documents/patent-0").json()
    assert sum(doc["passage_topics"].values()) >= 1


def test_explorer_served(client):
    res = client.get("/explorer")
    assert res.status_code == 200 and "Legal Topic Explorer" in res.text
    assert client.get("/", follow_redirects=False).status_code == 307
