import numpy as np

from citations import build_citation_graph, citation_analysis, doc_citations, extract_citations, strip_citations


def test_extract_normalises_old_reporters():
    text = "See Hyde v. Ruble, 104 U.S. 407, and 5 Wall. 290; also 2 Cranch 170 and 100 U. S. 5."
    assert extract_citations(text) == {(104, 407), (72, 290), (6, 170), (100, 5)}


def test_strip_removes_citations_and_names_but_keeps_prose():
    text = "granted, 300 U.S. 652, 57 S.Ct. 752, in Hyde v. Ruble, 104 U.S. 407, § 10; sentenced to 15 years in 1935."
    out = strip_citations(text)
    assert "U.S." not in out and "Hyde" not in out and "§" not in out
    assert "sentenced to 15 years in 1935" in out


def test_graph_direct_and_cocitation():
    ids = ["10.US.1", "10.US.2", "10.US.3", "10.US.4"]
    texts = [
        "cites 10 U.S. 2 and 99 U.S. 9",   # direct link to doc 2; shares 99 U.S. 9 with doc 3
        "nothing cited here",
        "relies on 99 U.S. 9 and 98 U.S. 1",
        "relies on 98 U.S. 1 and 10 U.S. 4",  # self-citation ignored
    ]
    g = build_citation_graph(ids, texts)
    assert g["direct"] == {frozenset(("10.US.1", "10.US.2"))}
    assert g["shared_count"] == {frozenset(("10.US.1", "10.US.3")): 1, frozenset(("10.US.3", "10.US.4")): 1}
    assert (10, 4) not in g["citations"]["10.US.4"]


def test_analysis_handles_no_links_and_no_tags():
    ids = [f"1.US.{i}" for i in range(6)]
    emb = np.eye(6)
    g = build_citation_graph(ids, ["no citations"] * 6)
    a = citation_analysis(ids, emb, g, [0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 2, 2], tags={})
    assert a["links"]["direct_pairs"] == 0 and a["tags"] is None and a["auc_direct"] is None
    assert a["grouping"]["topic"]["direct"] is None  # nothing to average, not NaN
    assert "direct_partner_rank" not in a


def test_analysis_with_links_and_tags():
    ids = [f"1.US.{i}" for i in range(24)]
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(24, 8)); emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    texts = ["cites 1 U.S. 1" if i == 0 else "cites 50 U.S. 5" if i in (2, 3) else "x" for i in range(24)]
    g = build_citation_graph(ids, texts)
    tags = {d: {"subject_category": "a" if i % 2 else "b", "question_category": "q" if i % 3 else "r"} for i, d in enumerate(ids)}
    a = citation_analysis(ids, emb, g, [i // 12 for i in range(24)], [i // 8 for i in range(24)], tags)
    assert a["links"] == {"docs": 24, "docs_linked": 4, "direct_pairs": 1, "cocitation_pairs": 1, "distinct_cited": 2}
    assert a["direct_partner_rank"]["n"] == 2
    assert set(a["tags"]["similarity_by_shared"]) == {"neither", "subject_only", "question_only", "both"}
    assert a["tags"]["neighbour_share"]["subject"]["chance"] > 0
    cites = doc_citations("1.US.0", g, a)
    assert cites["cites"][0]["id"] == "1.US.1" and isinstance(cites["cites"][0]["rank"], int)
    assert doc_citations("1.US.1", g, a)["cited_by"][0]["id"] == "1.US.0"
