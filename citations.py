"""
Citations between opinions, used as an external check on the embeddings.

Two opinions that cite the same precedent usually share a legal question,
whatever their facts are about. That makes the citation graph a ground truth
of sorts for "related cases" that does not come from the embeddings. To keep
the comparison fair, `strip_citations` removes citations and case names from
the text before it is embedded, so similarity cannot come from the citation
strings themselves.
"""

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

import numpy as np

Cite = Tuple[int, int]  # (U.S. Reports volume, first page)

# Early volumes were cited by reporter name; the U.S. volume is a fixed offset.
NOMINATIVE_OFFSET = {"dall": 0, "cranch": 4, "wheat": 13, "pet": 25, "how": 41, "black": 65, "wall": 67}

REPORTER_RE = re.compile(
    r"\b(\d{1,3})\s+(U\.\s?S\.|Dall\.|Cranch|Wheat\.|Pet\.|How\.|Black|Wall\.)\s*,?\s*(\d{1,4})\b"
)
# "Hyde v. Ruble", "United States v. Symonds", "Board of Comm'rs v. Aspinwall"
_NAME = r"[A-Z][\w.'&-]*(?:\s+(?:of|the|de|&|and|[A-Z][\w.'&-]*)){0,6}"
CASE_NAME_RE = re.compile(rf"\b{_NAME}\s+v\.\s+{_NAME}")
# Any other "volume Reporter page" cite: S.Ct., L.Ed.2d, F.Supp., Stat., state reporters (187 Wash. 364, 61 P.(2d) 293).
OTHER_CITE_RE = re.compile(
    r"\b\d{1,4}\s+(?:[A-Z][A-Za-z]{0,7}\.?\s?){1,3}(?:\(?[23]d\)?\s?)?,?\s*(?:\d{1,4}\b|[—–-]+)"
    r"|§+\s*[\d.]+[\w()-]*"
    r"|\b(?:Id|Ibid)\.(?:,?\s+at\s+\d+)?"
    r"|\bsupra(?:,?\s+at\s+\d+)?"
)


def parse_id(doc_id: str) -> Optional[Cite]:
    """'108.US.30' -> (108, 30). Docket-style ids (591.US.2019_19-465) have no page."""
    m = re.fullmatch(r"(\d+)\.US\.(\d+)", doc_id)
    return (int(m[1]), int(m[2])) if m else None


def extract_citations(text: str) -> Set[Cite]:
    """Every case cited in the text, as (volume, page) in U.S. Reports numbering."""
    out: Set[Cite] = set()
    for vol, reporter, page in REPORTER_RE.findall(text):
        key = reporter.replace(".", "").replace(" ", "").lower()
        offset = 0 if key == "us" else NOMINATIVE_OFFSET[key]
        out.add((int(vol) + offset, int(page)))
    return out


def strip_citations(text: str) -> str:
    """Remove case citations, case names and statute cites, leaving the prose."""
    text = REPORTER_RE.sub(" ", text)
    text = OTHER_CITE_RE.sub(" ", text)
    text = CASE_NAME_RE.sub(" ", text)
    return re.sub(r"[ \t]{2,}", " ", text)


def build_citation_graph(doc_ids: Sequence[str], texts: Sequence[str]) -> Dict:
    """
    Direct links (A cites B, both in the corpus) and co-citation links (A and B
    cite the same precedent, in or out of the corpus). Co-citation weights are
    summed inverse log-frequency, so a shared citation to a precedent that
    everyone cites counts for less than a shared citation to a rare one.
    """
    cites: Dict[str, Set[Cite]] = {}
    by_cite = {parse_id(d): d for d in doc_ids if parse_id(d)}
    for did, text in zip(doc_ids, texts):
        cites[did] = extract_citations(text) - {parse_id(did)}

    direct: Set[FrozenSet[str]] = set()
    for did, cs in cites.items():
        for c in cs:
            if c in by_cite and by_cite[c] != did:
                direct.add(frozenset((did, by_cite[c])))

    cited_by: Dict[Cite, List[str]] = defaultdict(list)
    for did, cs in cites.items():
        for c in cs:
            cited_by[c].append(did)
    n = len(doc_ids)
    cocite: Dict[FrozenSet[str], float] = Counter()
    shared: Dict[FrozenSet[str], int] = Counter()
    for c, ds in cited_by.items():
        if len(ds) < 2:
            continue
        w = 1.0 / (1.0 + math.log(len(ds)))
        for i in range(len(ds)):
            for j in range(i + 1, len(ds)):
                pair = frozenset((ds[i], ds[j]))
                cocite[pair] += w
                shared[pair] += 1

    return {
        "citations": cites,
        "direct": direct,
        "cocitation": dict(cocite),
        "shared_count": dict(shared),
        "num_cited": len(cited_by),
        "num_docs": n,
    }


# ---------------------------------------------------------------------------
# Tags and the comparison with the embeddings
# ---------------------------------------------------------------------------

TAG_FIELDS = ("subject", "subject_category", "legal_question", "question_category", "confidence")


def load_tags(path) -> Dict[str, Dict[str, str]]:
    """Per-case subject / legal-question tags (JSON Lines), or {} if the file is missing."""
    path = Path(path)
    if not path.exists():
        return {}
    tags = {}
    for line in path.read_text().splitlines():
        if line.strip():
            t = json.loads(line)
            tags[str(t["id"])] = {k: t.get(k) for k in TAG_FIELDS}
    return tags


def _auc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    """Probability that a random positive scores above a random negative (rank-based, no sklearn)."""
    if len(scores_pos) == 0 or len(scores_neg) == 0:
        return float("nan")
    allv = np.concatenate([scores_pos, scores_neg])
    ranks = np.empty(len(allv))
    order = np.argsort(allv, kind="mergesort")
    # average ranks for ties
    sorted_v = allv[order]
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n_pos = len(scores_pos)
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * len(scores_neg)))


def _group_share(same: np.ndarray, mask: np.ndarray) -> float:
    return float(same[mask].mean()) if mask.any() else float("nan")


def _no_nan(obj):
    """JSON has no NaN; a statistic with nothing to average over becomes null."""
    if isinstance(obj, dict):
        return {k: _no_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_no_nan(v) for v in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def _category_gaps(S: np.ndarray, cats: np.ndarray, other: np.ndarray) -> List[Dict]:
    rows = []
    for c in sorted(set(cats), key=lambda c: -int((cats == c).sum())):
        m = cats == c
        inside = S[np.ix_(m, m)]
        inside = inside[np.isfinite(inside)]
        outside = S[np.ix_(m, ~m)]
        rows.append({
            "category": c, "n": int(m.sum()), "spans": len(set(other[m])),
            "within": round(float(inside.mean()), 4) if inside.size else None,
            "outside": round(float(outside.mean()), 4) if outside.size else None,
        })
    return rows


def citation_analysis(
    doc_ids: Sequence[str],
    embeddings: np.ndarray,
    graph: Dict,
    topics: Sequence[int],
    communities: Sequence[int],
    tags: Dict[str, Dict[str, str]],
    k: int = 5,
) -> Dict:
    """
    How well embedding similarity, the topics and the communities recover
    the citation graph; and, when tags exist, whether similarity follows a
    shared factual subject or a shared legal question.
    """
    n = len(doc_ids)
    pos = {d: i for i, d in enumerate(doc_ids)}
    S = embeddings @ embeddings.T
    np.fill_diagonal(S, -np.inf)
    L = np.zeros((n, n), dtype=np.int8)  # 0 none, 1 one shared precedent, 2 two or more, 3 direct
    for pair, c in graph["shared_count"].items():
        i, j = (pos[x] for x in pair)
        L[i, j] = L[j, i] = max(L[i, j], 2 if c >= 2 else 1)
    for pair in graph["direct"]:
        i, j = (pos[x] for x in pair)
        L[i, j] = L[j, i] = 3
    iu = np.triu_indices(n, 1)
    lv, sv = L[iu], S[iu]
    topics_a, comms_a = np.asarray(topics), np.asarray(communities)
    same_t, same_c = (topics_a[:, None] == topics_a[None, :])[iu], (comms_a[:, None] == comms_a[None, :])[iu]
    linked_docs = np.flatnonzero((L > 0).any(axis=1))

    out: Dict = {
        "links": {
            "docs": n, "docs_linked": int(len(linked_docs)), "direct_pairs": len(graph["direct"]),
            "cocitation_pairs": len(graph["shared_count"]), "distinct_cited": graph["num_cited"],
        },
        "similarity_by_link": {
            name: {"n": int((lv == level).sum()), "sim": round(float(sv[lv == level].mean()), 4) if (lv == level).any() else None}
            for name, level in (("none", 0), ("shared_1", 1), ("shared_2plus", 2), ("direct", 3))
        },
        "auc_direct": round(_auc(sv[lv == 3], sv[lv == 0]), 4) if (lv == 3).any() else None,
        "grouping": {
            name: {"random": round(_group_share(same, np.ones_like(lv, bool)), 4),
                   "linked": round(_group_share(same, lv > 0), 4), "direct": round(_group_share(same, lv == 3), 4)}
            for name, same in (("topic", same_t), ("community", same_c))
        },
    }
    if len(linked_docs):
        prec = np.mean([(L[i, np.argsort(-S[i])[:k]] > 0).mean() for i in linked_docs])
        chance = np.mean([(L[i] > 0).sum() / (n - 1) for i in linked_docs])
        out["precision_at_k"] = {"k": k, "value": round(float(prec), 4), "chance": round(float(chance), 4)}

    ranks, rank_rows = [], []
    for pair in graph["direct"]:
        a, b = (pos[x] for x in pair)
        for i, j in ((a, b), (b, a)):
            r = 1 + int((S[i] > S[i, j]).sum())
            ranks.append(r)
            rank_rows.append((doc_ids[i], doc_ids[j], r))
    if ranks:
        ra = np.asarray(ranks)
        out["direct_partner_rank"] = {"n": len(ra), "median": float(np.median(ra)), "top10": round(float((ra <= 10).mean()), 4),
                                      "top50": round(float((ra <= 50).mean()), 4), "random_top10": round(10 / max(n - 1, 1), 4)}
    out["partner_rank"] = {f"{a}|{b}": r for a, b, r in rank_rows}

    tagged = [d for d in doc_ids if d in tags and tags[d].get("subject_category") and tags[d].get("question_category")]
    if len(tagged) < 0.9 * n or n < 20:
        out["tags"] = None
        return _no_nan(out)

    subj = np.array([tags.get(d, {}).get("subject_category") or "untagged" for d in doc_ids])
    ques = np.array([tags.get(d, {}).get("question_category") or "untagged" for d in doc_ids])
    same_s, same_q = (subj[:, None] == subj[None, :])[iu], (ques[:, None] == ques[None, :])[iu]
    nn = np.argsort(-S, axis=1)[:, :k]
    shared = {
        "neither": ~same_s & ~same_q, "subject_only": same_s & ~same_q, "question_only": ~same_s & same_q, "both": same_s & same_q,
    }
    rank_by = {}
    for pair in graph["direct"]:
        a, b = (pos[x] for x in pair)
        key = {(True, True): "both", (True, False): "subject_only", (False, True): "question_only", (False, False): "neither"}[
            (subj[a] == subj[b], ques[a] == ques[b])]
        for i, j in ((a, b), (b, a)):
            rank_by.setdefault(key, []).append(1 + int((S[i] > S[i, j]).sum()))
    out["tags"] = {
        "n_tagged": len(tagged),
        "similarity_by_shared": {
            name: {"n": int(m.sum()), "sim": round(float(sv[m].mean()), 4) if m.any() else None, "linked": round(float((lv[m] > 0).mean()), 5) if m.any() else None}
            for name, m in shared.items()
        },
        "neighbour_share": {
            "k": k,
            "subject": {"value": round(float(np.mean([(subj[nn[i]] == subj[i]).mean() for i in range(n)])), 4), "chance": round(float(same_s.mean()), 4)},
            "question": {"value": round(float(np.mean([(ques[nn[i]] == ques[i]).mean() for i in range(n)])), 4), "chance": round(float(same_q.mean()), 4)},
        },
        "auc": {"subject": round(_auc(sv[same_s], sv[~same_s]), 4), "question": round(_auc(sv[same_q], sv[~same_q]), 4)},
        "citation_share": {
            name: {"subject": round(_group_share(same_s, m), 4), "question": round(_group_share(same_q, m), 4)}
            for name, m in (("random", np.ones_like(lv, bool)), ("cocitation", (lv == 1) | (lv == 2)), ("direct", lv == 3))
        },
        "grouping_share": {
            name: {"subject": round(_group_share(same_s, m), 4), "question": round(_group_share(same_q, m), 4)}
            for name, m in (("topic", same_t), ("community", same_c))
        },
        "partner_rank_by_shared": {
            key: {"n": len(v), "median": float(np.median(v)), "top10": round(float(np.mean(np.asarray(v) <= 10)), 4)}
            for key, v in rank_by.items()
        },
        "question_categories": _category_gaps(S, ques, subj),
        "subject_categories": _category_gaps(S, subj, ques),
    }
    return _no_nan(out)


def doc_citations(doc_id: str, graph: Dict, analysis: Dict, limit: int = 10) -> Dict:
    """A case's in-corpus citation partners, with where the embedding ranked each one."""
    cites, cited_by, cocited = [], [], []
    ranks = analysis.get("partner_rank", {})
    for pair in graph["direct"]:
        if doc_id in pair:
            other = next(x for x in pair if x != doc_id)
            # direction: does this doc's text cite the other?
            target = parse_id(other)
            if target and target in graph["citations"].get(doc_id, set()):
                cites.append({"id": other, "rank": ranks.get(f"{doc_id}|{other}")})
            else:
                cited_by.append({"id": other, "rank": ranks.get(f"{doc_id}|{other}")})
    direct_ids = {p["id"] for p in cites + cited_by}
    for pair, count in graph["shared_count"].items():
        if doc_id in pair:
            other = next(x for x in pair if x != doc_id)
            if other not in direct_ids:  # listed once, under the stronger link
                cocited.append({"id": other, "shared": count})
    cocited.sort(key=lambda c: -c["shared"])
    return {"cites": cites, "cited_by": cited_by, "cocited": cocited[:limit], "num_citations": len(graph["citations"].get(doc_id, ()))}
