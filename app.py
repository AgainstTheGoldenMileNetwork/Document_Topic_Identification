import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, RedirectResponse

from config import get_settings
from models import (
    ClusterDetail,
    ClusterSummary,
    CommunityDetail,
    Comparison,
    CitationAnalysis,
    DisagreementAnalysis,
    Document,
    GraphInfo,
    SearchResult,
    SimilarDocument,
    SimilarityEdge,
)
from citations import doc_citations
from pipeline import CorpusIndex, build_index

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app")

STATIC_DIR = Path(__file__).parent / "static"

_index: CorpusIndex | None = None


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _index
    _index = build_index(get_settings())
    info = _index.graph.get_info()
    logger.info(
        "Ready — %d docs, %d topics, %d edges, %d communities",
        len(_index.docs), len(_index.topics), info["num_edges"], info["communities"],
    )
    yield
    _index = None


app = FastAPI(
    title="Legal Topic Sorter",
    description="Cluster legal case opinions into topics using BERTopic, "
    "with a similarity graph for finding related cases.",
    version="3.0.0",
    lifespan=lifespan,
)


def index() -> CorpusIndex:
    if _index is None:
        raise HTTPException(status_code=503, detail="Index not ready")
    return _index


def doc_or_404(doc_id: str) -> Dict[str, Any]:
    doc = index().docs_by_id.get(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


# ---------------------------------------------------------------------------
# Explorer UI
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/explorer")


@app.get("/explorer", include_in_schema=False)
def explorer():
    return FileResponse(STATIC_DIR / "explorer.html")


@app.get("/health")
def health():
    idx = _index
    return {
        "status": "ok" if idx else "starting",
        "docs": len(idx.docs) if idx else 0,
        "clusters": len(idx.topics) if idx else 0,
        "communities": len(idx.graph.communities) if idx else 0,
    }


# ---------------------------------------------------------------------------
# Topics (BERTopic)
# ---------------------------------------------------------------------------

@app.get("/clusters", response_model=List[ClusterSummary])
def list_clusters():
    return [ClusterSummary(cluster_id=cid, **info) for cid, info in sorted(index().topics.items())]


@app.get("/clusters/{cluster_id}", response_model=ClusterDetail)
def get_cluster(cluster_id: int):
    idx = index()
    if cluster_id not in idx.topics:
        raise HTTPException(status_code=404, detail="Cluster not found")
    info = idx.topics[cluster_id]
    docs = [d for d in idx.docs if d.get("cluster_id") == cluster_id]
    return ClusterDetail(
        cluster_id=cluster_id,
        label=info["label"],
        summary=info["summary"],
        top_terms=info["top_terms"],
        documents=[
            {
                "id": d.get("id"),
                "title": d.get("title"),
                "court": d.get("court"),
                "date": d.get("date"),
                "text": d.get("text", "")[:1000] + "...",
            }
            for d in docs
        ],
    )


@app.get("/documents/{doc_id}", response_model=Document)
def get_document(doc_id: str):
    d = doc_or_404(doc_id)
    return Document(
        id=str(d["id"]),
        title=d.get("title"),
        court=d.get("court"),
        date=d.get("date"),
        syllabus=d.get("syllabus"),
        text=d.get("text"),
        cluster_id=d.get("cluster_id"),
        community_id=d.get("community_id"),
        passage_topics=d.get("passage_topics", {}),
        subject=d.get("subject"),
        subject_category=d.get("subject_category"),
        legal_question=d.get("legal_question"),
        question_category=d.get("question_category"),
        citations=doc_citations(doc_id, index().citation_graph, index().citation_analysis),
    )


# ---------------------------------------------------------------------------
# Search and similarity
# ---------------------------------------------------------------------------

@app.get("/search", response_model=List[SearchResult])
def search(
    q: str = Query(..., min_length=2, max_length=1000, description="Plain-language description of a case"),
    top_k: int = Query(default=10, ge=1, le=50),
):
    """Semantic search: documents ranked by their best-matching passage."""
    return [SearchResult(**r) for r in index().search(q, top_k=top_k)]


@app.get("/documents/{doc_id}/similar", response_model=List[SimilarDocument])
def get_similar_documents(doc_id: str, top_k: int = Query(default=10, ge=1, le=50)):
    doc_or_404(doc_id)
    idx = index()
    return [
        SimilarDocument(id=did, title=idx.docs_by_id[did].get("title"), similarity=round(sim, 4))
        for did, sim in idx.graph.get_similar(doc_id, top_k=top_k)
    ]


# ---------------------------------------------------------------------------
# Graph endpoints
# ---------------------------------------------------------------------------

@app.get("/graph/info", response_model=GraphInfo)
def graph_info():
    return GraphInfo(**index().graph.get_info())


@app.get("/graph/stats")
def graph_stats():
    """Similarity distribution stats — useful for picking a good threshold."""
    return index().graph.get_sim_stats()


@app.get("/graph/communities", response_model=List[CommunityDetail])
def list_communities():
    return [
        CommunityDetail(community_id=i, document_ids=members)
        for i, members in enumerate(index().graph.communities)
    ]


@app.get("/graph/communities/{community_id}", response_model=CommunityDetail)
def get_community(community_id: int):
    communities = index().graph.communities
    if community_id < 0 or community_id >= len(communities):
        raise HTTPException(status_code=404, detail="Community not found")
    return CommunityDetail(community_id=community_id, document_ids=communities[community_id])


@app.get("/graph/edges", response_model=List[SimilarityEdge])
def list_edges(
    min_weight: float | None = Query(default=None, ge=-1.0, le=1.0, description="Omit to include every graph edge"),
    limit: int = Query(default=100, ge=1, le=1000),
):
    return [SimilarityEdge(**e) for e in index().graph.get_edges(min_weight=min_weight, limit=limit)]


@app.get("/graph/export")
def graph_export(max_edges: int = Query(default=4000, ge=0, le=50000)):
    """Everything the explorer needs in one payload: map positions, both groupings, strongest edges."""
    idx = index()
    coords = idx.coords
    span = (coords.max(axis=0) - coords.min(axis=0)).clip(min=1e-9)
    normed = (coords - coords.min(axis=0)) / span
    kinds = {did: info["kind"] for did, info in idx.disagreements["docs"].items()}
    return {
        "nodes": [
            {
                "id": str(d["id"]),
                "title": d.get("title"),
                "date": d.get("date"),
                "x": round(float(normed[i, 0]), 4),
                "y": round(float(normed[i, 1]), 4),
                "topic": d["cluster_id"],
                "community": d["community_id"],
                "disagrees": str(d["id"]) in kinds,
                "kind": kinds.get(str(d["id"]), "agree"),
                "subject": d.get("subject_category"),
                "question": d.get("question_category"),
            }
            for i, d in enumerate(idx.docs)
        ],
        "edges": [
            [e["source"], e["target"], round(e["weight"], 4)]
            for e in idx.graph.get_edges(limit=max_edges)
        ] if max_edges else [],
        "topics": {
            str(cid): {"label": t["label"], "label_source": t["label_source"], "size": t["size"], "top_terms": t["top_terms"]}
            for cid, t in sorted(idx.topics.items())
        },
        "graph": idx.graph.get_info(),
        "comparison": {k: v for k, v in idx.comparison.items() if k != "disagreements"},
        "disagreements": idx.disagreements,
        "citations": {
            "direct": [sorted(p) for p in idx.citation_graph["direct"]],
            "cocited": [[*sorted(p), c] for p, c in idx.citation_graph["shared_count"].items()],
            "analysis": {k: v for k, v in idx.citation_analysis.items() if k != "partner_rank"},
        },
        "settings": {
            "embed_model": idx.settings.embed_model,
            "embed_strategy": idx.settings.embed_strategy,
            "outlier_fraction": idx.outlier_fraction,
            "strip_citations": idx.settings.strip_citations,
        },
    }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

@app.get("/analysis/comparison", response_model=Comparison)
def comparison():
    """How closely BERTopic topics and Louvain communities agree, and where they don't."""
    return Comparison(**index().comparison)


@app.get("/analysis/disagreements", response_model=DisagreementAnalysis)
def disagreements():
    """
    Each disagreeing case sorted into lone / split / merged, with the evidence:
    nearest-neighbour votes and similarity to its topic vs its community's majority topic.
    """
    return DisagreementAnalysis(**index().disagreements)


@app.get("/analysis/citations", response_model=CitationAnalysis)
def citations():
    """
    How well similarity, topics and communities recover the citation graph, and
    (with tags) whether similarity follows shared facts or a shared legal question.
    """
    return CitationAnalysis(**{k: v for k, v in index().citation_analysis.items() if k != "partner_rank"})
