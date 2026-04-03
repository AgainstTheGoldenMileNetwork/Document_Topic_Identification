from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Query

from data_loader import compute_embeddings, load_docs
from models import (
    ClusterDetail,
    ClusterSummary,
    CommunityDetail,
    Document,
    GraphInfo,
    SimilarDocument,
    SimilarityEdge,
)
from similarity import SimilarityGraph
from topic_modeling import run_topic_modeling

app = FastAPI(
    title="Legal Topic Sorter",
    description="Cluster legal case opinions into topics using BERTopic, "
    "with a similarity graph for finding related cases.",
    version="2.0.0",
)

_docs: List[Dict[str, Any]] = []
_clusters: Dict[int, Dict[str, Any]] = {}
_sim_graph: SimilarityGraph | None = None


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
def startup_event():
    global _docs, _clusters, _sim_graph

    print("Loading documents...")
    _docs = load_docs()
    texts = [d["text"] for d in _docs]

    print("Computing embeddings...")
    embeddings = compute_embeddings(texts)

    print("Running BERTopic...")
    _docs, _clusters = run_topic_modeling(_docs, embeddings)

    print("Building similarity graph...")
    doc_ids = [str(d["id"]) for d in _docs]
    _sim_graph = SimilarityGraph(doc_ids, embeddings, threshold=0.42)

    info = _sim_graph.get_info()
    print(
        f"Done — {len(_docs)} docs, {len(_clusters)} topics, "
        f"{info['num_edges']} edges, {info['communities']} communities"
    )


# ---------------------------------------------------------------------------
# Existing endpoints (now BERTopic-powered)
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    info = _sim_graph.get_info() if _sim_graph else {}
    return {
        "status": "ok",
        "docs": len(_docs),
        "clusters": len(_clusters),
        "communities": info.get("communities", 0),
    }


@app.get("/clusters", response_model=List[ClusterSummary])
def list_clusters():
    result: List[ClusterSummary] = []
    for cid, info in sorted(_clusters.items()):
        result.append(
            ClusterSummary(
                cluster_id=cid,
                summary=info["summary"],
                top_terms=info["top_terms"],
                example_docs=info["example_docs"],
            )
        )
    return result


@app.get("/clusters/{cluster_id}", response_model=ClusterDetail)
def get_cluster(cluster_id: int):
    if cluster_id not in _clusters:
        raise HTTPException(status_code=404, detail="Cluster not found")
    info = _clusters[cluster_id]
    docs = [d for d in _docs if d.get("cluster_id") == cluster_id]
    return ClusterDetail(
        cluster_id=cluster_id,
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
    for d in _docs:
        if str(d.get("id")) == doc_id:
            return Document(
                id=str(d["id"]),
                title=d.get("title"),
                court=d.get("court"),
                date=d.get("date"),
                text=d.get("text"),
                cluster_id=d.get("cluster_id"),
            )
    raise HTTPException(status_code=404, detail="Document not found")


# ---------------------------------------------------------------------------
# Similarity endpoints
# ---------------------------------------------------------------------------

@app.get("/documents/{doc_id}/similar", response_model=List[SimilarDocument])
def get_similar_documents(doc_id: str, top_k: int = Query(default=10, ge=1, le=50)):
    if _sim_graph is None:
        raise HTTPException(status_code=503, detail="Graph not ready")
    pairs = _sim_graph.get_similar(doc_id, top_k=top_k)
    if not pairs and not any(str(d.get("id")) == doc_id for d in _docs):
        raise HTTPException(status_code=404, detail="Document not found")
    # Look up titles
    id_to_doc = {str(d["id"]): d for d in _docs}
    return [
        SimilarDocument(
            id=did,
            title=id_to_doc.get(did, {}).get("title"),
            similarity=round(sim, 4),
        )
        for did, sim in pairs
    ]


# ---------------------------------------------------------------------------
# Graph endpoints
# ---------------------------------------------------------------------------

@app.get("/graph/info", response_model=GraphInfo)
def graph_info():
    if _sim_graph is None:
        raise HTTPException(status_code=503, detail="Graph not ready")
    return GraphInfo(**_sim_graph.get_info())


@app.get("/graph/stats")
def graph_stats():
    """Similarity distribution stats — useful for picking a good threshold."""
    if _sim_graph is None:
        raise HTTPException(status_code=503, detail="Graph not ready")
    return _sim_graph.get_sim_stats()


@app.get("/graph/communities", response_model=List[CommunityDetail])
def list_communities():
    if _sim_graph is None:
        raise HTTPException(status_code=503, detail="Graph not ready")
    return [
        CommunityDetail(community_id=i, document_ids=members)
        for i, members in enumerate(_sim_graph.communities)
    ]


@app.get("/graph/communities/{community_id}", response_model=CommunityDetail)
def get_community(community_id: int):
    if _sim_graph is None:
        raise HTTPException(status_code=503, detail="Graph not ready")
    if community_id < 0 or community_id >= len(_sim_graph.communities):
        raise HTTPException(status_code=404, detail="Community not found")
    return CommunityDetail(
        community_id=community_id,
        document_ids=_sim_graph.communities[community_id],
    )


@app.get("/graph/edges", response_model=List[SimilarityEdge])
def list_edges(
    min_weight: float = Query(default=0.42, ge=0.0, le=1.0),
    limit: int = Query(default=100, ge=1, le=1000),
):
    if _sim_graph is None:
        raise HTTPException(status_code=503, detail="Graph not ready")
    edges = _sim_graph.get_edges(min_weight=min_weight, limit=limit)
    return [SimilarityEdge(**e) for e in edges]
