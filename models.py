from typing import Any, Dict, List

from pydantic import BaseModel


class ClusterSummary(BaseModel):
    cluster_id: int
    summary: str
    top_terms: List[str]
    example_docs: List[str]


class ClusterDetail(BaseModel):
    cluster_id: int
    summary: str
    top_terms: List[str]
    documents: List[Dict[str, Any]]


class Document(BaseModel):
    id: str
    title: str | None
    court: Any
    date: str | None
    text: str
    cluster_id: int | None = None


class SimilarDocument(BaseModel):
    id: str
    title: str | None
    similarity: float


class SimilarityEdge(BaseModel):
    source: str
    target: str
    weight: float


class GraphInfo(BaseModel):
    num_nodes: int
    num_edges: int
    density: float
    communities: int


class CommunityDetail(BaseModel):
    community_id: int
    document_ids: List[str]
