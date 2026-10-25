from typing import Any, Dict, List

from pydantic import BaseModel


class ClusterSummary(BaseModel):
    cluster_id: int
    label: str
    label_source: str
    summary: str
    size: int
    top_terms: List[str]
    example_docs: List[str]
    representative_docs: List[str]


class ClusterDetail(BaseModel):
    cluster_id: int
    label: str
    summary: str
    top_terms: List[str]
    documents: List[Dict[str, Any]]


class Document(BaseModel):
    id: str
    title: str | None
    court: Any
    date: str | None
    syllabus: str | None = None
    text: str
    cluster_id: int | None = None
    community_id: int | None = None
    passage_topics: Dict[int, int] = {}
    subject: str | None = None
    subject_category: str | None = None
    legal_question: str | None = None
    question_category: str | None = None
    citations: Dict[str, Any] | None = None


class SimilarDocument(BaseModel):
    id: str
    title: str | None
    similarity: float


class SearchResult(BaseModel):
    id: str
    title: str | None
    score: float
    cluster_id: int | None
    snippet: str


class SimilarityEdge(BaseModel):
    source: str
    target: str
    weight: float


class GraphInfo(BaseModel):
    num_nodes: int
    num_edges: int
    density: float
    communities: int
    modularity: float
    threshold: float
    knn: int


class CommunityDetail(BaseModel):
    community_id: int
    document_ids: List[str]


class Comparison(BaseModel):
    ari: float
    nmi: float
    num_topics: int
    num_communities: int
    disagreement_rate: float
    community_majority_topic: Dict[str, int]
    contingency: Dict[str, Dict[str, int]]
    disagreements: List[str]


class DisagreementGroup(BaseModel):
    kind: str
    topic: int
    community: int
    majority_topic: int
    home_community: int
    topic_size: int
    docs: List[str]


class DisagreementDetail(BaseModel):
    kind: str
    topic: int
    community: int
    majority_topic: int
    home_community: int
    topic_in_community: int
    topic_size: int
    neighbors: Dict[str, int]
    similarity: Dict[str, float]


class DisagreementAnalysis(BaseModel):
    counts: Dict[str, int]
    groups: List[DisagreementGroup]
    docs: Dict[str, DisagreementDetail]


class CitationAnalysis(BaseModel):
    links: Dict[str, int]
    similarity_by_link: Dict[str, Dict[str, Any]]
    auc_direct: float | None
    grouping: Dict[str, Dict[str, float | None]]
    precision_at_k: Dict[str, float] | None = None
    direct_partner_rank: Dict[str, float] | None = None
    tags: Dict[str, Any] | None
