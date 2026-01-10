from pathlib import Path
from typing import List, Dict, Any
from keybert import KeyBERT

import json
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_PATH = Path("data/documents.jsonl")

app = FastAPI(
    title="Legal Topic Sorter",
    description="Cluster legal case opinions into topics and expose via API.",
    version="1.0.0",
)

_docs: List[Dict[str, Any]] = []
_clusters: Dict[int, Dict[str, Any]] = {}
_model: SentenceTransformer | None = None
_kw_model: KeyBERT | None = None
_summarizer = None


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


def load_docs() -> List[Dict[str, Any]]:
    if not DATA_PATH.exists():
        raise RuntimeError(f"{DATA_PATH} does not exist. Run download_docs.py first.")
    docs = []
    with DATA_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    return docs


def compute_embeddings(texts: List[str]) -> np.ndarray:
    global _model
    if _model is None:
        # Small, fast sentence embedding model
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = _model.encode(texts, batch_size=16, show_progress_bar=True)
    return np.array(embeddings)

def get_kw_model() -> KeyBERT:
    global _kw_model
    if _kw_model is None:
        # Use the same sentence-transformer under the hood
        _kw_model = KeyBERT(model="all-MiniLM-L6-v2")
    return _kw_model

def get_summarizer():
    global _summarizer
    if _summarizer is None:
        # First call will download the model; after that it’s cached
        _summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return _summarizer


def cluster_docs(docs: List[Dict[str, Any]], k: int = 5):
    """
    Simple k-means clustering into k clusters.
    """
    texts = [d["text"] for d in docs]
    embeddings = compute_embeddings(texts)

    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(embeddings)

    cluster_topics: Dict[int, Dict[str, Any]] = {}
    unique_labels = sorted(set(int(l) for l in labels))
    kw_model = get_kw_model()
    summarizer = get_summarizer()

    for cid in unique_labels:
        idx = np.where(labels == cid)[0]
        cluster_docs = [docs[i] for i in idx]
        # Combine texts in this cluster
        combined_text = "\n\n".join(d["text"] for d in cluster_docs)

        # Extract transformer-based keyphrases
        keywords = kw_model.extract_keywords(
            combined_text,
            keyphrase_ngram_range=(1, 3),  # allow up to 3-word phrases
            stop_words="english",
            top_n=10,
        )
        # keywords is list of (phrase, score)
        top_terms = [phrase for phrase, score in keywords]

        label_input = ". ".join(top_terms)
        summary = summarizer(
            label_input,
            max_length=12,   # very short; think "Municipal bond authority"
            min_length=3,
            do_sample=False,
        )[0]["summary_text"]

        example_doc_ids = [d["id"] for d in cluster_docs[:10]]

        cluster_topics[cid] = {
            "top_terms": top_terms,         # whatever you already compute
            "summary": summary,             # NEW contextual label
            "example_docs": example_doc_ids,
        }

    # Attach cluster_id to docs
    for doc, label in zip(docs, labels):
        doc["cluster_id"] = int(label)

    return docs, cluster_topics


@app.on_event("startup")
def startup_event():
    global _docs, _clusters
    print("Loading documents and computing clusters...")
    _docs = load_docs()
    _docs, _clusters = cluster_docs(_docs, k=5)
    print(f"Loaded {len(_docs)} documents into {len(_clusters)} clusters.")


@app.get("/health")
def health():
    return {"status": "ok", "docs": len(_docs), "clusters": len(_clusters)}


@app.get("/clusters", response_model=List[ClusterSummary])
def list_clusters():
    result: List[ClusterSummary] = []
    for cid, info in sorted(_clusters.items(), key=lambda kv: kv[0]):
        result.append(
            ClusterSummary(
                cluster_id=cid,
                summary=info["summary"],          # <-- add this
                top_terms=info["top_terms"],
                example_docs=info["example_docs"]  # <-- fix key name
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
        summary=info["summary"],          # <-- add this
        top_terms=info["top_terms"],
        documents=[
            {
                "id": d.get("id"),
                "title": d.get("title"),
                "court": d.get("court"),
                "date": d.get("date"),
                "text": d.get("text")[:1000] + "..."
            }
            for d in docs
        ],
    )


@app.get("/documents/{doc_id}", response_model=Document)
def get_document(doc_id: str):
    for d in _docs:
        if str(d.get("id")) == doc_id:
            return Document(
                id=str(d.get("id")),
                title=d.get("title"),
                court=d.get("court"),
                date=d.get("date"),
                text=d.get("text"),
                cluster_id=d.get("cluster_id"),
            )
    raise HTTPException(status_code=404, detail="Document not found")