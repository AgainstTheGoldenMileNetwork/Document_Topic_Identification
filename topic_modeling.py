from typing import Any, Dict, List, Tuple

import numpy as np
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP


def build_topic_model() -> BERTopic:
    """
    Configure BERTopic with tuned UMAP/HDBSCAN params for a small corpus (~100 docs).
    """
    umap_model = UMAP(
        n_neighbors=10,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=5,
        min_samples=2,
        metric="euclidean",
        prediction_data=True,
    )
    vectorizer_model = CountVectorizer(stop_words="english")
    model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        verbose=True,
    )
    return model


def run_topic_modeling(
    docs: List[Dict[str, Any]],
    embeddings: np.ndarray,
) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    """
    Fit BERTopic on the documents using precomputed embeddings.
    Returns (docs_with_cluster_ids, topics_dict).
    """
    texts = [d["text"] for d in docs]
    model = build_topic_model()
    topics, _probs = model.fit_transform(texts, embeddings=embeddings)

    # Reassign outliers (topic -1) to the nearest real topic
    topics = model.reduce_outliers(texts, topics, strategy="distributions")

    # Build topics dict keyed by cluster_id
    topic_info = model.get_topic_info()
    topics_dict: Dict[int, Dict[str, Any]] = {}

    for _, row in topic_info.iterrows():
        tid = int(row["Topic"])
        if tid == -1:
            continue  # skip outlier meta-topic after reassignment

        # c-TF-IDF top terms from BERTopic
        topic_words = model.get_topic(tid)
        top_terms = [word for word, _score in topic_words[:10]]

        # Human-readable label from top 3 keywords
        summary = " / ".join(top_terms[:3])

        # Collect example doc ids for this topic
        doc_indices = [i for i, t in enumerate(topics) if t == tid]
        example_doc_ids = [str(docs[i]["id"]) for i in doc_indices[:10]]

        topics_dict[tid] = {
            "top_terms": top_terms,
            "summary": summary,
            "example_docs": example_doc_ids,
        }

    # Attach cluster_id to each doc
    for doc, topic_id in zip(docs, topics):
        doc["cluster_id"] = int(topic_id)

    return docs, topics_dict
