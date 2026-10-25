from typing import Any, Dict, List, Tuple

import numpy as np

# Generic legal vocabulary that shows up in nearly every opinion. c-TF-IDF
# already down-weights shared terms, but these still crowd out the words that
# actually distinguish one area of law from another.
LEGAL_STOP_WORDS = [
    "court", "courts", "case", "cases", "said", "act", "law", "laws", "state",
    "states", "united", "judgment", "judgments", "error", "opinion", "decree",
    "question", "questions", "made", "rule", "held", "think", "sec", "section",
    "mr", "justice", "chief", "appellant", "appellee", "plaintiff", "plaintiffs",
    "defendant", "defendants", "petitioner", "respondent", "shall", "upon",
    "would", "may", "must", "also", "u", "s", "v", "ct", "ed", "supra", "ibid",
    "id", "fed", "cl", "ante", "post", "art", "stat", "ann",
]


def build_topic_model(num_docs: int, random_state: int = 42):
    """
    Configure BERTopic. Cluster sizes scale with the corpus so the same code
    works for the bundled 100 opinions and for a few thousand.
    Imports are local so a cached startup never pays for loading UMAP/numba.
    """
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
    from umap import UMAP

    umap_model = UMAP(
        n_neighbors=10 if num_docs <= 200 else 15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=random_state,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=max(5, num_docs // 50),
        min_samples=2,
        metric="euclidean",
        prediction_data=True,
    )
    vectorizer_model = CountVectorizer(
        stop_words=list(ENGLISH_STOP_WORDS.union(LEGAL_STOP_WORDS)),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",  # drop citations and numbers
        ngram_range=(1, 2),
    )
    return BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        verbose=False,
    )


def run_topic_modeling(
    docs: List[Dict[str, Any]],
    embeddings: np.ndarray,
    random_state: int = 42,
) -> Tuple[List[int], Dict[int, Dict[str, Any]], float]:
    """
    Fit BERTopic on the documents using precomputed embeddings.
    Returns (topic_id per doc, topics dict, outlier fraction before reassignment).
    """
    texts = [d["text"] for d in docs]
    model = build_topic_model(len(docs), random_state=random_state)
    topics, _probs = model.fit_transform(texts, embeddings=embeddings)
    outlier_fraction = float(np.mean(np.asarray(topics) == -1))

    # Reassign outliers (topic -1) to the nearest real topic. If HDBSCAN found
    # no structure at all, everything stays in a single catch-all topic.
    if 0 < outlier_fraction < 1:
        topics = model.reduce_outliers(texts, topics, strategy="distributions")
        model.update_topics(texts, topics=topics, vectorizer_model=model.vectorizer_model)
    topics = [int(t) for t in topics]

    topics_dict: Dict[int, Dict[str, Any]] = {}
    for tid in sorted(set(topics)):
        # c-TF-IDF top terms from BERTopic
        topic_words = model.get_topic(tid) or []
        top_terms = [word for word, _score in topic_words[:10]]

        # Human-readable label from top 3 keywords
        summary = " / ".join(top_terms[:3]) if top_terms else f"topic {tid}"

        member_idx = [i for i, t in enumerate(topics) if t == tid]
        topics_dict[tid] = {
            "top_terms": top_terms,
            "summary": summary,
            "size": len(member_idx),
            "example_docs": [str(docs[i]["id"]) for i in member_idx[:10]],
            "representative_docs": representative_docs(docs, embeddings, member_idx),
        }

    return topics, topics_dict, outlier_fraction


def representative_docs(
    docs: List[Dict[str, Any]],
    embeddings: np.ndarray,
    member_idx: List[int],
    k: int = 3,
) -> List[str]:
    """The k members closest to the topic centroid in embedding space."""
    if not member_idx:
        return []
    member_emb = embeddings[member_idx]
    centroid = member_emb.mean(axis=0)
    order = np.argsort(-(member_emb @ centroid))[:k]
    return [str(docs[member_idx[i]]["id"]) for i in order]


def project_2d(embeddings: np.ndarray, random_state: int = 42) -> np.ndarray:
    """2-D UMAP layout of the documents for the explorer map."""
    from umap import UMAP

    n = len(embeddings)
    if n < 4:
        return np.zeros((n, 2))
    reducer = UMAP(
        n_neighbors=min(15, n - 1),
        n_components=2,
        min_dist=0.1,
        metric="cosine",
        random_state=random_state,
    )
    return reducer.fit_transform(embeddings)
