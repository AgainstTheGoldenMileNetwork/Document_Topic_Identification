import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

DATA_PATH = Path("data/documents.jsonl")

_model: SentenceTransformer | None = None


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
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = _model.encode(texts, batch_size=16, show_progress_bar=True)
    return np.array(embeddings)
