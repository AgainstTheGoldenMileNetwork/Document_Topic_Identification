import json
import random
import re
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_loader import Embedder  # noqa: E402

DIM = 64

THEMES = {
    "patent": "patent invention claim inventor machine device novelty infringement saw mechanism",
    "maritime": "vessel cargo ship master freight voyage collision harbor admiralty seamen",
    "tax": "tax assessment county bonds township levy revenue municipal coupons treasurer",
    "criminal": "indictment jury conviction guilty sentence prosecution witness trial offence crime",
}
COMMON = "the court held that this appeal was decided under the statute and the judgment below".split()


class FakeTokenizer:
    """Whitespace tokenizer that reports character offsets like a HF fast tokenizer."""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True, verbose=False):
        return {"offset_mapping": [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]}


class FakeModel:
    max_seq_length = 40  # -> 24-token windows
    tokenizer = FakeTokenizer()

    def encode(self, texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True):
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in re.findall(r"[a-z]+", text.lower()):
                out[i, zlib.crc32(word.encode()) % DIM] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.clip(norms, 1e-12, None)


class FakeEmbedder(Embedder):
    def __init__(self, strategy: str = "chunked"):
        super().__init__("fake-model", strategy, overlap=4)
        self._model = FakeModel()


def make_corpus(per_theme: int = 12, seed: int = 0):
    rng = random.Random(seed)
    docs = []
    for theme, vocab in THEMES.items():
        words = vocab.split()
        for i in range(per_theme):
            body = " ".join(rng.choice(words) if rng.random() < 0.6 else rng.choice(COMMON) for _ in range(90))
            docs.append({
                "id": f"{theme}-{i}",
                "title": f"{theme.title()} case {i}",
                "court": "SCOTUS",
                "date": None,
                "syllabus": f"A {theme} dispute about {' and '.join(rng.sample(words, 3))}.",
                "text": body,
                "theme": theme,
            })
    return docs


@pytest.fixture
def corpus():
    return make_corpus()


@pytest.fixture
def corpus_file(tmp_path, corpus):
    path = tmp_path / "docs.jsonl"
    path.write_text("\n".join(json.dumps(d) for d in corpus) + "\n")
    return path
