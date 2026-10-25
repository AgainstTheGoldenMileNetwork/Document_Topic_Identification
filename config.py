import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_float(name: str) -> float | None:
    value = os.getenv(name)
    return float(value) if value not in (None, "") else None


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """
    Runtime configuration, read from environment variables so the API, the
    evaluation script and the tests can all build the same pipeline.
    """

    docs_path: Path = field(default_factory=lambda: Path(os.getenv("DOCS_PATH", "data/documents.jsonl")))
    cache_dir: Path = field(default_factory=lambda: Path(os.getenv("CACHE_DIR", ".cache")))

    # Embeddings: "chunked" embeds the whole opinion in overlapping windows and
    # averages them; "truncated" is the old behaviour (first ~256 tokens only).
    embed_model: str = field(default_factory=lambda: os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2"))
    embed_strategy: str = field(default_factory=lambda: os.getenv("EMBED_STRATEGY", "chunked"))
    chunk_overlap: int = 32
    # Remove citations and case names before embedding, so similarity to a
    # cited case cannot come from the citation string itself.
    strip_citations: bool = field(default_factory=lambda: _env_bool("STRIP_CITATIONS"))

    # Subject / legal-question tags per case (written by Claude from the
    # syllabi). Defaults to data/legal_tags_<corpus>.jsonl; optional.
    tags_path: Path | None = field(default_factory=lambda: Path(os.getenv("TAGS_PATH")) if os.getenv("TAGS_PATH") else None)

    # Similarity graph: an explicit SIM_THRESHOLD wins; otherwise keep the
    # top (100 - SIM_PERCENTILE)% of document pairs as edges. Each document is
    # also linked to its SIM_KNN nearest neighbours so none are left isolated.
    # Defaults chosen from the threshold sweep in reports/.
    sim_threshold: float | None = field(default_factory=lambda: _env_float("SIM_THRESHOLD"))
    sim_percentile: float = field(default_factory=lambda: _env_float("SIM_PERCENTILE") or 95.0)
    sim_knn: int = field(default_factory=lambda: int(os.getenv("SIM_KNN", "3")))

    random_state: int = 42

    # LLM topic labels are opt-in because they call a paid API.
    llm_labels: bool = field(default_factory=lambda: _env_bool("ENABLE_LLM_LABELS"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_LABEL_MODEL", "claude-opus-5-5"))


    def resolved_tags_path(self) -> Path:
        return self.tags_path or self.docs_path.with_name(f"legal_tags_{self.docs_path.stem}.jsonl")


def get_settings() -> Settings:
    return Settings()
