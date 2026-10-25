"""
Optional Claude-generated topic names. BERTopic's keyword labels
("patent / invention / saw") are accurate but hard to scan; one request turns
every topic into a short name a lawyer would recognise. Results are cached on
disk by topic content, so restarts and unchanged topics cost nothing.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You name clusters of US Supreme Court opinions for a case-browsing tool. "
    "For each cluster you get its distinguishing keywords and excerpts from its most "
    "representative cases. Reply with a label of 3-8 words naming the area of law or "
    "legal issue the cases share, the way a law librarian would title a shelf "
    "(e.g. 'Patent infringement on industrial machinery', 'Admiralty cargo damage claims'). "
    "Base the label on the excerpts when keywords are noisy. Don't use the words 'cases' or 'cluster'."
)

LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        "labels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "topic_id": {"type": "integer"},
                    "label": {"type": "string"},
                },
                "required": ["topic_id", "label"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["labels"],
    "additionalProperties": False,
}

EXCERPT_CHARS = 700


def _excerpt(doc: Dict[str, Any]) -> str:
    text = (doc.get("syllabus") or doc.get("text") or "").strip()
    return " ".join(text[:EXCERPT_CHARS].split())


def build_prompt(topics: Dict[int, Dict[str, Any]], docs_by_id: Dict[str, Dict[str, Any]]) -> str:
    parts = []
    for tid, info in sorted(topics.items()):
        excerpts = "\n".join(
            f"  - {_excerpt(docs_by_id[did])}" for did in info.get("representative_docs", []) if did in docs_by_id
        )
        parts.append(
            f"<cluster id=\"{tid}\">\nKeywords: {', '.join(info['top_terms'])}\nExcerpts:\n{excerpts}\n</cluster>"
        )
    return "Label each of these clusters.\n\n" + "\n\n".join(parts)


def _cache_path(cache_dir: Path, model: str, prompt: str) -> Path:
    key = hashlib.sha1(f"{model}|{SYSTEM_PROMPT}|{prompt}".encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"labels-{key}.json"


def generate_labels(
    topics: Dict[int, Dict[str, Any]],
    docs_by_id: Dict[str, Dict[str, Any]],
    model: str,
    cache_dir: Path,
    client=None,
) -> Dict[int, str]:
    """
    Return {topic_id: label}. Any failure (no credentials, network, refusal,
    malformed reply) is logged and yields an empty dict, so callers fall back
    to the keyword summary instead of failing startup.
    """
    if not topics:
        return {}
    prompt = build_prompt(topics, docs_by_id)
    path = _cache_path(cache_dir, model, prompt)
    if path.exists():
        return {int(k): v for k, v in json.loads(path.read_text()).items()}

    try:
        import anthropic

        client = client or anthropic.Anthropic()
        response = client.beta.messages.create(
            model=model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": LABEL_SCHEMA}},
            # If a safety classifier declines, let the API retry on its recommended fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
    except Exception as exc:  # noqa: BLE001 - labels are a nice-to-have
        logger.warning("LLM topic labelling failed, using keyword labels: %s", exc)
        return {}

    if response.stop_reason != "end_turn":
        logger.warning("LLM topic labelling stopped early (%s), using keyword labels", response.stop_reason)
        return {}

    try:
        text = next(b.text for b in response.content if b.type == "text")
        items: List[Dict[str, Any]] = json.loads(text)["labels"]
    except (StopIteration, ValueError, KeyError) as exc:
        logger.warning("Could not parse LLM topic labels: %s", exc)
        return {}

    labels = {int(item["topic_id"]): item["label"].strip() for item in items if int(item["topic_id"]) in topics}
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({str(k): v for k, v in labels.items()}, indent=2))
    return labels
