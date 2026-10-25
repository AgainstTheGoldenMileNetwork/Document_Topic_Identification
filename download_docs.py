import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset

OUT_PATH = Path("data/documents.jsonl")

CITATION_RE = re.compile(r"^(\d+)\.US\.(\d+)$")
# Recent cases without a final page number: volume, term year, docket number.
DOCKET_CITATION_RE = re.compile(r"^(\d+)\.US\.(\d{4})_([\w-]+)$")


def parse_citation(citation: str) -> dict:
    """
    '108.US.30'           -> title '108 U.S. 30'
    '591.US.2019_19-465'  -> title '591 U.S. ___ (No. 19-465)', date '2019'
    """
    match = CITATION_RE.match(citation)
    if match:
        volume, page = int(match.group(1)), int(match.group(2))
        return {"title": f"{volume} U.S. {page}", "volume": volume, "page": page, "date": None}
    match = DOCKET_CITATION_RE.match(citation)
    if match:
        volume, year, docket = int(match.group(1)), match.group(2), match.group(3)
        return {"title": f"{volume} U.S. ___ (No. {docket})", "volume": volume, "page": None, "date": year}
    return {"title": citation, "volume": None, "page": None, "date": None}


def download_casesumm(num_docs: int = 100, sample: bool = False, seed: int = 42, out_path: Path = OUT_PATH) -> None:
    """
    Download `num_docs` cases from the ChicagoHAI/CaseSumm dataset on Hugging Face
    and write them to `out_path`, one JSON object per line:

      {id, title, volume, page, court, date, syllabus, text}

    The dataset has three columns: citation, syllabus (the official summary
    written by the Reporter of Decisions) and opinion (full text). It has no
    case names or decision dates, so `title` is the formatted citation and
    `date` is only set (to the term year) when the citation includes one.

    By default the first `num_docs` rows are taken, which reproduces the bundled
    data file. With `sample=True` rows are drawn at random across the whole
    dataset (~27k cases), which gives a far more varied corpus.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading ChicagoHAI/CaseSumm from Hugging Face...")
    ds = load_dataset("ChicagoHAI/CaseSumm", split="train")
    if sample:
        ds = ds.shuffle(seed=seed)

    selected = []
    seen = set()

    for row in ds:
        opinion_text = (row.get("opinion") or "").strip()
        if not opinion_text:
            continue  # skip rows with empty opinion

        citation = (row.get("citation") or "").strip() or f"case-{len(selected)}"
        if citation in seen:
            continue
        seen.add(citation)

        selected.append({
            "id": citation,
            **parse_citation(citation),
            "court": "SCOTUS",
            "syllabus": (row.get("syllabus") or "").strip() or None,
            "text": opinion_text,
        })

        if len(selected) >= num_docs:
            break

    if not selected:
        raise RuntimeError("No documents were loaded from CaseSumm (check network or dataset).")

    with out_path.open("w", encoding="utf-8") as f:
        for d in selected:
            f.write(json.dumps(d) + "\n")

    print(f"Wrote {len(selected)} documents to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Supreme Court opinions from CaseSumm.")
    parser.add_argument("--num-docs", type=int, default=100)
    parser.add_argument("--sample", action="store_true", help="draw a random sample instead of the first rows")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()
    download_casesumm(args.num_docs, sample=args.sample, seed=args.seed, out_path=args.out)
