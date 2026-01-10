import json
from pathlib import Path

from datasets import load_dataset
from keybert import KeyBERT

OUT_PATH = Path("data/documents.jsonl")


def download_casesumm(num_docs: int = 100) -> None:
    """
    Download `num_docs` cases from the ChicagoHAI/CaseSumm dataset on Hugging Face
    and write them to data/documents.jsonl in the format:

      {id, title, court, date, text}

    We use:
      - opinion  => full text of the case
      - citation => id/title
    """
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading ChicagoHAI/CaseSumm from Hugging Face...")
    ds = load_dataset("ChicagoHAI/CaseSumm", split="train")
    # Features are roughly: citation (str), syllabus (str), opinion (str). :contentReference[oaicite:1]{index=1}

    selected = []

    for row in ds:
        opinion_text = (row.get("opinion") or "").strip()
        if not opinion_text:
            continue  # skip rows with empty opinion

        citation = (row.get("citation") or "").strip()
        if not citation:
            citation = f"case-{len(selected)}"

        doc = {
            "id": citation,            # unique id
            "title": citation,         # you can later parse citation if you care
            "court": "SCOTUS",         # CaseSumm is US Supreme Court
            "date": None,              # dataset doesn't expose full date; ok to leave None
            "text": opinion_text,
        }
        selected.append(doc)

        if len(selected) >= num_docs:
            break

    if not selected:
        raise RuntimeError("No documents were loaded from CaseSumm (check network or dataset).")

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for d in selected:
            f.write(json.dumps(d) + "\n")

    print(f"Wrote {len(selected)} documents to {OUT_PATH}")


if __name__ == "__main__":
    download_casesumm(100)