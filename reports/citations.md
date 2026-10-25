# Facts or law? Checking the embeddings against the citation graph

Everything else in `reports/` compares one unsupervised method with another. This note checks them against something external: the citations in the opinions themselves. Numbers are from the 1,000-case sample (`data/sample_1000.jsonl`) with citations stripped before embedding (`STRIP_CITATIONS=1`); the raw output is `citations_sample_1000.json` and the explorer's **Citations** view shows the same tables for whatever corpus is loaded.

## The idea

When an opinion cites an earlier case, or two opinions cite the same precedent, a lawyer has judged them related, and that judgement is about the **legal question**, not the facts: a railroad case cites a shipping case because both turn on federal jurisdiction. So the citation graph is a ground truth for "related cases" that does not come from the model.

Separately, a Sonnet subagent read each case's syllabus and tagged two things: the factual **subject** (land, crime, shipping…) and the **legal question** the Court decides (jurisdiction, due process, statutory interpretation…), each as one of 13 categories (`data/legal_tags_*.jsonl`). The two often differ, and the point of the exercise is to see which one the embeddings follow.

Citations were stripped from the text before embedding so that similarity cannot come from shared citation strings. It barely mattered: similarities with and without them correlate at 0.99, and every number below moves by a hair.

## 1. The embeddings recover the citation graph reasonably well

622 of the 1,000 cases have at least one link: 213 pairs cite each other directly and 1,424 pairs cite a shared precedent.

| Pair | Mean similarity | Pairs |
|---|---|---|
| No citation link | 0.47 | 497,981 |
| Cite one shared precedent | 0.59 | 1,127 |
| Cite two or more | 0.66 | 179 |
| One cites the other | 0.70 | 213 |

If A cites B, B's median rank among A's 999 neighbours is **9**; it is in the top 10 for 52% of pairs and the top 50 for 70% (random: 1% and 5%). Similarity alone separates citing pairs from unlinked pairs with AUC 0.90. 17% of a case's five nearest neighbours are citation-linked, against 0.5% by chance.

## 2. The hard groupings lose a third of the links

| Pairs that are… | Same BERTopic topic | Same Louvain community |
|---|---|---|
| Random | 8% | 19% |
| Citation-linked | 41% | 56% |
| One cites the other | 62% | 69% |

So 30–40% of pairs that cite each other directly are filed in different groups by either method, even though the similarity ranking usually knows they are related (rank 9). A single label per case drops information that the ranking keeps.

## 3. Similarity follows the facts; citations follow the law

For every pair of cases the tags say whether they share a subject and whether they share a legal question.

| Pair shares… | Mean similarity | Pairs |
|---|---|---|
| Neither | 0.456 | 392,413 |
| Legal question only | 0.505 | 59,611 |
| Subject only | **0.542** | 34,324 |
| Both | 0.571 | 13,152 |

A shared subject moves similarity by +0.086, a shared legal question by +0.049. A case's five nearest neighbours share its subject 51% of the time (5.4× chance) and its legal question 44% (3.0× chance, and the question categories are broader, so that is the easier target). Similarity predicts "same subject" with AUC 0.71 and "same question" with AUC 0.63.

Citations lean the other way:

| Pairs that are… | Same subject | Same legal question |
|---|---|---|
| Random | 10% | 15% |
| Cite a shared precedent | 33% | 43% |
| One cites the other | 54% | 60% |
| Same BERTopic topic | **39%** | 29% |
| Same Louvain community | 24% | 23% |

BERTopic's topics are organised by subject (39% vs 29%), the citation graph by legal question (60% vs 54%, and 43% vs 33% for co-citation).

## 4. The decisive test: citing pairs whose facts differ

Among pairs that cite each other, how well does the embedding find the partner?

| The citing pair has… | Partner's median rank (of 999) | In top 10 | Pairs |
|---|---|---|---|
| Same facts, same question | 4 | 64% | 148 |
| Same facts, **different** question | 3 | 63% | 82 |
| **Different** facts, same question | 17 | 44% | 108 |
| Neither | 54 | 31% | 88 |

Sharing facts is what makes a citation partner findable; sharing the legal question barely helps. The worst-ranked examples are what a lawyer would call closely related cases: a tax case and a bridge case that both turn on whether a state-court judgment rests on a federal question (ranked 705th of 999 before stripping); a union-licensing case and a telephone-rate case that both turn on state power against federal law (658th).

## 5. Which legal questions the model can and cannot see

Mean similarity among cases with the same legal question, against their similarity to everything else:

| Legal question | Cases | Subjects spanned | Within | Other | Gap |
|---|---|---|---|---|---|
| Statutory interpretation | 182 | 12 | 0.448 | 0.451 | **−0.003** |
| Constitution: rights | 157 | 11 | 0.494 | 0.459 | +0.036 |
| Constitution: federal structure | 77 | 11 | 0.506 | 0.462 | +0.045 |
| Equity & remedies | 48 | 8 | 0.552 | 0.489 | +0.063 |
| Jurisdiction & procedure | 259 | 13 | 0.551 | 0.483 | +0.068 |
| Property law | 75 | 5 | 0.566 | 0.475 | +0.091 |
| Criminal law | 22 | 2 | 0.550 | 0.457 | +0.093 |
| Patent law | 16 | 1 | 0.519 | 0.358 | +0.161 |
| Admiralty | 23 | 2 | 0.631 | 0.431 | +0.200 |

The pattern is clean: the gap is large exactly where the legal question coincides with a factual domain (admiralty, patents, crime) and vanishes where it cuts across domains. **182 statutory-interpretation cases, spanning 12 subjects, are no more similar to each other than to the rest of the corpus.** The law they apply is invisible to the model.

## What this means

- An off-the-shelf sentence embedding is a good "what is this dispute about" detector and a poor "what law does this apply" detector. For case law, where relatedness means the latter, that is the wrong axis much of the time.
- The topic model inherits this: a BERTopic "topic" here is a factual domain. That is why a legal question like removal to federal court scatters across three topics, and why the [disagreements](disagreements.md) between methods cluster on cases that mix facts and law.
- The gap is measurable with nothing but the citations already in the text, so it is a cheap yardstick for trying alternatives: a legal-domain embedding model, or embedding a short Claude-written statement of the legal question instead of the opinion.

## Caveats

- 13 categories each is coarse. "Same legal question" at that granularity is a loose match, which flatters the embeddings on the legal axis rather than penalising them.
- Tags come from Sonnet reading truncated syllabi: 75% high confidence, 4% low. Restricting to high-confidence tags leaves the table in section 3 unchanged to two decimals.
- Citation extraction is regex-based (`citations.py`) and catches U.S. Reports and the early nominative reporters; cases cited by name alone are missed. Co-citation links are weighted down for precedents that everyone cites.
- A random 1,000-case sample is sparse for direct links (213 pairs). The bundled 100 cases have only 4, which is why the explorer tells you to load the sample.

## Reproduce

```bash
python download_docs.py --num-docs 1000 --sample --out data/sample_1000.jsonl   # tags for this sample are committed
DOCS_PATH=data/sample_1000.jsonl STRIP_CITATIONS=1 uvicorn app:app
# open http://localhost:8000/explorer?mode=citations, or GET /analysis/citations
```
