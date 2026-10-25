# What the disagreements mean

The explorer and `/analysis/comparison` flag a case as a **disagreement** when its BERTopic topic is not the most common topic in its Louvain community. On the bundled 100 cases that is 30% of the corpus; on the 1,000-case sample it is 55%. This note explains what those numbers actually measure, using the breakdown from `/analysis/disagreements` and a reading of the individual cases.

## The short version

- **The disagreement rate mostly measures granularity, not conflict.** BERTopic finds more, smaller groups than Louvain (9 topics vs 7 communities on 100 cases; 20 vs 9 on 1,000). Whenever one community holds two topics, every case in the smaller topic gets flagged, even if nothing about it is unusual.
- **Real conflicts are rare.** Only 8 cases in each corpus are placed differently by the two methods on their own.
- **Those cases are interesting.** They are mostly cases that genuinely span two areas of law, or cases neither method has a good home for.

| Kind | What it means | 100 cases | 1,000 cases |
|---|---|---|---|
| **Merged** | Most of the case's topic sits in this community, but shares it with a larger topic. The graph drew one group where BERTopic drew two. | 13 | 393 (71%) |
| **Split** | A group of cases from one topic sits in a community where another topic dominates, apart from the rest of its topic. | 9 | 153 |
| **Lone** | The only case of its topic in its community. | 8 | 8 |

## Background: why two methods disagree at all

The two methods answer different questions about the same embeddings:

- **BERTopic** asks *where a case sits in meaning-space*. It squeezes the 384-dimension embeddings down to 5 with UMAP, then HDBSCAN finds dense regions. Every case ends up in exactly one topic: cases outside any dense region are reassigned to the nearest one. c-TF-IDF then picks keywords that distinguish each topic.
- **Louvain** asks *who a case is connected to*. Each case is linked to its most similar cases (the top 5% of pairs plus its 3 nearest neighbours). Louvain then finds groups with many links inside and few between, which is called maximising *modularity*. At this scale modularity tends to produce a few large groups. That is a known property (the "resolution limit"), and it explains why there are fewer communities than topics.

The "majority topic" rule is simple, and it has a built-in bias: it can only see one topic per community. So two topics sharing a community always produce disagreements, even when both methods agree on everything except where to draw a boundary.

## The three kinds, with examples

### Merged: the graph is coarser here

On 100 cases, community 1 holds 8 of the 10 *bond / property / appeal* cases and 9 *flaglor / deed / title* cases. Those 8 cases are flagged only because the other topic is slightly larger.

On 1,000 cases this is the dominant pattern. One community holds *commission / rates / interstate* together with all 36 *union / labor* cases, all 29 *commerce / power / tax* cases, all 27 *religious* cases and 51 of 54 *city / water / company* cases. In effect, the graph sees "20th-century public and regulatory law" as one neighbourhood, while BERTopic splits it by subject. Neither is wrong; they are different zoom levels.

**Lesson:** this is why the agreement metrics differ so much on 1,000 cases. ARI (0.27) punishes a granularity mismatch heavily and NMI (0.51) less so, while the raw disagreement rate counts every merged case as a conflict.

### Split: part of a topic belongs somewhere else

On 100 cases, 4 of the 12 *patent* cases sit in community 2, which is mostly criminal and insurance cases. Reading them shows why:

- **106 U.S. 613** asks whether a suit over money owed under a patent assignment contract "arises under" the patent laws. It is a *jurisdiction* question that happens to involve a patent.
- **106 U.S. 672** asks whether someone who bought patent-infringement claims can sue in equity. That is a question about who may sue, not about inventions.

Two other "patent" cases, in community 4, aren't patent cases at all. **109 U.S. 618** is about a contract to build a wall for the United States, and **106 U.S. 399** is about a levee board's construction contract. BERTopic filed them badly. 109 U.S. 618 is actually more similar to its community's topic (0.55) than to the rest of the patent topic (0.51).

Split cases are measurably more mixed. On average only 63% of a split case's passages match its own topic, against 85% for cases where the methods agree (45% vs 69% on 1,000 cases).

### Lone: the real conflicts

- **A genuinely two-topic case.** 108 U.S. 436 concerns one railroad company buying another's franchises, and the states' approval of the sale. Its 44 passages split exactly between *railroad company* (18) and *bonds / coupons / city* (18). BERTopic chose railroad, while the graph puts it with the municipal bond cases: 6 of its 10 nearest cases are in that community, and it is closer to the bond topic (0.84 vs 0.80).
- **A legal question no topic captures.** 108 U.S. 158, 108 U.S. 561 and 108 U.S. 130 all decide the same question: when a suit can be *removed* from a state court to a federal court under the Act of 1875. BERTopic filed them under three different topics (*policy / suit / life*, *attorney / indictment*, *bond / property / appeal*) by their subject vocabulary. The graph put two of them in the same community. Neither method has a "removal jurisdiction" group at 100 cases, but the graph comes closer.
- **Orphans.** 109 U.S. 146, about an interpreter's salary and congressional appropriations, was filed under *bonds / coupons / city*, but it is barely similar to anything (0.52 to its topic, 0.47 to its community's topic). On 1,000 cases, 268 U.S. 346, about citizenship of Chinese women married to Americans, is similar to nothing (0.40 / 0.31). These cases have no natural group in the sample, and each method forces them somewhere different.

On 1,000 cases, only 36% of a lone case's passages match its own topic, the lowest of any kind.

## Patterns from reading the cases

These come from reading about 20 flagged cases, so treat them as observations, not measurements.

1. **Subject vs legal question.** An opinion has two axes: what the dispute is *about* (a patent, a bond, a ship) and what the Court actually *decides*, which in the 1880s was very often procedure (jurisdiction, removal, appeals, finality of decrees). BERTopic's keywords lean towards subject vocabulary, and many flagged cases are procedural cases filed by their subject.
2. **Mixed cases.** Some opinions really do cover two areas, and the passage bar in the explorer shows it directly.
3. **Orphans.** In a small sample some cases have no natural group, so where they land is close to arbitrary.
4. **Very short opinions.** Some are a single sentence, such as 120 U.S. 51, "affirmed on the authority of United States v. Symonds", or have 1–3 passages, so their embeddings carry little signal.

## Which method is right?

There is no ground truth, so each flagged case gets two signals. Each signal is biased towards one method:

- **Nearest-neighbour vote** (how many of its 10 most similar cases share its topic vs its community) favours Louvain, because the graph is built from those same neighbours.
- **Similarity to the topic's centre** (with the case itself left out) favours BERTopic, because topics are regions of the same embedding space.

So the explorer only gives a verdict when both signals agree:

| | 100 cases | 1,000 cases |
|---|---|---|
| Both favour BERTopic's topic | 3 | 85 |
| Both favour the graph's community | 5 | 74 |
| Signals point different ways | 22 | 395 |

Neither method systematically wins. Most flagged cases sit between two areas, which is what you'd expect at the boundary of two groups.

## A hypothesis that didn't hold: era

The giant 1,000-case community looked like it might be grouping by era (writing style) rather than subject. The volume number is a rough proxy for date, so I measured how much of its variance each grouping explains:

| Grouping | Share of volume variance explained |
|---|---|
| Louvain communities | 0.21 |
| BERTopic topics | 0.39 |

So the opposite is true: **topics are more tied to era than communities are**. Keyword-based topics pick up period vocabulary. For example, "sct" (the modern *S. Ct.* citation format) appears in the keywords of three modern-era topics. That word should probably be added to the stop-word list.

## Caveats

- **100 cases is small.** Topics only partly survive a change of random seed (ARI 0.37 between runs), so an individual case's flag may change on another run. The 1,000-case numbers are more trustworthy.
- **The kinds depend on the majority rule.** A different definition of disagreement would sort cases differently.
- **Passage matching is approximate.** Each passage is matched against topic centres built from whole-opinion vectors, so short or generic passages can land on broad topics such as *jurisdiction / writ*.
- **The 2-D map distorts distances.** It squeezes 384 dimensions into 2, so a case's nearest neighbours can appear far apart. 108 U.S. 436 in the explorer is a good example.
- **OCR noise is in the text**, e.g. "dollms" and "Cbnnecticut". Odd keywords like "flaglor" probably come from party names or scanning errors.

## Reproduce

- Explorer: `http://localhost:8000/explorer?mode=agreement`. Click a lone case for its evidence, or a split or merged row to highlight it.
- API: `GET /analysis/disagreements` returns every flagged case with its kind and evidence, and `GET /documents/{id}` includes `passage_topics`.
- The 1,000-case figures come from `DOCS_PATH=data/sample_1000.jsonl`. The classification is `explain_disagreements` in `evaluation.py`.
