# Legal Topic Sorter

A FastAPI application that clusters US Supreme Court opinions into topics and exposes a similarity graph for finding related cases. Built as an ML exploration project comparing two different approaches to document grouping.

## What it does

1. **Downloads** ~100 Supreme Court opinions from the [CaseSumm](https://huggingface.co/datasets/ChicagoHAI/CaseSumm) dataset
2. **Embeds** them using `all-MiniLM-L6-v2` (384-dimensional sentence embeddings)
3. **Clusters** them into topics using BERTopic (UMAP + HDBSCAN + c-TF-IDF)
4. **Builds a similarity graph** using pairwise cosine similarity + Louvain community detection

## Two approaches to grouping documents

### BERTopic (topic modeling)

Answers: *"What distinct topics exist in this corpus?"*

- Embeddings are compressed to 5 dimensions via UMAP (non-linear dimensionality reduction)
- HDBSCAN finds dense regions in that reduced space
- c-TF-IDF extracts distinguishing keywords per topic
- Produces fine-grained topics like "patent / invention / saw" or "cargo / vessel / master"

### Similarity graph + Louvain (community detection)

Answers: *"Which documents are most connected to each other?"*

- Pairwise cosine similarity is computed in the full 384D embedding space (no reduction)
- Edges are created between documents above a similarity threshold
- Louvain algorithm finds tightly-knit communities by optimizing modularity
- Produces broader groupings — tends to merge topics that BERTopic keeps separate

The two methods often agree on clearly distinct topics (e.g., patent law cases) but diverge where legal language overlaps across areas (e.g., estate, maritime, and water rights cases may form one community but three separate BERTopic topics).

## API endpoints

### Topics (BERTopic)

| Endpoint | Description |
|---|---|
| `GET /clusters` | List all topics with keywords and example docs |
| `GET /clusters/{id}` | Topic detail with all member documents |
| `GET /documents/{id}` | Single document with its topic assignment |

### Similarity graph

| Endpoint | Description |
|---|---|
| `GET /documents/{id}/similar?top_k=10` | Most similar documents by cosine similarity |
| `GET /graph/info` | Graph stats: nodes, edges, density, community count |
| `GET /graph/stats` | Similarity distribution (min, max, percentiles) |
| `GET /graph/communities` | All Louvain communities |
| `GET /graph/communities/{id}` | Single community's members |
| `GET /graph/edges?min_weight=0.42&limit=100` | Strongest similarity edges |

### Other

| Endpoint | Description |
|---|---|
| `GET /health` | Status check |
| `GET /docs` | Interactive Swagger UI |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Download the dataset
python download_docs.py

# Run the server
uvicorn app:app --reload
```

## Project structure

```
app.py              — FastAPI routes and startup
models.py           — Pydantic response models
data_loader.py      — Document loading and embedding computation
topic_modeling.py   — BERTopic configuration and topic extraction
similarity.py       — Cosine similarity graph and Louvain communities
download_docs.py    — Dataset downloader (CaseSumm from Hugging Face)
```

## Key dependencies

- **BERTopic** — topic modeling (UMAP + HDBSCAN + c-TF-IDF)
- **sentence-transformers** — document embeddings (all-MiniLM-L6-v2)
- **networkx** — similarity graph and Louvain community detection
- **FastAPI** — REST API
