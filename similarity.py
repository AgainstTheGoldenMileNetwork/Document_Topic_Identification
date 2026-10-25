from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
from networkx.algorithms.community import louvain_communities, modularity


class SimilarityGraph:
    """
    Builds a similarity graph from document embeddings and provides
    query methods for similar docs, graph stats, and community detection.

    The edge threshold is either given explicitly or picked as a percentile of
    all pairwise similarities. A percentile adapts to the embedding model:
    chunk-averaged vectors sit much closer together than truncated ones, so a
    fixed cut-off that suited one would connect everything under the other.

    A global cut-off alone strands documents whose similarities are uniformly
    low (often short procedural opinions); each becomes a one-document
    community. Every document is therefore also linked to its `knn` most
    similar neighbours, so the graph has no isolated nodes.
    """

    def __init__(
        self,
        doc_ids: List[str],
        embeddings: np.ndarray,
        threshold: float | None = None,
        percentile: float = 95.0,
        knn: int = 3,
        seed: int = 42,
    ):
        self.doc_ids = doc_ids
        self.id_to_idx = {did: i for i, did in enumerate(doc_ids)}
        normed = embeddings / np.clip(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12, None)
        self.sim_matrix = normed @ normed.T
        self._iu = np.triu_indices(len(doc_ids), k=1)
        self._pair_sims = self.sim_matrix[self._iu]
        self.percentile = None if threshold is not None else percentile
        if threshold is None:
            threshold = float(np.percentile(self._pair_sims, percentile)) if len(self._pair_sims) else 1.0
        self.threshold = threshold
        self.knn = knn
        self.seed = seed
        self.graph = self._build_graph()
        self.communities = self._detect_communities()

    def _build_graph(self) -> nx.Graph:
        G = nx.Graph()
        G.add_nodes_from(self.doc_ids)
        keep = self._pair_sims >= self.threshold
        rows, cols = self._iu[0][keep], self._iu[1][keep]
        G.add_weighted_edges_from(
            (self.doc_ids[i], self.doc_ids[j], float(w))
            for i, j, w in zip(rows, cols, self._pair_sims[keep])
        )
        n = len(self.doc_ids)
        if self.knn > 0 and n > 1:
            sims = self.sim_matrix.copy()
            np.fill_diagonal(sims, -np.inf)
            k = min(self.knn, n - 1)
            neighbours = np.argpartition(-sims, k - 1, axis=1)[:, :k]
            G.add_weighted_edges_from(
                (self.doc_ids[i], self.doc_ids[j], float(self.sim_matrix[i, j]))
                for i in range(n)
                for j in neighbours[i]
            )
        return G

    def _detect_communities(self) -> List[List[str]]:
        if self.graph.number_of_edges() == 0:
            # Every node is its own community if no edges
            return [[did] for did in self.doc_ids]
        communities = louvain_communities(self.graph, weight="weight", seed=self.seed)
        # Largest first, so community ids are stable and meaningful.
        return sorted((sorted(c) for c in communities), key=lambda c: (-len(c), c[0]))

    def community_labels(self) -> List[int]:
        """Community id for each document, aligned with doc_ids."""
        labels = [0] * len(self.doc_ids)
        for cid, members in enumerate(self.communities):
            for did in members:
                labels[self.id_to_idx[did]] = cid
        return labels

    def get_similar(self, doc_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Return top_k most similar documents to doc_id (excluding itself)."""
        if doc_id not in self.id_to_idx:
            return []
        idx = self.id_to_idx[doc_id]
        sims = self.sim_matrix[idx].copy()
        sims[idx] = -np.inf
        ranked = np.argsort(-sims)[:top_k]
        return [(self.doc_ids[i], float(sims[i])) for i in ranked]

    def get_sim_stats(self) -> Dict[str, Any]:
        """Stats on the upper triangle of the similarity matrix (all unique pairs)."""
        arr = self._pair_sims
        return {
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "threshold_used": self.threshold,
            "threshold_percentile": self.percentile,
            "knn": self.knn,
            "edges_at_threshold": int((arr >= self.threshold).sum()),
            "edges_total": self.graph.number_of_edges(),
        }

    def get_info(self) -> Dict[str, Any]:
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "communities": len(self.communities),
            "modularity": self.modularity(),
            "threshold": self.threshold,
            "knn": self.knn,
        }

    def modularity(self) -> float:
        if self.graph.number_of_edges() == 0:
            return 0.0
        return float(modularity(self.graph, self.communities, weight="weight"))

    def get_edges(self, min_weight: float | None = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Strongest edges first. Without min_weight this includes nearest-neighbour edges below the threshold."""
        min_weight = -np.inf if min_weight is None else min_weight
        edges = []
        for u, v, data in sorted(
            self.graph.edges(data=True), key=lambda e: e[2]["weight"], reverse=True
        ):
            if data["weight"] < min_weight:
                break
            edges.append({"source": u, "target": v, "weight": data["weight"]})
            if len(edges) >= limit:
                break
        return edges
