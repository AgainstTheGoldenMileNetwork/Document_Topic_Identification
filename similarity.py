from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
from networkx.algorithms.community import louvain_communities
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityGraph:
    """
    Builds a similarity graph from document embeddings and provides
    query methods for similar docs, graph stats, and community detection.
    """

    def __init__(
        self,
        doc_ids: List[str],
        embeddings: np.ndarray,
        threshold: float = 0.42,
    ):
        self.doc_ids = doc_ids
        self.id_to_idx = {did: i for i, did in enumerate(doc_ids)}
        self.sim_matrix = cosine_similarity(embeddings)
        self.threshold = threshold
        self.graph = self._build_graph()
        self.communities = self._detect_communities()

    def _build_graph(self) -> nx.Graph:
        G = nx.Graph()
        for doc_id in self.doc_ids:
            G.add_node(doc_id)
        n = len(self.doc_ids)
        for i in range(n):
            for j in range(i + 1, n):
                weight = float(self.sim_matrix[i][j])
                if weight >= self.threshold:
                    G.add_edge(self.doc_ids[i], self.doc_ids[j], weight=weight)
        return G

    def _detect_communities(self) -> List[List[str]]:
        if self.graph.number_of_edges() == 0:
            # Every node is its own community if no edges
            return [[did] for did in self.doc_ids]
        communities = louvain_communities(self.graph, weight="weight", seed=42)
        return [sorted(c) for c in communities]

    def get_similar(self, doc_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Return top_k most similar documents to doc_id (excluding itself)."""
        if doc_id not in self.id_to_idx:
            return []
        idx = self.id_to_idx[doc_id]
        sims = self.sim_matrix[idx]
        # Get indices sorted by descending similarity, skip self
        ranked = np.argsort(sims)[::-1]
        results = []
        for i in ranked:
            if self.doc_ids[i] == doc_id:
                continue
            results.append((self.doc_ids[i], float(sims[i])))
            if len(results) >= top_k:
                break
        return results

    def get_sim_stats(self) -> Dict[str, Any]:
        """Stats on the upper triangle of the similarity matrix (all unique pairs)."""
        n = len(self.doc_ids)
        upper = [
            float(self.sim_matrix[i][j])
            for i in range(n)
            for j in range(i + 1, n)
        ]
        arr = np.array(upper)
        return {
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "threshold_used": self.threshold,
            "edges_at_threshold": self.graph.number_of_edges(),
        }

    def get_info(self) -> Dict[str, Any]:
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "communities": len(self.communities),
        }

    def get_edges(self, min_weight: float = 0.42, limit: int = 100) -> List[Dict[str, Any]]:
        edges = []
        for u, v, data in sorted(
            self.graph.edges(data=True), key=lambda e: e[2]["weight"], reverse=True
        ):
            if data["weight"] < min_weight:
                continue
            edges.append({"source": u, "target": v, "weight": data["weight"]})
            if len(edges) >= limit:
                break
        return edges
