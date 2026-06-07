"""Confidence-aware ranking and graph context expansion."""

from __future__ import annotations

from collections import defaultdict

import networkx as nx

from graph_rag.config import RAGConfig
from graph_rag.models import DocumentChunk, SearchResult


class ConfidenceRanker:
    """Combine semantic similarity, RRF evidence and graph centrality."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.pagerank: dict[str, float] = {}

    def fit_graph(self, graph: nx.Graph | None) -> None:
        """Pre-compute graph centrality."""
        if not graph or graph.number_of_nodes() == 0:
            self.pagerank = {}
            return
        try:
            self.pagerank = nx.pagerank(graph, alpha=0.85, max_iter=200)
        except Exception:
            self.pagerank = {node: 1.0 / graph.number_of_nodes() for node in graph.nodes}

    def rerank(self, results: list[SearchResult], entities: list[dict], graph: nx.Graph | None) -> list[SearchResult]:
        """Apply centrality and confidence calibration."""
        for result in results:
            linked_entities = self._entities_for_chunk(result.chunk, entities)
            centrality = max((self.pagerank.get(entity["id"], 0.0) for entity in linked_entities), default=0.0)
            result.centrality = centrality
            semantic = max(result.dense_score, 0.0)
            lexical = min(1.0, result.bm25_score / 10.0) if result.bm25_score else 0.0
            result.score = (
                result.rrf_score
                + self.config.semantic_weight * semantic
                + self.config.centrality_weight * centrality
                + self.config.lexical_weight * lexical
            )
        results.sort(key=lambda item: item.score, reverse=True)
        if not results:
            return results
        top = results[0].score
        second = results[1].score if len(results) > 1 else 0.0
        for result in results:
            gap_factor = min(1.0, max(0.0, top - second) * 2.0)
            evidence = min(1.0, result.score / (top + 1e-9))
            result.confidence = round(0.75 * evidence + 0.25 * gap_factor, 4)
        return results

    def label(self, results: list[SearchResult]) -> str:
        """Convert calibrated confidence to an API label."""
        if not results:
            return "NONE"
        score = results[0].confidence
        if score >= 0.72:
            return "HIGH"
        if score >= 0.45:
            return "MEDIUM"
        return "LOW"

    def graph_augment_context(self, results: list[SearchResult], entities: list[dict], graph: nx.Graph | None, all_chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """Add graph-neighbor chunks to the direct retrieval context."""
        direct = [result.chunk for result in results]
        if graph is None:
            return direct
        source_to_chunks: dict[str, list[DocumentChunk]] = defaultdict(list)
        row_to_chunks: dict[int, list[DocumentChunk]] = defaultdict(list)
        for chunk in all_chunks:
            source_to_chunks[chunk.source].append(chunk)
            if chunk.row_index is not None:
                row_to_chunks[int(chunk.row_index)].append(chunk)
        seen_ids = {chunk.id for chunk in direct}
        extras: list[DocumentChunk] = []
        for result in results:
            frontier = {entity["id"] for entity in self._entities_for_chunk(result.chunk, entities) if entity["id"] in graph.nodes}
            visited: set[str] = set()
            for _ in range(self.config.graph_hop_depth):
                next_frontier: set[str] = set()
                for node_id in frontier:
                    if node_id in visited:
                        continue
                    visited.add(node_id)
                    neighbors = sorted(graph.neighbors(node_id), key=lambda n: graph.degree(n), reverse=True)[: self.config.graph_max_neighbors]
                    for neighbor in neighbors:
                        node = graph.nodes[neighbor]
                        for chunk in source_to_chunks.get(node.get("source_file", ""), []):
                            if chunk.id not in seen_ids:
                                extras.append(chunk)
                                seen_ids.add(chunk.id)
                        row_index = node.get("row_index")
                        if row_index is not None:
                            for chunk in row_to_chunks.get(int(row_index), []):
                                if chunk.id not in seen_ids:
                                    extras.append(chunk)
                                    seen_ids.add(chunk.id)
                        next_frontier.add(neighbor)
                frontier = next_frontier - visited
        return direct + extras[: self.config.graph_max_neighbors * 3]

    def _entities_for_chunk(self, chunk: DocumentChunk, entities: list[dict]) -> list[dict]:
        if chunk.type == "excel":
            return [entity for entity in entities if entity.get("row_index") == chunk.row_index]
        return [entity for entity in entities if entity.get("source_file") == chunk.source]
