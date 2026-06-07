"""High-level Graph RAG orchestration API."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
from groq import Groq
from neo4j import GraphDatabase

from graph_rag.config import RAGConfig
from graph_rag.graph_builder import GraphRelationshipBuilder
from graph_rag.ingestion import DocumentIngestor
from graph_rag.memory import ConversationMemory
from graph_rag.models import DocumentChunk, SearchResult
from graph_rag.ranking import ConfidenceRanker
from graph_rag.retrieval import HybridRetriever
from graph_rag.utils import CacheManager, normalize_text

logger = logging.getLogger(__name__)


class QueryIntentClassifier:
    """Rule-based query intent routing."""

    patterns = {
        "DOC_SPECIFIC": [r"\bPO[-\s]?\w+", r"\bGRN[-\s]?\w+", r"\bINV[-\s]?\w+", r"\binvoice\s+(?:no|number|#)", r"\bdocument\s+(?:no|number)"],
        "ADDRESS": [r"\baddress\b", r"\blocation\b", r"\bwhere\s+is\b", r"\bpin\s*code\b"],
        "AGG": [r"\btotal\b", r"\bsum\b", r"\bhow many\b", r"\bcount\b", r"\baverage\b", r"\bbreakdown\b", r"\bsummary\b"],
        "COMPARISON": [r"\bcompare\b", r"\bvs\.?\b", r"\bversus\b", r"\bdifference between\b"],
    }

    def classify(self, query: str) -> str:
        """Return one of DOC_SPECIFIC, ADDRESS, AGG, COMPARISON or GENERAL."""
        for intent, patterns in self.patterns.items():
            if any(re.search(pattern, query, flags=re.IGNORECASE) for pattern in patterns):
                return intent
        return "GENERAL"

    def k_for_intent(self, intent: str, base_k: int) -> int:
        """Adaptive retrieval depth per intent."""
        return {"AGG": 30, "ADDRESS": 40, "COMPARISON": 20, "DOC_SPECIFIC": base_k, "GENERAL": base_k}.get(intent, base_k)


class KnowledgeGraphRAG:
    """Production-oriented Graph RAG facade used by Flask and scripts."""

    def __init__(self, groq_api_key: str | None = None, use_ai: bool = True, cache_dir: str = "cache", config: RAGConfig | None = None):
        self.config = config or RAGConfig(cache_dir=cache_dir)
        self.cache_manager = CacheManager(self.config.cache_dir, self.config.cache_version)
        self.ingestor = DocumentIngestor(self.config, self.cache_manager)
        self.graph_builder = GraphRelationshipBuilder()
        self.retriever = HybridRetriever(self.config, self.cache_manager)
        self.ranker = ConfidenceRanker(self.config)
        self.intent_clf = QueryIntentClassifier()
        self.memory = ConversationMemory(self.config.max_history_turns)
        self.groq_api_key = groq_api_key if use_ai else None
        self.groq_client = Groq(api_key=groq_api_key) if groq_api_key and use_ai else None

        self.df: pd.DataFrame | None = None
        self.graph: nx.Graph | None = None
        self.entities: list[dict[str, Any]] = []
        self.relationships: list[dict[str, Any]] = []
        self.metadata_list: list[dict[str, Any]] = []
        self.chunks: list[DocumentChunk] = []
        self.build_info: dict[str, Any] = {}

    def build_system(self, excel_path: str = "data/excel", documents_folder: str = "data") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Build ingestion, graph, vector and BM25 indexes."""
        corpus_key = self.cache_manager.corpus_key(documents_folder)
        self.df = self.ingestor.load_excel_data(excel_path)
        document_chunks = asyncio.run(self.ingestor.load_documents(documents_folder))
        excel_chunks = self.ingestor.create_excel_chunks(self.df)
        self.chunks = excel_chunks + document_chunks
        if not self.chunks:
            raise ValueError("No chunks found. Add spreadsheets or documents under data/.")

        excel_entities = self.graph_builder.extract_entities_from_excel(self.df)
        document_entities = self.graph_builder.extract_entities_from_documents(document_chunks)
        self.entities = excel_entities + document_entities
        self.relationships = self.graph_builder.build_relationships(excel_entities, document_entities)
        self.graph = self.graph_builder.create_network_graph(self.entities, self.relationships)
        self.ranker.fit_graph(self.graph)
        self.retriever.build(self.chunks, corpus_key)
        self.metadata_list = [chunk.to_dict() for chunk in self.chunks]
        self.build_info = {
            "indexed_chunks": len(self.chunks),
            "nodes": self.graph.number_of_nodes() if self.graph else 0,
            "edges": self.graph.number_of_edges() if self.graph else 0,
            "index": "FAISS HNSW + BM25 + RRF",
        }
        return self.entities, self.relationships

    def search(self, query: str, k: int = 10, conversation_id: str = "default") -> list[dict[str, Any]]:
        """Return ranked result dictionaries for compatibility callers."""
        return [result.to_dict() for result in self.search_results(query, k, conversation_id)]

    def search_results(self, query: str, k: int = 10, conversation_id: str = "default") -> list[SearchResult]:
        """Return typed ranked results with score components."""
        retrieval_query = self.memory.rewrite_query(conversation_id, query)
        results = self.retriever.search(retrieval_query, k=max(k, 5))
        return self.ranker.rerank(results, self.entities, self.graph)[:k]

    def search_and_answer(self, query: str, base_k: int = 10, conversation_id: str = "default") -> dict[str, Any]:
        """Full multi-turn RAG pipeline."""
        intent = self.intent_clf.classify(query)
        k = self.intent_clf.k_for_intent(intent, base_k)
        results = self.search_results(query, k=k, conversation_id=conversation_id)
        confidence = self.ranker.label(results)
        address_answer = self._address_shortcut(query, intent)
        if address_answer:
            answer = address_answer
            context_chunks = [result.chunk for result in results]
        else:
            context_chunks = self.ranker.graph_augment_context(results, self.entities, self.graph, self.chunks)
            answer = self.query_with_groq(query, context_chunks[:10], intent, confidence, conversation_id)
        self.memory.add(conversation_id, query, answer)
        related = self.find_related_documents(query, max_results=10)
        return {
            "query": query,
            "intent": intent,
            "confidence": confidence,
            "answer": answer,
            "search_results": [result.to_dict() for result in results],
            "sources": [result.to_dict() for result in results[:8]],
            "related_documents": related,
            "num_results": len(results),
            "num_relationships": len(related),
            "conversation_id": conversation_id,
            "chat_history": self.memory.history(conversation_id),
        }

    def _address_shortcut(self, query: str, intent: str) -> str | None:
        if intent != "ADDRESS":
            return None
        match = re.search(r"address\s+of\s+(.+)$", query, flags=re.IGNORECASE)
        if not match:
            return None
        wanted = normalize_text(match.group(1)).lower()
        for chunk in self.chunks:
            fields = chunk.structured_data
            for key in ("supplier_name", "customer_name"):
                if wanted and normalize_text(fields.get(key)).lower() == wanted and fields.get("address"):
                    return f"Address: {fields['address']} (source: {chunk.source})"
        return None

    def query_with_groq(self, query: str, context_chunks: list[DocumentChunk], intent: str = "GENERAL", confidence: str = "MEDIUM", conversation_id: str = "default") -> str:
        """Generate an answer using Groq, or return a source-grounded fallback."""
        if not self.groq_client:
            citations = ", ".join(f"{chunk.source}#{chunk.chunk_index}" for chunk in context_chunks[:4])
            preview = "\n\n".join(f"[{chunk.source}#{chunk.chunk_index}] {chunk.text[:450]}" for chunk in context_chunks[:3])
            return f"LLM disabled because GROQ_API_KEY is not configured. Retrieved sources: {citations}\n\nTop context:\n{preview}"
        prompt = self._build_prompt(query, context_chunks, intent, confidence, conversation_id)
        try:
            response = self.groq_client.chat.completions.create(
                model=self.config.groq_model,
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        except Exception as exc:
            logger.exception("Groq answer generation failed")
            return f"Groq answer generation failed: {exc}"

    def _build_prompt(self, query: str, chunks: list[DocumentChunk], intent: str, confidence: str, conversation_id: str) -> str:
        context = "\n\n---\n\n".join(f"[{chunk.source}#{chunk.chunk_index} | {chunk.document_type}]\n{chunk.text[:900]}" for chunk in chunks)
        history = self.memory.context(conversation_id)
        return f"""You are a careful document intelligence assistant.

Intent: {intent}
Retrieval confidence: {confidence}

Conversation history:
{history or "No prior turns."}

Retrieved context with citations:
{context}

Question: {query}

Answer only from the retrieved context. Cite source ids such as [filename#chunk]. If evidence is incomplete, say what is missing."""

    def find_related_documents(self, query: str, max_results: int = 10) -> list[dict[str, Any]]:
        """Return high-strength graph relationships around retrieved results."""
        if not self.graph:
            return []
        results = self.search_results(query, k=10)
        related: list[dict[str, Any]] = []
        visited_edges: set[tuple[str, str]] = set()
        for result in results:
            seed_entities = self.ranker._entities_for_chunk(result.chunk, self.entities)
            for entity in seed_entities:
                node_id = entity["id"]
                if node_id not in self.graph:
                    continue
                for neighbor in self.graph.neighbors(node_id):
                    edge_key = tuple(sorted([node_id, neighbor]))
                    if edge_key in visited_edges:
                        continue
                    visited_edges.add(edge_key)
                    node = self.graph.nodes[neighbor]
                    edge = self.graph.edges[node_id, neighbor]
                    related.append(
                        {
                            "entity_name": entity.get("name"),
                            "related_entity": node.get("name", neighbor),
                            "relationship_type": edge.get("type", "RELATED"),
                            "relationship_strength": edge.get("strength", 0.0),
                            "related_source": node.get("source_file") or node.get("source"),
                            "related_type": node.get("type"),
                            "description": edge.get("description", ""),
                        }
                    )
        related.sort(key=lambda item: item["relationship_strength"], reverse=True)
        return related[:max_results]

    def get_graph_data(self, query: str = "", max_nodes: int = 80) -> dict[str, Any]:
        """Return graph data for browser visualization."""
        if not self.graph:
            return {"nodes": [], "links": []}
        if query:
            results = self.search_results(query, k=15)
            node_ids: set[str] = set()
            for result in results:
                for entity in self.ranker._entities_for_chunk(result.chunk, self.entities):
                    if entity["id"] in self.graph:
                        node_ids.add(entity["id"])
                        node_ids.update(list(self.graph.neighbors(entity["id"]))[: self.config.graph_max_neighbors])
        else:
            node_ids = set(sorted(self.graph.nodes, key=lambda node: self.graph.degree(node), reverse=True)[:max_nodes])
        if len(node_ids) > max_nodes:
            node_ids = set(sorted(node_ids, key=lambda node: self.graph.degree(node), reverse=True)[:max_nodes])
        nodes = [
            {"id": node_id, "name": self.graph.nodes[node_id].get("name", node_id), "type": self.graph.nodes[node_id].get("type", "unknown"), "source": self.graph.nodes[node_id].get("source", "unknown")}
            for node_id in node_ids
        ]
        links = []
        for source, target, edge in self.graph.edges(data=True):
            if source in node_ids and target in node_ids:
                links.append({"source": source, "target": target, "type": edge.get("type", "RELATED"), "strength": edge.get("strength", 0.5)})
        return {"nodes": nodes, "links": links}

    def get_system_statistics(self) -> dict[str, Any]:
        """Return operational stats and index metadata."""
        if not self.graph:
            return {"error": "System not built"}
        node_types: dict[str, int] = {}
        edge_types: dict[str, int] = {}
        for _, data in self.graph.nodes(data=True):
            node_types[data.get("type", "unknown")] = node_types.get(data.get("type", "unknown"), 0) + 1
        for _, _, data in self.graph.edges(data=True):
            edge_types[data.get("type", "RELATED")] = edge_types.get(data.get("type", "RELATED"), 0) + 1
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "connected_components": nx.number_connected_components(self.graph),
            "node_types": node_types,
            "edge_types": edge_types,
            "total_chunks": len(self.chunks),
            "bm25_documents": len(self.retriever.bm25.docs),
            "faiss_index": "HNSW",
            "build_info": self.build_info,
        }

    def clear_cache(self) -> dict[str, Any]:
        """Clear all versioned local caches."""
        self.cache_manager.clear()
        return {"success": True, "message": "Cache cleared"}

    def export_graph_data(self, output_file: str = "graph_data.json") -> dict[str, Any]:
        """Export full graph JSON."""
        data = self.get_graph_data(max_nodes=10_000)
        Path(output_file).write_text(json.dumps(data, indent=2), encoding="utf-8")
        return data

    def export_to_neo4j(self, neo4j_uri: str = "bolt://localhost:7687", neo4j_user: str = "neo4j", neo4j_password: str = "password", clear_database: bool = True) -> dict[str, Any]:
        """Export the NetworkX graph to Neo4j."""
        if not self.graph:
            return {"error": "Graph not built"}
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        node_count = rel_count = 0
        try:
            with driver.session() as session:
                if clear_database:
                    session.run("MATCH (n) DETACH DELETE n")
                for node_id, data in self.graph.nodes(data=True):
                    labels = ["Entity", str(data.get("type", "Unknown")).title().replace("_", "")]
                    session.run(f"CREATE (n:{':'.join(labels)}) SET n = $props", props={"id": node_id, **data})
                    node_count += 1
                for source, target, data in self.graph.edges(data=True):
                    rel_type = re.sub(r"[^A-Z0-9_]", "_", str(data.get("type", "RELATED")).upper())
                    session.run(
                        f"MATCH (a {{id:$source}}),(b {{id:$target}}) CREATE (a)-[r:{rel_type}]->(b) SET r = $props",
                        source=source,
                        target=target,
                        props=data,
                    )
                    rel_count += 1
        finally:
            driver.close()
        return {"success": True, "nodes_created": node_count, "relationships_created": rel_count}
