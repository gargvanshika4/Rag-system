"""Central configuration for the Graph RAG system."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RAGConfig:
    """Runtime and indexing defaults."""

    cache_dir: str = "cache"
    cache_version: str = "3.0"
    model_name: str = "all-MiniLM-L6-v2"
    chunk_min_chars: int = 450
    chunk_target_chars: int = 1100
    chunk_max_chars: int = 1700
    chunk_overlap_sentences: int = 1
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    rrf_k: int = 60
    hnsw_m: int = 32
    hnsw_ef_construction: int = 80
    hnsw_ef_search: int = 64
    graph_hop_depth: int = 2
    graph_max_neighbors: int = 5
    semantic_weight: float = 0.65
    centrality_weight: float = 0.25
    lexical_weight: float = 0.10
    max_history_turns: int = 8
    groq_model: str = "llama-3.1-8b-instant"
