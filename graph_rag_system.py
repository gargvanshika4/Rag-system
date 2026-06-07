#!/usr/bin/env python3
"""Backward-compatible import shim for the modular Graph RAG package."""

from graph_rag import GraphRAGSystem, KnowledgeGraphRAG

__all__ = ["KnowledgeGraphRAG", "GraphRAGSystem"]


def main() -> None:
    """Run a small CLI smoke demo."""
    import json
    import os

    rag = KnowledgeGraphRAG(os.getenv("GROQ_API_KEY"), use_ai=bool(os.getenv("GROQ_API_KEY")))
    rag.build_system("data/excel", "data")
    print(json.dumps(rag.get_system_statistics(), indent=2))


if __name__ == "__main__":
    main()
