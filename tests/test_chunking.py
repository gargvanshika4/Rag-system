from graph_rag.config import RAGConfig
from graph_rag.ingestion import AdaptiveSemanticChunker


def test_adaptive_chunker_keeps_semantic_boundaries():
    chunker = AdaptiveSemanticChunker(RAGConfig(chunk_min_chars=40, chunk_target_chars=80, chunk_max_chars=120))
    text = "Invoice Summary. This supplier billed INR 100. " * 8
    chunks = chunker.split(text)
    assert len(chunks) > 1
    assert all(len(chunk) <= 140 for chunk in chunks)
    assert all("Invoice" in chunk or "supplier" in chunk for chunk in chunks)
