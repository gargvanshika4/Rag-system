from graph_rag.config import RAGConfig
from graph_rag.retrieval import BM25Index, HybridRetriever
from graph_rag.utils import CacheManager


def test_bm25_prioritizes_keyword_match():
    bm25 = BM25Index(RAGConfig())
    bm25.build(["alpha invoice supplier", "beta purchase order"])
    assert bm25.search("invoice", k=1)[0][0] == 0


def test_rrf_fusion_promotes_documents_seen_by_both_rankers(tmp_path):
    retriever = HybridRetriever(RAGConfig(cache_dir=str(tmp_path)), CacheManager(str(tmp_path)))
    fused = retriever.rrf_fuse([1, 2, 3], [3, 1, 4])
    assert fused[0][0] in {1, 3}
    assert {doc_id for doc_id, _ in fused} == {1, 2, 3, 4}
