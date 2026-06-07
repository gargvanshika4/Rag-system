"""Hybrid dense/BM25 retrieval, embedding cache and FAISS HNSW indexing."""

from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Any

import numpy as np

from graph_rag.config import RAGConfig
from graph_rag.models import DocumentChunk, SearchResult
from graph_rag.utils import CacheManager, tokenize

logger = logging.getLogger(__name__)


class BM25Index:
    """Small Okapi BM25 implementation without extra dependencies."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.docs: list[list[str]] = []
        self.doc_len: list[int] = []
        self.idf: dict[str, float] = {}
        self.avgdl = 0.0

    def build(self, texts: list[str]) -> None:
        """Build token statistics."""
        self.docs = [tokenize(text) for text in texts]
        self.doc_len = [len(doc) for doc in self.docs]
        total_docs = len(self.docs)
        self.avgdl = sum(self.doc_len) / total_docs if total_docs else 0.0
        df: Counter[str] = Counter()
        for doc in self.docs:
            df.update(set(doc))
        self.idf = {term: max(0.0, math.log((total_docs - freq + 0.5) / (freq + 0.5) + 1.0)) for term, freq in df.items()}

    def score(self, query_tokens: list[str], doc_idx: int) -> float:
        """Score a document for query tokens."""
        if doc_idx < 0 or doc_idx >= len(self.docs) or not self.docs[doc_idx]:
            return 0.0
        tf = Counter(self.docs[doc_idx])
        dl = self.doc_len[doc_idx] or 1
        avgdl = self.avgdl or 1.0
        score = 0.0
        for token in query_tokens:
            freq = tf.get(token, 0)
            if not freq:
                continue
            denom = freq + self.config.bm25_k1 * (1 - self.config.bm25_b + self.config.bm25_b * dl / avgdl)
            score += self.idf.get(token, 0.0) * (freq * (self.config.bm25_k1 + 1) / denom)
        return float(score)

    def search(self, query: str, candidate_ids: list[int] | None = None, k: int = 20) -> list[tuple[int, float]]:
        """Return top BM25 results."""
        query_tokens = tokenize(query)
        ids = candidate_ids if candidate_ids is not None else list(range(len(self.docs)))
        scored = [(idx, self.score(query_tokens, idx)) for idx in ids]
        return sorted(scored, key=lambda item: item[1], reverse=True)[:k]


class HybridRetriever:
    """Dense vector + BM25 retrieval with reciprocal rank fusion."""

    def __init__(self, config: RAGConfig, cache: CacheManager):
        self.config = config
        self.cache = cache
        self.model: Any | None = None
        self.index: Any | None = None
        self.bm25 = BM25Index(config)
        self.chunks: list[DocumentChunk] = []
        self.embeddings: np.ndarray | None = None

    def build(self, chunks: list[DocumentChunk], corpus_key: str) -> None:
        """Build or refresh indexes for chunks."""
        self.chunks = chunks
        texts = [chunk.text for chunk in chunks]
        self.bm25.build(texts)
        if self.model is None:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.config.model_name)
        cache_key = f"embeddings_{self.config.model_name}_{corpus_key}_{len(chunks)}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            embeddings = np.asarray(cached, dtype="float32")
        else:
            embeddings = self.model.encode(texts, batch_size=128, show_progress_bar=False, convert_to_numpy=True).astype("float32")
            embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
            self.cache.set(cache_key, embeddings)
        self.embeddings = embeddings
        self.index = self._build_hnsw(embeddings)

    def _build_hnsw(self, embeddings: np.ndarray) -> Any:
        """Build a cosine-search HNSW index."""
        import faiss

        if embeddings.size == 0:
            raise ValueError("Cannot build FAISS index with no embeddings")
        index = faiss.IndexHNSWFlat(embeddings.shape[1], self.config.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = self.config.hnsw_ef_construction
        index.hnsw.efSearch = self.config.hnsw_ef_search
        index.add(embeddings)
        return index

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        """Search dense and lexical indexes, then fuse rankings."""
        if self.index is None or self.model is None or self.embeddings is None:
            raise ValueError("Retriever index has not been built")
        search_k = max(k * 5, 50)
        query_embedding = self.model.encode([query], convert_to_numpy=True).astype("float32")
        query_embedding /= np.linalg.norm(query_embedding, axis=1, keepdims=True) + 1e-12
        distances, indices = self.index.search(query_embedding, min(search_k, len(self.chunks)))
        dense_all = [(int(idx), float(distances[0][pos])) for pos, idx in enumerate(indices[0]) if int(idx) >= 0]
        dense_rank = dense_all[:search_k]
        bm25_rank = self.bm25.search(query, k=search_k)
        fused = self.rrf_fuse([idx for idx, _ in dense_rank], [idx for idx, _ in bm25_rank])[:search_k]
        dense_scores = dict(dense_rank)
        bm25_scores = dict(bm25_rank)
        results = [
            SearchResult(
                chunk=self.chunks[idx],
                score=rrf,
                confidence=min(1.0, rrf * 30),
                dense_score=dense_scores.get(idx, 0.0),
                bm25_score=bm25_scores.get(idx, 0.0),
                rrf_score=rrf,
            )
            for idx, rrf in fused
        ]
        return results[:k]

    def rrf_fuse(self, dense_ids: list[int], bm25_ids: list[int]) -> list[tuple[int, float]]:
        """Fuse rankings using reciprocal rank fusion."""
        scores: dict[int, float] = {}
        for rank, doc_id in enumerate(dense_ids, 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self.config.rrf_k + rank)
        for rank, doc_id in enumerate(bm25_ids, 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self.config.rrf_k + rank)
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)
