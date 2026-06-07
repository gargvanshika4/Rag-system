"""Shared typed records used across Graph RAG modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentChunk:
    """A retrievable unit with source and document metadata."""

    id: str
    text: str
    source: str
    type: str
    metadata: dict[str, Any] = field(default_factory=dict)
    document_type: str = "document"
    row_index: int | None = None
    chunk_index: int = 0
    structured_data: dict[str, Any] = field(default_factory=dict)
    row_data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/pickle-friendly representation."""
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "type": self.type,
            "metadata": self.metadata,
            "document_type": self.document_type,
            "row_index": self.row_index,
            "chunk_index": self.chunk_index,
            "structured_data": self.structured_data,
            "row_data": self.row_data,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocumentChunk":
        """Build a chunk from legacy or current dict data."""
        return cls(
            id=str(data.get("id", "")),
            text=str(data.get("text", "")),
            source=str(data.get("source", "unknown")),
            type=str(data.get("type", "document")),
            metadata=dict(data.get("metadata") or {}),
            document_type=str(data.get("document_type", "document")),
            row_index=data.get("row_index"),
            chunk_index=int(data.get("chunk_index", 0) or 0),
            structured_data=dict(data.get("structured_data") or {}),
            row_data=dict(data.get("row_data") or {}),
        )


@dataclass
class SearchResult:
    """A ranked retrieval result with explainable score components."""

    chunk: DocumentChunk
    score: float
    confidence: float
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    centrality: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return an API-compatible result dictionary."""
        data = self.chunk.to_dict()
        data.update(
            {
                "similarity": self.score,
                "confidence_score": self.confidence,
                "dense_score": self.dense_score,
                "bm25_score": self.bm25_score,
                "rrf_score": self.rrf_score,
                "centrality": self.centrality,
                "citation": f"{self.chunk.source}#{self.chunk.chunk_index}",
            }
        )
        return data
