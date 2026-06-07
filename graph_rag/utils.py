"""Utility helpers for caching, hashing and text normalisation."""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from unicodedata import normalize as unicode_normalize

from graph_rag.config import RAGConfig

logger = logging.getLogger(__name__)


def normalize_text(value: Any) -> str:
    """Normalize whitespace and unicode variants."""
    if value is None:
        return ""
    text = unicode_normalize("NFKC", str(value).replace("\xa0", " "))
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 and light lexical metrics."""
    return [t for t in re.findall(r"[A-Za-z0-9]+", text.lower()) if len(t) > 2]


def file_fingerprint(path: str | Path) -> str:
    """Stable fingerprint from path, size and modification time."""
    p = Path(path)
    stat = p.stat()
    raw = f"{p.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class CacheManager:
    """Versioned pickle cache for documents, embeddings and manifests."""

    def __init__(self, cache_dir: str = RAGConfig().cache_dir, version: str = RAGConfig().cache_version):
        self.cache_dir = Path(cache_dir)
        self.version = version
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        """Return cache path for key."""
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", key)
        return self.cache_dir / f"{safe}.pkl"

    def get(self, key: str) -> Any | None:
        """Read a versioned cache entry."""
        path = self.path(key)
        if not path.exists():
            return None
        try:
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            if payload.get("version") == self.version:
                return payload.get("data")
        except Exception as exc:
            logger.warning("Cache read failed for %s: %s", key, exc)
        return None

    def set(self, key: str, data: Any) -> None:
        """Write a versioned cache entry."""
        payload = {
            "version": self.version,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data,
        }
        with self.path(key).open("wb") as handle:
            pickle.dump(payload, handle)

    def clear(self) -> None:
        """Remove cached pickle files."""
        for path in self.cache_dir.glob("*.pkl"):
            try:
                path.unlink()
            except OSError as exc:
                logger.warning("Could not remove cache file %s: %s", path, exc)

    def corpus_key(self, root: str | Path) -> str:
        """Fingerprint all files below a file or directory."""
        p = Path(root)
        if p.is_file():
            return file_fingerprint(p)
        hashes: list[str] = []
        for dirpath, _, filenames in os.walk(p):
            for filename in sorted(filenames):
                path = Path(dirpath) / filename
                if path.is_file():
                    hashes.append(file_fingerprint(path))
        return hashlib.sha256("|".join(hashes).encode("utf-8")).hexdigest()
