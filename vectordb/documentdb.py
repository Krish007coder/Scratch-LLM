"""
DocumentDB — HNSW + BruteForce index for real Ollama embeddings.
Embedding dimension is determined at runtime from the first inserted vector.
"""
from __future__ import annotations

import threading
from typing import Dict, List, Optional, Tuple

from .bruteforce import BruteForce
from .hnsw import HNSW
from .metrics import cosine
from .types import DocItem, VectorItem


class DocumentDB:
    """
    Stores document chunks with their pre-computed Ollama embeddings.
    Uses HNSW for large collections, BruteForce for small ones (< 10 items).
    """

    def __init__(self) -> None:
        self._store: Dict[int, DocItem] = {}
        self._hnsw = HNSW(M=16, ef_build=200)
        self._bf = BruteForce()
        self._lock = threading.Lock()
        self._next_id = 1
        self._dims: int = 0

    # ------------------------------------------------------------------ write

    def insert(self, title: str, text: str, emb: List[float]) -> int:
        with self._lock:
            if self._dims == 0:
                self._dims = len(emb)
            doc = DocItem(id=self._next_id, title=title, text=text, emb=emb)
            self._next_id += 1
            self._store[doc.id] = doc
            vi = VectorItem(id=doc.id, metadata=title, category="doc", emb=emb)
            self._hnsw.insert(vi, cosine)
            self._bf.insert(vi)
            return doc.id

    def remove(self, doc_id: int) -> bool:
        with self._lock:
            if doc_id not in self._store:
                return False
            del self._store[doc_id]
            self._hnsw.remove(doc_id)
            self._bf.remove(doc_id)
            return True

    # ------------------------------------------------------------------ read

    def search(
        self,
        q: List[float],
        k: int,
        max_dist: float = 0.7,
    ) -> List[Tuple[float, DocItem]]:
        with self._lock:
            if not self._store:
                return []
            if len(self._store) < 10:
                raw = self._bf.knn(q, k, cosine)
            else:
                raw = self._hnsw.knn(q, k, ef=50, dist=cosine)
            return [
                (d, self._store[item_id])
                for d, item_id in raw
                if item_id in self._store and d <= max_dist
            ]

    def all(self) -> List[DocItem]:
        with self._lock:
            return list(self._store.values())

    def size(self) -> int:
        with self._lock:
            return len(self._store)

    @property
    def dims(self) -> int:
        return self._dims
