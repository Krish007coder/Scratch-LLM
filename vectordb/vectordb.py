"""
VectorDB — thread-safe unified interface over BruteForce, KD-Tree, and HNSW.
Used for the 16D demo vectors.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .bruteforce import BruteForce
from .hnsw import HNSW, GraphInfo
from .kdtree import KDTree
from .metrics import DistFn, get_dist_fn
from .types import VectorItem


@dataclass
class Hit:
    id: int
    meta: str
    cat: str
    emb: List[float]
    dist: float


@dataclass
class SearchOut:
    hits: List[Hit]
    latency_us: int
    algo: str
    metric: str


@dataclass
class BenchOut:
    bf_us: int
    kd_us: int
    hnsw_us: int
    item_count: int


class VectorDB:
    """
    Manages three concurrent indices (BruteForce, KDTree, HNSW) for the
    fixed-dimension demo vectors. All public methods are thread-safe.
    """

    def __init__(self, dims: int) -> None:
        self.dims = dims
        self._store: Dict[int, VectorItem] = {}
        self._bf = BruteForce()
        self._kdt = KDTree(dims)
        self._hnsw = HNSW(M=16, ef_build=200)
        self._lock = threading.Lock()
        self._next_id = 1

    # ------------------------------------------------------------------ write

    def insert(
        self,
        metadata: str,
        category: str,
        emb: List[float],
        dist: DistFn,
    ) -> int:
        with self._lock:
            item = VectorItem(id=self._next_id, metadata=metadata, category=category, emb=emb)
            self._next_id += 1
            self._store[item.id] = item
            self._bf.insert(item)
            self._kdt.insert(item)
            self._hnsw.insert(item, dist)
            return item.id

    def remove(self, item_id: int) -> bool:
        with self._lock:
            if item_id not in self._store:
                return False
            del self._store[item_id]
            self._bf.remove(item_id)
            self._hnsw.remove(item_id)
            # KD-Tree requires a full rebuild after deletion
            self._kdt.rebuild(list(self._store.values()))
            return True

    # ------------------------------------------------------------------ read

    def search(self, q: List[float], k: int, metric: str, algo: str) -> SearchOut:
        with self._lock:
            dist_fn = get_dist_fn(metric)
            t0 = time.perf_counter()

            if algo == "bruteforce":
                raw = self._bf.knn(q, k, dist_fn)
            elif algo == "kdtree":
                raw = self._kdt.knn(q, k, dist_fn)
            else:
                raw = self._hnsw.knn(q, k, ef=50, dist=dist_fn)

            latency_us = int((time.perf_counter() - t0) * 1_000_000)

            hits = [
                Hit(
                    id=item_id,
                    meta=self._store[item_id].metadata,
                    cat=self._store[item_id].category,
                    emb=self._store[item_id].emb,
                    dist=d,
                )
                for d, item_id in raw
                if item_id in self._store
            ]
            return SearchOut(hits=hits, latency_us=latency_us, algo=algo, metric=metric)

    def benchmark(self, q: List[float], k: int, metric: str) -> BenchOut:
        with self._lock:
            dist_fn = get_dist_fn(metric)

            def time_fn(fn):
                t = time.perf_counter()
                fn()
                return int((time.perf_counter() - t) * 1_000_000)

            bf_us = time_fn(lambda: self._bf.knn(q, k, dist_fn))
            kd_us = time_fn(lambda: self._kdt.knn(q, k, dist_fn))
            hnsw_us = time_fn(lambda: self._hnsw.knn(q, k, ef=50, dist=dist_fn))

            return BenchOut(
                bf_us=bf_us,
                kd_us=kd_us,
                hnsw_us=hnsw_us,
                item_count=len(self._store),
            )

    def all(self) -> List[VectorItem]:
        with self._lock:
            return list(self._store.values())

    def hnsw_info(self) -> GraphInfo:
        with self._lock:
            return self._hnsw.get_info()

    def size(self) -> int:
        with self._lock:
            return len(self._store)
