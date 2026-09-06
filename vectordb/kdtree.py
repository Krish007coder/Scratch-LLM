"""
KD-Tree (K-Dimensional Tree) — binary space partitioning.
Complexity: O(log N) average case; degrades to O(N) in high dimensions
           (curse of dimensionality — use HNSW for 768D).

This is a direct Python port of the C++ KDNode / KDTree classes.
"""
from __future__ import annotations

import heapq
import math
from typing import List, Optional, Tuple

from .metrics import DistFn
from .types import VectorItem

DistId = Tuple[float, int]


class _KDNode:
    __slots__ = ("item", "left", "right")

    def __init__(self, item: VectorItem) -> None:
        self.item: VectorItem = item
        self.left: Optional[_KDNode] = None
        self.right: Optional[_KDNode] = None


class KDTree:
    """
    Axis-cycling KD-Tree with max-heap pruning for KNN queries.
    The tree must be rebuilt after deletions (same behaviour as C++ version).
    """

    def __init__(self, dims: int) -> None:
        self.dims = dims
        self._root: Optional[_KDNode] = None

    # ------------------------------------------------------------------ build

    def insert(self, item: VectorItem) -> None:
        self._root = self._insert(self._root, item, depth=0)

    def _insert(
        self, node: Optional[_KDNode], item: VectorItem, depth: int
    ) -> _KDNode:
        if node is None:
            return _KDNode(item)
        axis = depth % self.dims
        if item.emb[axis] < node.item.emb[axis]:
            node.left = self._insert(node.left, item, depth + 1)
        else:
            node.right = self._insert(node.right, item, depth + 1)
        return node

    def rebuild(self, items: List[VectorItem]) -> None:
        """Rebuild the tree from scratch (used after deletions)."""
        self._root = None
        for item in items:
            self.insert(item)

    # ------------------------------------------------------------------ query

    def knn(self, q: List[float], k: int, dist: DistFn) -> List[DistId]:
        """Return up to k nearest neighbors, sorted by distance ascending."""
        # Max-heap: Python's heapq is a min-heap, so we store negative distances.
        heap: List[Tuple[float, int]] = []  # (-dist, id)
        self._knn(self._root, q, k, depth=0, dist=dist, heap=heap)
        result = [(-d, nid) for d, nid in heap]
        result.sort()
        return result

    def _knn(
        self,
        node: Optional[_KDNode],
        q: List[float],
        k: int,
        depth: int,
        dist: DistFn,
        heap: List[Tuple[float, int]],
    ) -> None:
        if node is None:
            return

        dn = dist(q, node.item.emb)
        # Heap stores (-dist, id) so the worst (largest dist) is at index 0.
        if len(heap) < k or dn < -heap[0][0]:
            heapq.heappush(heap, (-dn, node.item.id))
            if len(heap) > k:
                heapq.heappop(heap)

        axis = depth % self.dims
        diff = q[axis] - node.item.emb[axis]
        closer = node.left if diff < 0 else node.right
        farther = node.right if diff < 0 else node.left

        self._knn(closer, q, k, depth + 1, dist, heap)

        # Only explore the farther subtree if it could contain a closer point.
        if len(heap) < k or abs(diff) < -heap[0][0]:
            self._knn(farther, q, k, depth + 1, dist, heap)
