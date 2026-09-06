"""
Brute Force K-Nearest Neighbor search.
Complexity: O(N · d)  — exact, used as a baseline.
"""
from typing import List, Tuple

from .metrics import DistFn
from .types import VectorItem

# (distance, id)
DistId = Tuple[float, int]


class BruteForce:
    """Linear scan over all stored vectors."""

    def __init__(self) -> None:
        self.items: List[VectorItem] = []

    def insert(self, item: VectorItem) -> None:
        self.items.append(item)

    def knn(self, q: List[float], k: int, dist: DistFn) -> List[DistId]:
        """Return the k nearest neighbors sorted by distance ascending."""
        results: List[DistId] = [(dist(q, v.emb), v.id) for v in self.items]
        results.sort(key=lambda x: x[0])
        return results[:k]

    def remove(self, item_id: int) -> None:
        self.items = [v for v in self.items if v.id != item_id]
