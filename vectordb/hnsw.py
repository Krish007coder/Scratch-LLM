"""
HNSW — Hierarchical Navigable Small World graph index.
O(log N) approximate nearest neighbor search.

This is a direct Python port of the C++ HNSW class in main.cpp.
Same algorithm used by Pinecone, Weaviate, Chroma, and Milvus.
"""
from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .metrics import DistFn
from .types import VectorItem

DistId = Tuple[float, int]


@dataclass
class _Node:
    item: VectorItem
    max_lyr: int
    # nbrs[layer] = list of neighbor IDs at that layer
    nbrs: List[List[int]] = field(default_factory=list)


@dataclass
class GraphInfo:
    top_layer: int
    node_count: int
    nodes_per_layer: List[int]
    edges_per_layer: List[int]
    nodes: List[dict]
    edges: List[dict]


class HNSW:
    """
    Hierarchical Navigable Small World graph.

    Parameters
    ----------
    M         : max connections per node per layer (default 16)
    ef_build  : beam width during construction (default 200)
    seed      : RNG seed for reproducibility
    """

    def __init__(self, M: int = 16, ef_build: int = 200, seed: int = 42) -> None:
        self.M = M
        self.M0 = 2 * M            # layer-0 has more connections
        self.ef_build = ef_build
        self.mL = 1.0 / math.log(M)  # level multiplier
        self._rng = random.Random(seed)
        self._graph: Dict[int, _Node] = {}
        self._top_layer: int = -1
        self._entry_pt: Optional[int] = None

    # ------------------------------------------------------------------ utils

    def _rand_level(self) -> int:
        return int(math.floor(-math.log(self._rng.random()) * self.mL))

    # ------------------------------------------------------------------ layer search

    def _search_layer(
        self,
        q: List[float],
        ep: int,
        ef: int,
        lyr: int,
        dist: DistFn,
    ) -> List[DistId]:
        """
        Beam search on a single layer.
        Returns up to `ef` candidates sorted by distance ascending.
        """
        visited: set = {ep}
        d0 = dist(q, self._graph[ep].item.emb)

        # candidates: min-heap by distance
        cands: List[DistId] = [(d0, ep)]
        # found: max-heap (negate distances)
        found: List[Tuple[float, int]] = [(-d0, ep)]

        while cands:
            cd, cid = heapq.heappop(cands)
            # If the closest candidate is worse than our worst found, stop.
            if len(found) >= ef and cd > -found[0][0]:
                break
            node = self._graph.get(cid)
            if node is None or lyr >= len(node.nbrs):
                continue
            for nid in node.nbrs[lyr]:
                if nid in visited or nid not in self._graph:
                    continue
                visited.add(nid)
                nd = dist(q, self._graph[nid].item.emb)
                if len(found) < ef or nd < -found[0][0]:
                    heapq.heappush(cands, (nd, nid))
                    heapq.heappush(found, (-nd, nid))
                    if len(found) > ef:
                        heapq.heappop(found)

        result = [(-d, nid) for d, nid in found]
        result.sort()
        return result

    def _select_nbrs(self, cands: List[DistId], max_m: int) -> List[int]:
        return [nid for _, nid in cands[:max_m]]

    # ------------------------------------------------------------------ insert

    def insert(self, item: VectorItem, dist: DistFn) -> None:
        node_id = item.id
        lvl = self._rand_level()
        node = _Node(item=item, max_lyr=lvl, nbrs=[[] for _ in range(lvl + 1)])
        self._graph[node_id] = node

        if self._entry_pt is None:
            self._entry_pt = node_id
            self._top_layer = lvl
            return

        ep = self._entry_pt

        # Greedy descent from top layer down to lvl+1 (single candidate)
        for lc in range(self._top_layer, lvl, -1):
            if self._graph[ep].nbrs and lc < len(self._graph[ep].nbrs):
                W = self._search_layer(item.emb, ep, 1, lc, dist)
                if W:
                    ep = W[0][1]

        # Beam search and connect from min(top_layer, lvl) down to 0
        for lc in range(min(self._top_layer, lvl), -1, -1):
            W = self._search_layer(item.emb, ep, self.ef_build, lc, dist)
            max_m = self.M0 if lc == 0 else self.M
            sel = self._select_nbrs(W, max_m)
            self._graph[node_id].nbrs[lc] = sel

            for nid in sel:
                if nid not in self._graph:
                    continue
                nbr_node = self._graph[nid]
                # Ensure the neighbor has enough layers
                while len(nbr_node.nbrs) <= lc:
                    nbr_node.nbrs.append([])
                conn = nbr_node.nbrs[lc]
                conn.append(node_id)
                # Trim neighbor list if it exceeds max_m
                if len(conn) > max_m:
                    ds = [
                        (dist(nbr_node.item.emb, self._graph[c].item.emb), c)
                        for c in conn
                        if c in self._graph
                    ]
                    ds.sort()
                    nbr_node.nbrs[lc] = [c for _, c in ds[:max_m]]

            if W:
                ep = W[0][1]

        if lvl > self._top_layer:
            self._top_layer = lvl
            self._entry_pt = node_id

    # ------------------------------------------------------------------ search

    def knn(
        self, q: List[float], k: int, ef: int, dist: DistFn
    ) -> List[DistId]:
        """Return up to k approximate nearest neighbors."""
        if self._entry_pt is None:
            return []

        ep = self._entry_pt

        for lc in range(self._top_layer, 0, -1):
            node = self._graph.get(ep)
            if node and lc < len(node.nbrs):
                W = self._search_layer(q, ep, 1, lc, dist)
                if W:
                    ep = W[0][1]

        W = self._search_layer(q, ep, max(ef, k), 0, dist)
        return W[:k]

    # ------------------------------------------------------------------ delete

    def remove(self, item_id: int) -> None:
        if item_id not in self._graph:
            return
        # Remove this node from all neighbor lists
        for node in self._graph.values():
            for layer in node.nbrs:
                if item_id in layer:
                    layer.remove(item_id)
        # Update entry point if needed
        if self._entry_pt == item_id:
            self._entry_pt = next(
                (nid for nid in self._graph if nid != item_id), None
            )
        del self._graph[item_id]

    # ------------------------------------------------------------------ info

    def get_info(self) -> GraphInfo:
        max_l = max(self._top_layer + 1, 1)
        nodes_per_layer = [0] * max_l
        edges_per_layer = [0] * max_l
        nodes_out = []
        edges_out = []

        for node_id, node in self._graph.items():
            nodes_out.append({
                "id": node_id,
                "metadata": node.item.metadata,
                "category": node.item.category,
                "maxLyr": node.max_lyr,
            })
            for lc in range(min(node.max_lyr + 1, max_l)):
                nodes_per_layer[lc] += 1
                if lc < len(node.nbrs):
                    for nid in node.nbrs[lc]:
                        if node_id < nid:
                            edges_per_layer[lc] += 1
                            edges_out.append({"src": node_id, "dst": nid, "lyr": lc})

        return GraphInfo(
            top_layer=self._top_layer,
            node_count=len(self._graph),
            nodes_per_layer=nodes_per_layer,
            edges_per_layer=edges_per_layer,
            nodes=nodes_out,
            edges=edges_out,
        )

    def __len__(self) -> int:
        return len(self._graph)
