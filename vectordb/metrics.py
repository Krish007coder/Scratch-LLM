"""
Distance metric functions — pure Python, no external dependencies.
Matches the C++ implementations in main.cpp exactly.
"""
import math
from typing import Callable, List

Vector = List[float]
DistFn = Callable[[Vector, Vector], float]


def euclidean(a: Vector, b: Vector) -> float:
    """Straight-line distance between two vectors."""
    s = 0.0
    for x, y in zip(a, b):
        d = x - y
        s += d * d
    return math.sqrt(s)


def cosine(a: Vector, b: Vector) -> float:
    """Cosine distance (1 - cosine similarity). Range [0, 2]."""
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return 1.0 - dot / (math.sqrt(na) * math.sqrt(nb))


def manhattan(a: Vector, b: Vector) -> float:
    """Sum of absolute differences (L1 norm)."""
    return sum(abs(x - y) for x, y in zip(a, b))


def get_dist_fn(metric: str) -> DistFn:
    """Return the distance function for the given metric name."""
    if metric == "cosine":
        return cosine
    if metric == "manhattan":
        return manhattan
    return euclidean
