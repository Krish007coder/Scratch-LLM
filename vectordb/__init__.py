"""
VectorDB package — exports all public classes.
"""
from .bruteforce import BruteForce
from .documentdb import DocumentDB
from .hnsw import HNSW, GraphInfo
from .kdtree import KDTree
from .metrics import cosine, euclidean, get_dist_fn, manhattan
from .ollama_client import OllamaClient
from .types import DocItem, VectorItem
from .vectordb import VectorDB

__all__ = [
    "BruteForce",
    "KDTree",
    "HNSW",
    "GraphInfo",
    "VectorDB",
    "DocumentDB",
    "OllamaClient",
    "VectorItem",
    "DocItem",
    "euclidean",
    "cosine",
    "manhattan",
    "get_dist_fn",
]
