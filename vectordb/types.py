"""
Shared data types for the VectorDB package.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class VectorItem:
    """A single vector entry stored in the demo 16D index."""
    id: int
    metadata: str
    category: str
    emb: List[float]


@dataclass
class DocItem:
    """A document chunk stored in the DocumentDB (real Ollama embeddings)."""
    id: int
    title: str
    text: str
    emb: List[float]
