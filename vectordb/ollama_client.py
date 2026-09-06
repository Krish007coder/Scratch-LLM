"""
OllamaClient — HTTP wrapper around the local Ollama REST API.

Install Ollama: https://ollama.com
Pull models:
    ollama pull nomic-embed-text
    ollama pull llama3.2
"""
from __future__ import annotations

from typing import List, Optional

import requests


class OllamaClient:
    """Thin HTTP client for the Ollama local LLM server."""

    embed_model: str = "nomic-embed-text"
    gen_model: str = "llama3.2"

    def __init__(self, host: str = "127.0.0.1", port: int = 11434) -> None:
        self._base = f"http://{host}:{port}"

    # ------------------------------------------------------------------ public

    def is_available(self) -> bool:
        """Return True if the Ollama server is reachable."""
        try:
            r = requests.get(f"{self._base}/api/tags", timeout=2)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def embed(self, text: str) -> List[float]:
        """
        Embed `text` using the configured embedding model.
        Returns an empty list if Ollama is unavailable or the model is missing.
        """
        try:
            r = requests.post(
                f"{self._base}/api/embeddings",
                json={"model": self.embed_model, "prompt": text},
                timeout=33,  # 3s connect + 30s read
            )
            if r.status_code != 200:
                return []
            data = r.json()
            return data.get("embedding", [])
        except requests.RequestException:
            return []

    def generate(self, prompt: str) -> str:
        """
        Generate a response from the configured generation model.
        Returns an error string if Ollama is unavailable.
        """
        try:
            r = requests.post(
                f"{self._base}/api/generate",
                json={"model": self.gen_model, "prompt": prompt, "stream": False},
                timeout=183,  # 3s connect + 180s read (LLMs can be slow)
            )
            if r.status_code != 200:
                return "ERROR: Ollama unavailable. Run: ollama serve"
            return r.json().get("response", "")
        except requests.RequestException:
            return "ERROR: Ollama unavailable. Run: ollama serve"
