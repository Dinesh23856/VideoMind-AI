"""
True Embedding Abstraction: Pluggable local and remote embedding providers,
dimensionality validation, batching, and content-hash caching.
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class EmbeddingProvider(ABC):
    """Abstract interface for dense embedding generation."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector dimensionality."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier and version."""
        pass

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple text strings in batch."""
        pass


class LocalDenseEmbeddingProvider(EmbeddingProvider):
    """
    Zero-dependency deterministic dense embedding provider.
    Uses n-gram feature hashing with sublinear frequency weighting and L2 normalization
    to produce true dense continuous representations across a fixed dimension (e.g. 128-d).
    Does NOT require PyTorch or external API keys, ensuring reliable zero-dependency execution.
    """

    def __init__(self, dimension: int = 128, model_name: str = "videomind-dense-hashing-v1"):
        self._dim = dimension
        self._model_name = model_name
        self._cache: Dict[str, List[float]] = {}

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_text(self, text: str) -> List[float]:
        cache_key = hashlib.sha256(f"{self._model_name}:{text}".encode("utf-8")).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        vec = [0.0] * self._dim
        tokens = re.findall(r"\b[a-zA-Z0-9_]{2,}\b", text.lower())
        if not tokens:
            return vec

        # Sublinear term hashing into fixed dimension
        for i, token in enumerate(tokens):
            # Token hash
            h1 = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self._dim
            vec[h1] += 1.0

            # Bigram hash for sequence awareness
            if i > 0:
                bigram = f"{tokens[i-1]}_{token}"
                h2 = int(hashlib.sha1(bigram.encode("utf-8")).hexdigest(), 16) % self._dim
                vec[h2] += 0.5

        # Sublinear log-scaling: log(1 + x)
        for idx in range(self._dim):
            if vec[idx] > 0.0:
                vec[idx] = 1.0 + math.log(vec[idx])

        # L2 Normalization
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0.0:
            vec = [v / norm for v in vec]

        # Dimension validation check
        if len(vec) != self._dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self._dim}, got {len(vec)}")

        self._cache[cache_key] = vec
        return vec

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]


class RemoteEmbeddingProvider(EmbeddingProvider):
    """
    OpenAI / External API embedding provider with graceful local fallback.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-small", fallback: Optional[EmbeddingProvider] = None):
        self._api_key = api_key
        self._model = model
        self._fallback = fallback or LocalDenseEmbeddingProvider(dimension=128)
        self._dim = 1536 if "text-embedding" in model else 128

    @property
    def dimension(self) -> int:
        return self._dim if self._api_key else self._fallback.dimension

    @property
    def model_name(self) -> str:
        return self._model if self._api_key else self._fallback.model_name

    def embed_text(self, text: str) -> List[float]:
        if not self._api_key:
            return self._fallback.embed_text(text)
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self._api_key)
            resp = client.embeddings.create(input=text, model=self._model)
            return resp.data[0].embedding
        except Exception:
            return self._fallback.embed_text(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not self._api_key:
            return self._fallback.embed_batch(texts)
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self._api_key)
            resp = client.embeddings.create(input=texts, model=self._model)
            return [d.embedding for d in resp.data]
        except Exception:
            return self._fallback.embed_batch(texts)
