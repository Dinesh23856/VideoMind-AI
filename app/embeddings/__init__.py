"""Embedding abstraction module."""
from app.embeddings.provider import (
    EmbeddingProvider,
    LocalDenseEmbeddingProvider,
    RemoteEmbeddingProvider,
)

__all__ = [
    "EmbeddingProvider",
    "LocalDenseEmbeddingProvider",
    "RemoteEmbeddingProvider",
]
