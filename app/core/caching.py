"""
Intelligent Caching Layer 2.0: Multi-tier semantic and query caching with model/version metadata isolation.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, Optional


class IntelligentCache:
    """
    In-memory and TTL-aware cache for media metadata, embeddings, retrieval, and LLM outputs.
    Guarantees cache key isolation by including model version, provider, and parameters.
    """

    def __init__(self, default_ttl_seconds: int = 3600):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl_seconds

    def make_key(
        self,
        namespace: str,
        content: str,
        provider: str = "default",
        model: str = "v1",
        extra: Optional[str] = None,
    ) -> str:
        raw = f"{namespace}:{provider}:{model}:{content}:{extra or ''}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        entry = self._cache.get(key)
        if not entry:
            return None

        # Check expiration
        if time.time() > entry["expires_at"]:
            del self._cache[key]
            return None

        return entry["value"]

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        self._cache[key] = {
            "value": value,
            "expires_at": time.time() + ttl,
            "created_at": time.time(),
        }

    def clear(self) -> None:
        self._cache.clear()

    def stats(self) -> Dict[str, Any]:
        valid_entries = sum(1 for e in self._cache.values() if e["expires_at"] > time.time())
        return {
            "total_keys": len(self._cache),
            "valid_keys": valid_entries,
        }
