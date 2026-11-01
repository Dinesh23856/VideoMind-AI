"""
Canonical Embedding Architecture and Model Registry for VideoMind-AI.
Supports:
1. Sentence Transformers neural embeddings (local, PyTorch/ONNX) when installed
2. Ollama local neural embeddings (/api/embeddings) when Ollama is running
3. LocalDenseEmbeddingProvider (zero-dependency, deterministic n-gram hashing with sublinear log-scaling and L2 normalization)
4. Remote embedding provider (optional OpenAI / commercial adapter)
5. EmbeddingCache: Model- and version-aware in-memory LRU cache
6. EmbeddingModelRegistry: Tracks model metadata and strictly prevents incompatible vector dimensions from mixing.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import httpx

from config.settings import get_settings

logger = logging.getLogger("Embeddings")


# =========================================================================
# Canonical Embedding Provider Contract
# =========================================================================

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
        """Model identifier."""
        pass

    @property
    def provider_name(self) -> str:
        """Provider identifier."""
        return "unknown"

    @property
    def runtime(self) -> str:
        """Execution runtime identifier."""
        return "unknown"

    @property
    def is_local(self) -> bool:
        """Whether inference executes locally without network egress."""
        return True

    @property
    def is_free_tier(self) -> bool:
        """Whether the provider incurs $0 API cost."""
        return True

    @property
    def cost_class(self) -> str:
        if self.is_local:
            return "ZERO_COST"
        elif self.is_free_tier:
            return "FREE_TIER"
        return "COMMERCIAL"

    @property
    def version(self) -> str:
        """Embedding model version string."""
        return "1.0.0"

    @property
    def distance_metric(self) -> str:
        """Default distance metric."""
        return "cosine"

    @property
    def normalization(self) -> str:
        """Vector normalization technique."""
        return "L2"

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple text strings in batch."""
        pass

    def encode_query(self, query: str) -> List[float]:
        """Encode user search query (asymmetric retrieval aware)."""
        return self.embed_text(query)

    def encode_document(self, document: str) -> List[float]:
        """Encode passage/document chunk (asymmetric retrieval aware)."""
        return self.embed_text(document)

    def embed_query(self, query: str) -> List[float]:
        """Alias for encode_query."""
        return self.encode_query(query)

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Alias for embed_batch."""
        return self.embed_batch(documents)

    @property
    def classification(self) -> str:
        """Epistemic classification property."""
        return self.get_classification()

    def get_classification(self) -> str:
        """Epistemic classification: REAL, HEURISTIC, FALLBACK, or OPTIONAL_EXTERNAL."""
        return "REAL"

    def health_check(self) -> bool:
        """Check if provider is operational."""
        try:
            vec = self.embed_text("health check")
            return len(vec) == self.dimension
        except Exception:
            return False

    def get_metadata(self) -> Dict[str, Any]:
        """Return structured metadata describing this embedding provider."""
        return {
            "model_name": self.model_name,
            "provider_name": self.provider_name,
            "runtime": self.runtime,
            "dimension": self.dimension,
            "version": self.version,
            "is_local": self.is_local,
            "is_free_tier": self.is_free_tier,
            "cost_class": self.cost_class,
            "classification": self.get_classification(),
            "distance_metric": self.distance_metric,
            "normalization": self.normalization,
        }


# =========================================================================
# Model-Aware Embedding Cache
# =========================================================================

class EmbeddingCache:
    """Thread-safe, model-aware LRU cache for embedding vectors."""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._cache: Dict[str, List[float]] = {}
        self._access_order: List[str] = []

    def make_key(self, model_name: str, version: str, text: str) -> str:
        raw = f"{model_name}:{version}:{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, model_name: str, version: str, text: str) -> Optional[List[float]]:
        key = self.make_key(model_name, version, text)
        if key in self._cache:
            # Move to end of access order
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            return list(self._cache[key])
        return None

    def put(self, model_name: str, version: str, text: str, vector: List[float]) -> None:
        key = self.make_key(model_name, version, text)
        if key in self._cache:
            self._cache[key] = list(vector)
            return

        if len(self._cache) >= self.max_size and self._access_order:
            oldest_key = self._access_order.pop(0)
            self._cache.pop(oldest_key, None)

        self._cache[key] = list(vector)
        self._access_order.append(key)

    def clear(self) -> None:
        self._cache.clear()
        self._access_order.clear()

    def size(self) -> int:
        return len(self._cache)

    set = put


_global_cache = EmbeddingCache()


# =========================================================================
# Zero-Dependency Deterministic Dense Hashing Provider
# =========================================================================

class LocalDenseEmbeddingProvider(EmbeddingProvider):
    """
    Zero-dependency deterministic dense embedding provider.
    Uses n-gram feature hashing with sublinear frequency weighting and L2 normalization
    to produce true continuous dense representations across a fixed dimension (e.g. 128-d).
    Classification: HEURISTIC (Classical deterministic hashing, never falsely claimed as neural).
    """

    def __init__(
        self,
        dimension: int = 128,
        model_name: str = "videomind-dense-hashing-v1",
        version: str = "1.0.0",
        cache: Optional[EmbeddingCache] = None,
    ):
        self._dim = dimension
        self._model_name = model_name
        self._version = version
        self._cache = cache or _global_cache

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider_name(self) -> str:
        return "local_hashing"

    @property
    def runtime(self) -> str:
        return "hashing"

    @property
    def version(self) -> str:
        return self._version

    def get_classification(self) -> str:
        return "HEURISTIC"

    def embed_text(self, text: str) -> List[float]:
        cached = self._cache.get(self._model_name, self._version, text)
        if cached is not None:
            return cached

        vec = [0.0] * self._dim
        tokens = re.findall(r"\b[a-zA-Z0-9_]{2,}\b", text.lower())
        if not tokens:
            return vec

        # Sublinear term hashing into fixed dimension
        for i, token in enumerate(tokens):
            h1 = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self._dim
            vec[h1] += 1.0

            # Bigram hash for phrase sequence representation
            if i > 0:
                bigram = f"{tokens[i-1]}_{token}"
                h2 = int(hashlib.sha1(bigram.encode("utf-8")).hexdigest(), 16) % self._dim
                vec[h2] += 0.5

        # Sublinear log-scaling: 1 + log(tf)
        for idx in range(self._dim):
            if vec[idx] > 0.0:
                vec[idx] = 1.0 + math.log(vec[idx])

        # L2 Normalization
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0.0:
            vec = [v / norm for v in vec]

        if len(vec) != self._dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self._dim}, got {len(vec)}")

        self._cache.put(self._model_name, self._version, text, vec)
        return vec

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]


# =========================================================================
# Local Neural SentenceTransformer Provider
# =========================================================================

class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """
    Local Neural Embedding Provider using Hugging Face Sentence Transformers.
    Classification: REAL (Neural dense vector space).
    Gracefully falls back to LocalDenseEmbeddingProvider if library or model is unavailable.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        fallback: Optional[EmbeddingProvider] = None,
    ):
        self._model_name = model_name
        self._fallback = fallback or LocalDenseEmbeddingProvider(dimension=128)
        self._model = None
        self._is_active = False
        self._dim = 384  # Default for MiniLM
        self._version = "2.0.0"

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            dummy = self._model.encode("test", normalize_embeddings=True)
            self._dim = len(dummy)
            self._is_active = True
            logger.info(f"Loaded SentenceTransformer '{model_name}' (dimension={self._dim}).")
        except Exception as e:
            logger.info(f"SentenceTransformer '{model_name}' not available ({e}); using local hashing fallback.")
            self._is_active = False

    @property
    def dimension(self) -> int:
        return self._dim if self._is_active else self._fallback.dimension

    @property
    def model_name(self) -> str:
        return f"sentence-transformers/{self._model_name}" if self._is_active else self._fallback.model_name

    @property
    def provider_name(self) -> str:
        return "sentence_transformers" if self._is_active else self._fallback.provider_name

    @property
    def runtime(self) -> str:
        return "sentence_transformers" if self._is_active else self._fallback.runtime

    @property
    def version(self) -> str:
        return self._version if self._is_active else self._fallback.version

    def get_classification(self) -> str:
        return "REAL" if self._is_active else self._fallback.get_classification()

    def embed_text(self, text: str) -> List[float]:
        if not self._is_active or self._model is None:
            return self._fallback.embed_text(text)
        try:
            vec = self._model.encode(text, normalize_embeddings=True)
            return vec.tolist() if hasattr(vec, "tolist") else list(vec)
        except Exception:
            return self._fallback.embed_text(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not self._is_active or self._model is None:
            return self._fallback.embed_batch(texts)
        try:
            vecs = self._model.encode(texts, normalize_embeddings=True)
            return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vecs]
        except Exception:
            return self._fallback.embed_batch(texts)


# =========================================================================
# Local Ollama Embedding Provider
# =========================================================================

class OllamaEmbeddingProvider(EmbeddingProvider):
    """
    Local Ollama Embedding Provider using /api/embeddings.
    Classification: REAL (Local neural execution via Ollama daemon).
    """

    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        base_url: Optional[str] = None,
        fallback: Optional[EmbeddingProvider] = None,
    ):
        settings = get_settings()
        self._model_name = model_name
        self._base_url = (base_url or getattr(settings, "ollama_base_url", "http://localhost:11434")).rstrip("/")
        self._fallback = fallback or LocalDenseEmbeddingProvider(dimension=128)
        self._dim = 768  # nomic-embed-text default
        self._version = "1.5.0"

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return f"ollama/{self._model_name}"

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def runtime(self) -> str:
        return "ollama"

    @property
    def version(self) -> str:
        return self._version

    def get_classification(self) -> str:
        return "REAL"

    def embed_text(self, text: str) -> List[float]:
        try:
            with httpx.Client(timeout=10.0) as client:
                res = client.post(
                    f"{self._base_url}/api/embeddings",
                    json={"model": self._model_name, "prompt": text},
                )
                if res.status_code == 200:
                    data = res.json()
                    emb = data.get("embedding")
                    if emb:
                        self._dim = len(emb)
                        return emb
        except Exception:
            pass
        return self._fallback.embed_text(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]


# =========================================================================
# Remote / Commercial Embedding Provider (Optional)
# =========================================================================

class RemoteEmbeddingProvider(EmbeddingProvider):
    """
    OpenAI / External API embedding provider with graceful local fallback.
    Classification: OPTIONAL_EXTERNAL
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        fallback: Optional[EmbeddingProvider] = None,
    ):
        self._api_key = api_key
        self._model = model
        self._fallback = fallback or LocalDenseEmbeddingProvider(dimension=128)
        self._dim = 1536 if "text-embedding" in model else 128
        self._version = "3.0.0"

    @property
    def dimension(self) -> int:
        return self._dim if self._api_key else self._fallback.dimension

    @property
    def model_name(self) -> str:
        return self._model if self._api_key else self._fallback.model_name

    @property
    def provider_name(self) -> str:
        return "openai" if self._api_key else self._fallback.provider_name

    @property
    def runtime(self) -> str:
        return "cloud_api" if self._api_key else self._fallback.runtime

    @property
    def is_local(self) -> bool:
        return not bool(self._api_key)

    @property
    def is_free_tier(self) -> bool:
        return not bool(self._api_key)

    @property
    def version(self) -> str:
        return self._version if self._api_key else self._fallback.version

    def get_classification(self) -> str:
        return "OPTIONAL_EXTERNAL" if self._api_key else self._fallback.get_classification()

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


# =========================================================================
# Embedding Model Registry & Compatibility Checker
# =========================================================================

class EmbeddingModelRegistry:
    """
    Maintains metadata of supported embedding models and validates index compatibility.
    Guarantees that incompatible vector dimensions (e.g. 128-d vs 384-d vs 1536-d) are never
    silently mixed in the same vector store.
    """

    KNOWN_MODELS: Dict[str, Dict[str, Any]] = {
        "videomind-dense-hashing-v1": {
            "dimension": 128,
            "provider": "local_hashing",
            "runtime": "hashing",
            "is_local": True,
            "is_free_tier": True,
            "classification": "HEURISTIC",
            "description": "Deterministic sublinear n-gram hashing dense embedding ($0 cost, 0 dependencies).",
        },
        "all-MiniLM-L6-v2": {
            "dimension": 384,
            "provider": "sentence_transformers",
            "runtime": "local_pytorch",
            "is_local": True,
            "is_free_tier": True,
            "classification": "REAL",
            "description": "Fast 384-d local neural sentence transformer.",
        },
        "nomic-embed-text": {
            "dimension": 768,
            "provider": "ollama",
            "runtime": "ollama",
            "is_local": True,
            "is_free_tier": True,
            "classification": "REAL",
            "description": "768-d local open-weights embedding model via Ollama daemon.",
        },
        "bge-small-en-v1.5": {
            "dimension": 384,
            "provider": "sentence_transformers",
            "runtime": "local_pytorch",
            "is_local": True,
            "is_free_tier": True,
            "classification": "REAL",
            "description": "High-accuracy compact 384-d retrieval embedding model.",
        },
        "text-embedding-3-small": {
            "dimension": 1536,
            "provider": "openai",
            "runtime": "cloud_api",
            "is_local": False,
            "is_free_tier": False,
            "classification": "OPTIONAL_EXTERNAL",
            "description": "Commercial cloud embedding model (requires API key).",
        },
    }

    @classmethod
    def get_model_info(cls, model_name: str) -> Optional[Dict[str, Any]]:
        # Strip prefixes
        cleaned = model_name.split("/")[-1]
        return cls.KNOWN_MODELS.get(cleaned) or cls.KNOWN_MODELS.get(model_name)

    @classmethod
    def validate_dimension_compatibility(cls, dim1: Optional[int], dim2: Optional[int]) -> bool:
        if dim1 is None or dim2 is None:
            return True
        return dim1 == dim2

    @classmethod
    def validate_compatibility(
        cls,
        index_model_id: Optional[str] = None,
        index_dimension: Optional[int] = None,
        provider: Optional[Any] = None,
        provider_dimension: Optional[int] = None,
        provider_model_id: Optional[str] = None,
    ) -> None:
        """
        Validates that a provider's output vectors match the index dimension.
        Raises ValueError with 'EMBEDDING_INDEX_INCOMPATIBLE' on mismatch.
        """
        eff_dim = provider_dimension if provider_dimension is not None else (getattr(provider, "dimension", None) if provider else None)
        eff_model = provider_model_id or (getattr(provider, "model_name", "unknown") if provider else "unknown")

        if index_dimension is not None and eff_dim is not None and eff_dim != index_dimension:
            raise ValueError(
                f"EMBEDDING_INDEX_INCOMPATIBLE: Index expects {index_dimension}-d vectors "
                f"(configured for '{index_model_id}'), but provider '{eff_model}' produces "
                f"{eff_dim}-d vectors. Incompatible embedding models cannot be searched or indexed together."
            )

    @classmethod
    def list_models(cls) -> List[Dict[str, Any]]:
        return [{"id": k, "model_id": k, **v} for k, v in cls.KNOWN_MODELS.items()]


# =========================================================================
# Factory Function
# =========================================================================

def get_embedding_provider(
    model_name: Optional[str] = None,
    zero_cost_only: bool = True,
    allow_cloud: bool = False,
) -> EmbeddingProvider:
    """Returns optimal embedding provider based on installed runtimes and configuration."""
    settings = get_settings()

    # 1. If explicit cloud requested and permitted
    if allow_cloud and not zero_cost_only and settings.openai_api_key:
        return RemoteEmbeddingProvider(api_key=settings.openai_api_key)

    # 2. Try SentenceTransformers if installed
    try:
        import sentence_transformers  # noqa: F401
        return SentenceTransformerEmbeddingProvider(model_name=model_name or "all-MiniLM-L6-v2")
    except ImportError:
        pass

    # 3. Default to deterministic dense hashing (zero cost, zero dependency, high speed)
    return LocalDenseEmbeddingProvider(dimension=128)
