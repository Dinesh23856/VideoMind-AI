"""Vector store abstraction, high-performance in-memory vector index, and ChromaDB provider."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text as sa_text

from app.database.models import VectorEmbeddingRecord
from app.database.session import SessionLocal
from app.embeddings.provider import EmbeddingProvider, LocalDenseEmbeddingProvider

logger = logging.getLogger("VectorStore")


class BaseVectorStore(ABC):
    @abstractmethod
    def add_texts(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        pass

    def delete(self, ids: List[str]) -> int:
        """Delete documents by their IDs. Return number of deleted items."""
        return 0

    def delete_by_filter(self, filters: Dict[str, Any]) -> int:
        """Delete documents matching metadata filter. Return number of deleted items."""
        return 0

    def get_index_metadata(self) -> Dict[str, Any]:
        """Return index metadata (model, dimension, version, document count)."""
        return {}


VectorStoreProvider = BaseVectorStore


class InMemoryVectorStore(BaseVectorStore):
    """
    Deterministic In-Memory Vector Store with Multi-Tenancy & Index Safety.

    Default Mode: Classical Vector Space Model (Salton, 1975) with sublinear TF-IDF term weighting
    and cosine similarity. Note: This default is an exact Information Retrieval Vector Space Model,
    NOT neural embeddings (conforming to truthful epistemic standards).

    Dense Mode: When initialized with an EmbeddingProvider (e.g. LocalDenseEmbeddingProvider or RemoteEmbeddingProvider),
    indexes continuous dense vectors and performs dense cosine similarity search.
    """

    STOPWORDS = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
        "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
        "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
        "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
        "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
        "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself",
        "let's", "me", "more", "most", "my", "myself", "no", "nor", "not", "of", "off", "on",
        "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over",
        "own", "same", "she", "should", "so", "some", "such", "than", "that", "the", "their",
        "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those",
        "through", "to", "too", "under", "until", "up", "very", "was", "we", "were", "what",
        "when", "where", "which", "while", "who", "whom", "why", "with", "would", "you",
        "your", "yours", "yourself", "yourselves", "using", "uses"
    }

    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        index_model_id: Optional[str] = None,
        index_dimension: Optional[int] = None,
        index_version: str = "1.0.0",
    ):
        self.embedding_provider = embedding_provider
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.idf: Dict[str, float] = {}
        self.doc_count: int = 0
        self.index_dimension: Optional[int] = (
            index_dimension
            if index_dimension is not None
            else (getattr(embedding_provider, "dimension", None) if embedding_provider else None)
        )
        self.index_model_id: Optional[str] = (
            index_model_id
            or (getattr(embedding_provider, "model_name", None) if embedding_provider else "tf-idf-sparse")
        )
        self.index_version: str = index_version

    def get_index_metadata(self) -> Dict[str, Any]:
        """Return index metadata for inspection and compatibility tracking."""
        return {
            "index_model_id": self.index_model_id,
            "index_dimension": self.index_dimension,
            "index_version": self.index_version,
            "doc_count": len(self.documents),
            "is_dense": self.embedding_provider is not None,
            "store_type": "in_memory",
        }

    def _tokenize(self, text: str) -> List[str]:
        words = [w.lower() for w in re.findall(r"\b[a-zA-Z0-9_]{2,}\b", text)]
        filtered = [w for w in words if w not in self.STOPWORDS]
        return filtered if filtered else words

    def _compute_tf(self, tokens: List[str]) -> Dict[str, float]:
        tf: Dict[str, float] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0.0) + 1.0
        # Sublinear scaling: 1 + log(tf)
        return {t: 1.0 + math.log(cnt) for t, cnt in tf.items()}

    def _recalculate_idf(self):
        df: Dict[str, int] = {}
        for item in self.documents.values():
            tokens = set(self._tokenize(item["text"]))
            for t in tokens:
                df[t] = df.get(t, 0) + 1
        n = max(len(self.documents), 1)
        self.idf = {t: math.log(1.0 + (n / count)) for t, count in df.items()}
        self.doc_count = len(self.documents)

    def add_texts(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        generated_ids: List[str] = []
        dense_vectors = (
            self.embedding_provider.embed_batch(texts)
            if self.embedding_provider
            else None
        )

        if dense_vectors:
            vec_dim = len(dense_vectors[0])
            if self.index_dimension is not None and vec_dim != self.index_dimension:
                raise ValueError(
                    f"EMBEDDING_INDEX_INCOMPATIBLE: Incompatible embedding dimension: index expects {self.index_dimension}-d vectors, "
                    f"but received {vec_dim}-d vectors from provider '{getattr(self.embedding_provider, 'model_name', 'unknown')}'. "
                    f"Cannot mix incompatible embedding models in vector index."
                )
            if self.index_dimension is None:
                self.index_dimension = vec_dim
            if self.embedding_provider and self.index_model_id not in (None, "tf-idf-sparse"):
                prov_model = getattr(self.embedding_provider, "model_name", None)
                if prov_model and prov_model != self.index_model_id:
                    raise ValueError(
                        f"EMBEDDING_INDEX_INCOMPATIBLE: Incompatible embedding model: index configured for '{self.index_model_id}', "
                        f"but received vectors from '{prov_model}'."
                    )

        for i, text in enumerate(texts):
            doc_id = ids[i] if ids and i < len(ids) else f"vec_{len(self.documents) + 1:05d}"
            meta = dict(metadatas[i]) if i < len(metadatas) else {}
            tokens = self._tokenize(text)
            tf = self._compute_tf(tokens)
            doc_entry: Dict[str, Any] = {
                "id": doc_id,
                "text": text,
                "metadata": meta,
                "tf": tf,
            }
            if dense_vectors is not None:
                doc_entry["vector"] = dense_vectors[i]
            self.documents[doc_id] = doc_entry
            generated_ids.append(doc_id)

        self._recalculate_idf()
        return generated_ids

    def search(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        if not query.strip() or not self.documents:
            return []

        # Build effective multi-tenant filters
        effective_filters = dict(filters) if filters else {}
        if tenant_id is not None:
            effective_filters["tenant_id"] = tenant_id
        if workspace_id is not None:
            effective_filters["workspace_id"] = workspace_id
        if user_id is not None:
            effective_filters["user_id"] = user_id

        def matches_filters(doc_meta: Dict[str, Any]) -> bool:
            if not effective_filters:
                return True
            for fk, fv in effective_filters.items():
                if doc_meta.get(fk) != fv:
                    return False
            return True

        # Dense Vector Search Branch
        if self.embedding_provider is not None:
            prov_dim = getattr(self.embedding_provider, "dimension", None)
            if self.index_dimension is not None and prov_dim is not None and prov_dim != self.index_dimension:
                raise ValueError(
                    f"EMBEDDING_INDEX_INCOMPATIBLE: Query vector dimension mismatch: provider produces {prov_dim}-d vectors, "
                    f"but index is built with {self.index_dimension}-d vectors."
                )
            q_vec = self.embedding_provider.embed_text(query)
            if self.index_dimension is not None and len(q_vec) != self.index_dimension:
                raise ValueError(
                    f"EMBEDDING_INDEX_INCOMPATIBLE: Query vector dimension mismatch: query vector is {len(q_vec)}-d, "
                    f"but index expects {self.index_dimension}-d."
                )
            q_norm = math.sqrt(sum(v * v for v in q_vec)) or 1.0
            scores: List[Tuple[Dict[str, Any], float]] = []
            for doc in self.documents.values():
                if not matches_filters(doc["metadata"]):
                    continue
                d_vec = doc.get("vector")
                if not d_vec:
                    continue
                dot = sum(a * b for a, b in zip(q_vec, d_vec))
                d_norm = math.sqrt(sum(v * v for v in d_vec)) or 1.0
                cosine = dot / (q_norm * d_norm)
                if cosine > 0.15:  # Separates semantic match from background
                    scores.append((doc["metadata"], round(cosine, 4)))
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:k]

        # Classical Sparse TF-IDF Vector Space Model Branch
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return []

        q_tf = self._compute_tf(q_tokens)
        q_vec_sparse = {t: val * self.idf.get(t, 1.0) for t, val in q_tf.items()}
        q_norm_sparse = math.sqrt(sum(v * v for v in q_vec_sparse.values())) or 1.0

        sparse_scores: List[Tuple[Dict[str, Any], float]] = []
        for doc in self.documents.values():
            if not matches_filters(doc["metadata"]):
                continue

            dot = 0.0
            doc_tf = doc["tf"]
            for t, q_val in q_vec_sparse.items():
                if t in doc_tf:
                    doc_val = doc_tf[t] * self.idf.get(t, 1.0)
                    dot += q_val * doc_val

            if dot == 0.0:
                continue

            doc_norm = math.sqrt(sum((v * self.idf.get(t, 1.0)) ** 2 for t, v in doc_tf.items())) or 1.0
            cosine = dot / (q_norm_sparse * doc_norm)
            if cosine > 0.12:
                sparse_scores.append((doc["metadata"], round(cosine, 4)))

        sparse_scores.sort(key=lambda x: x[1], reverse=True)
        return sparse_scores[:k]

    def delete(self, ids: List[str]) -> int:
        """Delete specific documents by ID."""
        deleted = 0
        for doc_id in ids:
            if doc_id in self.documents:
                del self.documents[doc_id]
                deleted += 1
        if deleted > 0:
            self._recalculate_idf()
        return deleted

    def delete_by_filter(self, filters: Dict[str, Any]) -> int:
        """Delete documents matching metadata filter."""
        to_delete = []
        for doc_id, doc in self.documents.items():
            if all(doc["metadata"].get(k) == v for k, v in filters.items()):
                to_delete.append(doc_id)
        return self.delete(to_delete)


class ChromaVectorStore(BaseVectorStore):
    """
    Adapter for ChromaDB vector store with graceful optional dependency handling.
    Truthfully reports 'is_active': False and classification: 'FALLBACK' or 'NOT_AVAILABLE'
    when chromadb is not installed.
    """

    def __init__(
        self,
        collection_name: str = "videomind_collection",
        embedding_provider: Optional[EmbeddingProvider] = None,
    ):
        self.collection_name = collection_name
        self.embedding_provider = embedding_provider or LocalDenseEmbeddingProvider()
        self.is_active = False
        self._client = None
        self._collection = None

        try:
            import chromadb  # type: ignore
            self._client = chromadb.Client()
            self._collection = self._client.get_or_create_collection(name=collection_name)
            self.is_active = True
        except Exception:
            self.is_active = False

        self.classification = "REAL" if self.is_active else "NOT_AVAILABLE"

    def get_index_metadata(self) -> Dict[str, Any]:
        return {
            "store_type": "chroma",
            "collection_name": self.collection_name,
            "is_active": self.is_active,
            "classification": self.classification,
        }

    def add_texts(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        if not self.is_active or not self._collection:
            return []
        ids = ids or [f"chroma_{uuid.uuid4().hex[:12]}" for _ in texts]
        self._collection.add(documents=texts, metadatas=metadatas, ids=ids)
        return ids

    def search(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        if not self.is_active or not self._collection:
            return []
        res = self._collection.query(query_texts=[query], n_results=k)
        hits = []
        if res and res.get("metadatas") and res["metadatas"][0]:
            for meta, dist in zip(res["metadatas"][0], res.get("distances", [[]])[0]):
                hits.append((meta, float(1.0 - dist)))
        return hits


class PersistentSQLiteVectorStore(BaseVectorStore):
    """
    Persistent SQLite-backed vector store with optional sqlite-vec SIMD acceleration.
    Persists dense vectors, text, parent-child relations, and metadata across restarts and worker processes.
    Classification: REAL (when sqlite-vec is available) or FALLBACK (persistent SQLite with optimized cosine math).
    """

    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        index_model_id: Optional[str] = None,
        index_dimension: Optional[int] = None,
        index_version: str = "1.0.0",
    ):
        self.embedding_provider = embedding_provider or LocalDenseEmbeddingProvider()
        self.index_dimension = (
            index_dimension
            if index_dimension is not None
            else getattr(self.embedding_provider, "dimension", 128)
        )
        self.index_model_id = (
            index_model_id
            or getattr(self.embedding_provider, "model_name", "videomind-dense-hashing-v1")
        )
        self.index_version = index_version
        self.is_sqlite_vec = False

        # Attempt to detect and test sqlite-vec extension
        try:
            import sqlite_vec  # type: ignore
            with SessionLocal() as db:
                db.execute(sa_text("SELECT vec_version()")).scalar()
            self.is_sqlite_vec = True
        except Exception as e:
            logger.debug(f"sqlite-vec not available or failed to initialize: {e}")
            self.is_sqlite_vec = False

        self.classification = "REAL" if self.is_sqlite_vec else "FALLBACK"
        try:
            from app.database.session import init_db
            init_db()
        except Exception:
            pass

        if self.is_sqlite_vec and self.index_dimension:
            self._ensure_sqlite_vec_table()

    def _ensure_sqlite_vec_table(self, db_session=None) -> None:
        """Ensures the sqlite-vec virtual table for current index dimension exists."""
        if not self.is_sqlite_vec or not self.index_dimension:
            return

        table_name = f"vec_items_{self.index_dimension}"
        create_sql = (
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {table_name} "
            f"USING vec0(id text primary key, embedding float[{self.index_dimension}] distance_metric=cosine)"
        )
        if db_session:
            try:
                db_session.execute(sa_text(create_sql))
            except Exception as e:
                logger.warning(f"Failed to create sqlite-vec virtual table {table_name}: {e}")
        else:
            try:
                with SessionLocal() as db:
                    db.execute(sa_text(create_sql))
                    db.commit()
            except Exception as e:
                logger.warning(f"Failed to create sqlite-vec virtual table {table_name}: {e}")

    def get_index_metadata(self) -> Dict[str, Any]:
        doc_count = 0
        try:
            with SessionLocal() as db:
                doc_count = db.query(VectorEmbeddingRecord).count()
        except Exception:
            pass

        return {
            "store_type": "sqlite_persistent",
            "is_sqlite_vec": self.is_sqlite_vec,
            "classification": self.classification,
            "doc_count": doc_count,
            "index_dimension": self.index_dimension,
            "index_model_id": self.index_model_id,
            "index_version": self.index_version,
            "is_dense": True,
        }

    def index_documents(self, docs: List[Any]) -> List[str]:
        """Index a list of MultiVectorDocument instances."""
        texts = [getattr(d, "content", getattr(d, "text", "")) for d in docs]
        metadatas = []
        ids = [d.id for d in docs]
        for d in docs:
            m = dict(getattr(d, "metadata", {}))
            if hasattr(d, "tenant_id"):
                m["tenant_id"] = d.tenant_id
            if hasattr(d, "workspace_id"):
                m["workspace_id"] = d.workspace_id
            if getattr(d, "user_id", None):
                m["user_id"] = d.user_id
            if getattr(d, "source_id", None):
                m["source_id"] = d.source_id
            if getattr(d, "parent_id", None):
                m["parent_id"] = d.parent_id
            if getattr(d, "parent_content", None):
                m["parent_content"] = d.parent_content
            if getattr(d, "content_hash", None):
                m["content_hash"] = d.content_hash
            elif hasattr(d, "compute_content_hash"):
                m["content_hash"] = d.compute_content_hash()
            if getattr(d, "start_ms", None) is not None:
                m["start_ms"] = d.start_ms
            if getattr(d, "end_ms", None) is not None:
                m["end_ms"] = d.end_ms
            modality = getattr(d, "modality", None)
            if modality:
                m["modality"] = modality.value if hasattr(modality, "value") else str(modality)
            metadatas.append(m)
        return self.add_texts(texts=texts, metadatas=metadatas, ids=ids)

    def add_texts(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        if not texts:
            return []

        dense_vectors = self.embedding_provider.embed_batch(texts)
        if dense_vectors:
            vec_dim = len(dense_vectors[0])
            if self.index_dimension is not None and vec_dim != self.index_dimension:
                raise ValueError(
                    f"EMBEDDING_INDEX_INCOMPATIBLE: Incompatible embedding dimension: index expects {self.index_dimension}-d vectors, "
                    f"but received {vec_dim}-d vectors from provider '{getattr(self.embedding_provider, 'model_name', 'unknown')}'. "
                    f"Cannot mix incompatible embedding models in vector index."
                )
            if self.index_dimension is None:
                self.index_dimension = vec_dim

        generated_ids: List[str] = []
        now = time.time()
        with SessionLocal() as db:
            if self.is_sqlite_vec and self.index_dimension:
                self._ensure_sqlite_vec_table(db)

            table_name = f"vec_items_{self.index_dimension}" if self.index_dimension else None

            for i, text in enumerate(texts):
                doc_id = ids[i] if ids and i < len(ids) else f"vec_{uuid.uuid4().hex[:12]}"
                meta = dict(metadatas[i]) if i < len(metadatas) else {}
                vec = dense_vectors[i] if dense_vectors else []

                tenant_id = meta.get("tenant_id", "default")
                workspace_id = meta.get("workspace_id", "default")
                user_id = meta.get("user_id")
                modality = meta.get("modality", "transcript")
                source_id = meta.get("source_id", meta.get("video_id"))
                parent_id = meta.get("parent_id")
                parent_content = meta.get("parent_content")
                content_hash = meta.get("content_hash")
                if not content_hash and text:
                    content_hash = hashlib.sha256(f"{text}:{modality}".encode("utf-8")).hexdigest()
                start_ms = meta.get("start_ms")
                end_ms = meta.get("end_ms")

                # Upsert record in vector_embeddings
                rec = (
                    db.query(VectorEmbeddingRecord)
                    .filter(
                        (VectorEmbeddingRecord.id == doc_id)
                        | (VectorEmbeddingRecord.document_id == doc_id)
                    )
                    .first()
                )

                if rec:
                    rec.text = text
                    rec.parent_id = parent_id
                    rec.parent_content = parent_content
                    rec.content_hash = content_hash
                    rec.tenant_id = tenant_id
                    rec.workspace_id = workspace_id
                    rec.user_id = user_id
                    rec.modality = modality
                    rec.source_id = source_id
                    rec.start_ms = start_ms
                    rec.end_ms = end_ms
                    rec.vector_json = json.dumps(vec)
                    rec.metadata_json = json.dumps(meta)
                    rec.updated_at = now
                else:
                    rec = VectorEmbeddingRecord(
                        id=doc_id,
                        document_id=doc_id,
                        parent_id=parent_id,
                        parent_content=parent_content,
                        content_hash=content_hash,
                        tenant_id=tenant_id,
                        workspace_id=workspace_id,
                        user_id=user_id,
                        embedding_model=self.index_model_id,
                        embedding_version=self.index_version,
                        embedding_dimension=self.index_dimension or len(vec),
                        modality=modality,
                        source_id=source_id,
                        start_ms=start_ms,
                        end_ms=end_ms,
                        text=text,
                        vector_json=json.dumps(vec),
                        metadata_json=json.dumps(meta),
                        created_at=now,
                        updated_at=now,
                    )
                    db.add(rec)

                # Sync into sqlite-vec virtual table if enabled
                if self.is_sqlite_vec and table_name and vec and len(vec) == self.index_dimension:
                    try:
                        db.execute(
                            sa_text(f"INSERT OR REPLACE INTO {table_name}(id, embedding) VALUES (:id, :emb)"),
                            {"id": doc_id, "emb": json.dumps(vec)},
                        )
                    except Exception as e:
                        logger.warning(f"Failed to insert into sqlite-vec virtual table: {e}")

                generated_ids.append(doc_id)

            db.commit()

        return generated_ids

    def upsert(
        self,
        id: str,
        text: str,
        metadata: Dict[str, Any],
        vector: Optional[List[float]] = None,
    ) -> str:
        """Upsert a single document into the persistent vector store."""
        if vector is not None:
            if self.index_dimension is not None and len(vector) != self.index_dimension:
                raise ValueError(
                    f"EMBEDDING_INDEX_INCOMPATIBLE: Dimension mismatch in upsert: expected {self.index_dimension}, got {len(vector)}"
                )
        generated = self.add_texts(texts=[text], metadatas=[metadata], ids=[id])
        return generated[0] if generated else id

    def upsert_documents(self, docs: List[Any]) -> List[str]:
        """Batch upsert MultiVectorDocument instances."""
        return self.index_documents(docs)

    def update(
        self,
        id: str,
        text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        vector: Optional[List[float]] = None,
    ) -> bool:
        """Update an existing vector embedding record."""
        if vector is not None:
            if self.index_dimension is not None and len(vector) != self.index_dimension:
                raise ValueError(
                    f"EMBEDDING_INDEX_INCOMPATIBLE: Dimension mismatch in update: expected {self.index_dimension}, got {len(vector)}"
                )
        with SessionLocal() as db:
            rec = (
                db.query(VectorEmbeddingRecord)
                .filter(
                    (VectorEmbeddingRecord.id == id)
                    | (VectorEmbeddingRecord.document_id == id)
                )
                .first()
            )
            if not rec:
                return False

            now = time.time()
            if text is not None:
                rec.text = text
                if vector is None:
                    vector = self.embedding_provider.embed_text(text)
            if metadata is not None:
                try:
                    cur_meta = json.loads(rec.metadata_json)
                except Exception:
                    cur_meta = {}
                cur_meta.update(metadata)
                rec.metadata_json = json.dumps(cur_meta)
                if "tenant_id" in metadata:
                    rec.tenant_id = metadata["tenant_id"]
                if "workspace_id" in metadata:
                    rec.workspace_id = metadata["workspace_id"]
                if "user_id" in metadata:
                    rec.user_id = metadata["user_id"]
                if "parent_id" in metadata:
                    rec.parent_id = metadata["parent_id"]
                if "parent_content" in metadata:
                    rec.parent_content = metadata["parent_content"]

            if vector is not None:
                if self.index_dimension is not None and len(vector) != self.index_dimension:
                    raise ValueError(
                        f"EMBEDDING_INDEX_INCOMPATIBLE: Dimension mismatch in update: expected {self.index_dimension}, got {len(vector)}"
                    )
                rec.vector_json = json.dumps(vector)
                if self.is_sqlite_vec and self.index_dimension:
                    table_name = f"vec_items_{self.index_dimension}"
                    try:
                        db.execute(
                            sa_text(f"INSERT OR REPLACE INTO {table_name}(id, embedding) VALUES (:id, :emb)"),
                            {"id": rec.id, "emb": json.dumps(vector)},
                        )
                    except Exception as e:
                        logger.warning(f"Failed to update sqlite-vec virtual table: {e}")

            rec.updated_at = now
            db.commit()
            return True

    def search(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        if not query.strip():
            return []

        prov_dim = getattr(self.embedding_provider, "dimension", None)
        if self.index_dimension is not None and prov_dim is not None and prov_dim != self.index_dimension:
            raise ValueError(
                f"EMBEDDING_INDEX_INCOMPATIBLE: Query vector dimension mismatch: provider produces {prov_dim}-d vectors, "
                f"but index is built with {self.index_dimension}-d vectors."
            )

        q_vec = self.embedding_provider.embed_text(query)
        if self.index_dimension is not None and len(q_vec) != self.index_dimension:
            raise ValueError(
                f"EMBEDDING_INDEX_INCOMPATIBLE: Query vector dimension mismatch: query vector is {len(q_vec)}-d, "
                f"but index expects {self.index_dimension}-d."
            )

        effective_filters = dict(filters) if filters else {}
        if tenant_id is not None:
            effective_filters["tenant_id"] = tenant_id
        if workspace_id is not None:
            effective_filters["workspace_id"] = workspace_id
        if user_id is not None:
            effective_filters["user_id"] = user_id

        hits: List[Tuple[Dict[str, Any], float]] = []

        # 1. Try sqlite-vec SIMD accelerated search first if available
        if self.is_sqlite_vec and self.index_dimension:
            try:
                table_name = f"vec_items_{self.index_dimension}"
                candidate_k = max(k * 15, 150)
                with SessionLocal() as db:
                    vec_matches = db.execute(
                        sa_text(
                            f"SELECT id, distance FROM {table_name} "
                            f"WHERE embedding MATCH :q AND k = :cand_k "
                            f"ORDER BY distance ASC"
                        ),
                        {"q": json.dumps(q_vec), "cand_k": candidate_k},
                    ).fetchall()

                    if vec_matches:
                        matched_ids = [m[0] for m in vec_matches]
                        dist_map = {m[0]: m[1] for m in vec_matches}

                        query_obj = db.query(VectorEmbeddingRecord).filter(
                            VectorEmbeddingRecord.id.in_(matched_ids)
                        )
                        if "tenant_id" in effective_filters:
                            query_obj = query_obj.filter(VectorEmbeddingRecord.tenant_id == effective_filters["tenant_id"])
                        if "workspace_id" in effective_filters:
                            query_obj = query_obj.filter(VectorEmbeddingRecord.workspace_id == effective_filters["workspace_id"])
                        if "user_id" in effective_filters and effective_filters["user_id"] is not None:
                            query_obj = query_obj.filter(VectorEmbeddingRecord.user_id == effective_filters["user_id"])
                        if "video_id" in effective_filters:
                            query_obj = query_obj.filter(VectorEmbeddingRecord.source_id == effective_filters["video_id"])

                        records = query_obj.all()
                        for rec in records:
                            try:
                                meta = json.loads(rec.metadata_json)
                            except Exception:
                                meta = {}

                            match = True
                            for fk, fv in effective_filters.items():
                                if fk in ("tenant_id", "workspace_id", "user_id"):
                                    continue
                                if meta.get(fk) != fv and getattr(rec, fk, None) != fv:
                                    match = False
                                    break
                            if not match:
                                continue

                            if rec.parent_id and "parent_id" not in meta:
                                meta["parent_id"] = rec.parent_id
                            if rec.parent_content and "parent_content" not in meta:
                                meta["parent_content"] = rec.parent_content
                            if rec.content_hash and "content_hash" not in meta:
                                meta["content_hash"] = rec.content_hash

                            raw_dist = dist_map.get(rec.id, 1.0)
                            similarity = max(0.0, min(1.0, 1.0 - raw_dist))
                            if similarity > 0.15:
                                hits.append((meta, round(similarity, 4)))

                        hits.sort(key=lambda x: x[1], reverse=True)
                        if len(hits) >= k or not effective_filters:
                            return hits[:k]
            except Exception as e:
                logger.debug(f"sqlite-vec query failed or table not found, falling back to SQLite cosine math: {e}")
                hits.clear()

        # 2. Deterministic SQLite Cosine-Math Fallback
        q_norm = math.sqrt(sum(v * v for v in q_vec)) or 1.0
        with SessionLocal() as db:
            query_obj = db.query(VectorEmbeddingRecord)
            if "tenant_id" in effective_filters:
                query_obj = query_obj.filter(VectorEmbeddingRecord.tenant_id == effective_filters["tenant_id"])
            if "workspace_id" in effective_filters:
                query_obj = query_obj.filter(VectorEmbeddingRecord.workspace_id == effective_filters["workspace_id"])
            if "user_id" in effective_filters and effective_filters["user_id"] is not None:
                query_obj = query_obj.filter(VectorEmbeddingRecord.user_id == effective_filters["user_id"])
            if "video_id" in effective_filters:
                query_obj = query_obj.filter(VectorEmbeddingRecord.source_id == effective_filters["video_id"])

            records = query_obj.all()
            for rec in records:
                try:
                    meta = json.loads(rec.metadata_json)
                except Exception:
                    meta = {}

                match = True
                for fk, fv in effective_filters.items():
                    if fk in ("tenant_id", "workspace_id", "user_id"):
                        continue
                    if meta.get(fk) != fv and getattr(rec, fk, None) != fv:
                        match = False
                        break
                if not match:
                    continue

                if rec.parent_id and "parent_id" not in meta:
                    meta["parent_id"] = rec.parent_id
                if rec.parent_content and "parent_content" not in meta:
                    meta["parent_content"] = rec.parent_content
                if rec.content_hash and "content_hash" not in meta:
                    meta["content_hash"] = rec.content_hash

                try:
                    d_vec = json.loads(rec.vector_json)
                except Exception:
                    continue

                if not d_vec:
                    continue

                dot = sum(a * b for a, b in zip(q_vec, d_vec))
                d_norm = math.sqrt(sum(v * v for v in d_vec)) or 1.0
                cosine = dot / (q_norm * d_norm)

                if cosine > 0.15:
                    hits.append((meta, round(cosine, 4)))

        hits.sort(key=lambda x: x[1], reverse=True)
        return hits[:k]

    def delete(self, ids: List[str]) -> int:
        if not ids:
            return 0
        with SessionLocal() as db:
            deleted = (
                db.query(VectorEmbeddingRecord)
                .filter(
                    (VectorEmbeddingRecord.id.in_(ids))
                    | (VectorEmbeddingRecord.document_id.in_(ids))
                )
                .delete(synchronize_session=False)
            )
            if self.is_sqlite_vec and self.index_dimension:
                table_name = f"vec_items_{self.index_dimension}"
                for did in ids:
                    try:
                        db.execute(sa_text(f"DELETE FROM {table_name} WHERE id = :id"), {"id": did})
                    except Exception:
                        pass
            db.commit()
            return deleted

    def delete_by_video(
        self,
        video_id: str,
        tenant_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> int:
        """Delete all vectors for a specific video, honoring tenant and workspace boundaries."""
        with SessionLocal() as db:
            q = db.query(VectorEmbeddingRecord).filter(VectorEmbeddingRecord.source_id == video_id)
            if tenant_id:
                q = q.filter(VectorEmbeddingRecord.tenant_id == tenant_id)
            if workspace_id:
                q = q.filter(VectorEmbeddingRecord.workspace_id == workspace_id)

            records = q.all()
            ids_to_del = [r.id for r in records]
            if not ids_to_del:
                return 0

            deleted = (
                db.query(VectorEmbeddingRecord)
                .filter(VectorEmbeddingRecord.id.in_(ids_to_del))
                .delete(synchronize_session=False)
            )
            if self.is_sqlite_vec and self.index_dimension:
                table_name = f"vec_items_{self.index_dimension}"
                for did in ids_to_del:
                    try:
                        db.execute(sa_text(f"DELETE FROM {table_name} WHERE id = :id"), {"id": did})
                    except Exception:
                        pass
            db.commit()
            return deleted

    def delete_by_document(
        self,
        doc_id: str,
        tenant_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> int:
        """Delete a document or parent and its children."""
        with SessionLocal() as db:
            q = db.query(VectorEmbeddingRecord).filter(
                (VectorEmbeddingRecord.id == doc_id)
                | (VectorEmbeddingRecord.document_id == doc_id)
                | (VectorEmbeddingRecord.parent_id == doc_id)
            )
            if tenant_id:
                q = q.filter(VectorEmbeddingRecord.tenant_id == tenant_id)
            if workspace_id:
                q = q.filter(VectorEmbeddingRecord.workspace_id == workspace_id)

            records = q.all()
            ids_to_del = [r.id for r in records]
            if not ids_to_del:
                return 0

            deleted = (
                db.query(VectorEmbeddingRecord)
                .filter(VectorEmbeddingRecord.id.in_(ids_to_del))
                .delete(synchronize_session=False)
            )
            if self.is_sqlite_vec and self.index_dimension:
                table_name = f"vec_items_{self.index_dimension}"
                for did in ids_to_del:
                    try:
                        db.execute(sa_text(f"DELETE FROM {table_name} WHERE id = :id"), {"id": did})
                    except Exception:
                        pass
            db.commit()
            return deleted

    def delete_by_workspace(self, workspace_id: str, tenant_id: Optional[str] = None) -> int:
        """Delete all vectors for a specific workspace."""
        with SessionLocal() as db:
            q = db.query(VectorEmbeddingRecord).filter(VectorEmbeddingRecord.workspace_id == workspace_id)
            if tenant_id:
                q = q.filter(VectorEmbeddingRecord.tenant_id == tenant_id)
            records = q.all()
            ids_to_del = [r.id for r in records]
            if not ids_to_del:
                return 0

            deleted = (
                db.query(VectorEmbeddingRecord)
                .filter(VectorEmbeddingRecord.id.in_(ids_to_del))
                .delete(synchronize_session=False)
            )
            if self.is_sqlite_vec and self.index_dimension:
                table_name = f"vec_items_{self.index_dimension}"
                for did in ids_to_del:
                    try:
                        db.execute(sa_text(f"DELETE FROM {table_name} WHERE id = :id"), {"id": did})
                    except Exception:
                        pass
            db.commit()
            return deleted

    def delete_by_tenant(self, tenant_id: str) -> int:
        """Delete all vectors for a specific tenant."""
        with SessionLocal() as db:
            records = db.query(VectorEmbeddingRecord).filter(VectorEmbeddingRecord.tenant_id == tenant_id).all()
            ids_to_del = [r.id for r in records]
            if not ids_to_del:
                return 0

            deleted = (
                db.query(VectorEmbeddingRecord)
                .filter(VectorEmbeddingRecord.id.in_(ids_to_del))
                .delete(synchronize_session=False)
            )
            if self.is_sqlite_vec and self.index_dimension:
                table_name = f"vec_items_{self.index_dimension}"
                for did in ids_to_del:
                    try:
                        db.execute(sa_text(f"DELETE FROM {table_name} WHERE id = :id"), {"id": did})
                    except Exception:
                        pass
            db.commit()
            return deleted

    def delete_by_filter(self, filters: Dict[str, Any]) -> int:
        if not filters:
            return 0
        deleted = 0
        with SessionLocal() as db:
            query_obj = db.query(VectorEmbeddingRecord)
            if "tenant_id" in filters:
                query_obj = query_obj.filter(VectorEmbeddingRecord.tenant_id == filters["tenant_id"])
            if "workspace_id" in filters:
                query_obj = query_obj.filter(VectorEmbeddingRecord.workspace_id == filters["workspace_id"])
            if "video_id" in filters:
                query_obj = query_obj.filter(VectorEmbeddingRecord.source_id == filters["video_id"])
            if "user_id" in filters:
                query_obj = query_obj.filter(VectorEmbeddingRecord.user_id == filters["user_id"])

            records = query_obj.all()
            ids_to_del = []
            for rec in records:
                try:
                    meta = json.loads(rec.metadata_json)
                except Exception:
                    meta = {}
                if all(meta.get(k) == v for k, v in filters.items()):
                    ids_to_del.append(rec.id)

            if ids_to_del:
                deleted = (
                    db.query(VectorEmbeddingRecord)
                    .filter(VectorEmbeddingRecord.id.in_(ids_to_del))
                    .delete(synchronize_session=False)
                )
                if self.is_sqlite_vec and self.index_dimension:
                    table_name = f"vec_items_{self.index_dimension}"
                    for did in ids_to_del:
                        try:
                            db.execute(sa_text(f"DELETE FROM {table_name} WHERE id = :id"), {"id": did})
                        except Exception:
                            pass
                db.commit()

        return deleted

    def count(
        self,
        filters: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> int:
        """Count records matching filter criteria."""
        with SessionLocal() as db:
            q = db.query(VectorEmbeddingRecord)
            effective = dict(filters) if filters else {}
            if tenant_id:
                effective["tenant_id"] = tenant_id
            if workspace_id:
                effective["workspace_id"] = workspace_id

            if "tenant_id" in effective:
                q = q.filter(VectorEmbeddingRecord.tenant_id == effective["tenant_id"])
            if "workspace_id" in effective:
                q = q.filter(VectorEmbeddingRecord.workspace_id == effective["workspace_id"])
            if "video_id" in effective:
                q = q.filter(VectorEmbeddingRecord.source_id == effective["video_id"])
            if "user_id" in effective:
                q = q.filter(VectorEmbeddingRecord.user_id == effective["user_id"])
            return q.count()

    def health_check(self) -> Dict[str, Any]:
        """Diagnostic health report for vector store."""
        status = "healthy"
        doc_count = 0
        try:
            with SessionLocal() as db:
                doc_count = db.query(VectorEmbeddingRecord).count()
        except Exception as e:
            status = f"unhealthy: {e}"

        return {
            "status": status,
            "store_type": "sqlite_persistent",
            "is_sqlite_vec": self.is_sqlite_vec,
            "classification": self.classification,
            "doc_count": doc_count,
            "index_dimension": self.index_dimension,
            "index_model_id": self.index_model_id,
            "index_version": self.index_version,
        }

    def rebuild_index(self) -> Dict[str, Any]:
        """Rebuild sqlite-vec virtual table from persistent SQLite records."""
        if not self.is_sqlite_vec or not self.index_dimension:
            return {"status": "skipped", "reason": "sqlite-vec not active"}

        table_name = f"vec_items_{self.index_dimension}"
        synced = 0
        with SessionLocal() as db:
            try:
                db.execute(sa_text(f"DROP TABLE IF EXISTS {table_name}"))
                self._ensure_sqlite_vec_table(db)
                records = db.query(VectorEmbeddingRecord).all()
                for rec in records:
                    try:
                        vec = json.loads(rec.vector_json)
                        if vec and len(vec) == self.index_dimension:
                            db.execute(
                                sa_text(f"INSERT OR REPLACE INTO {table_name}(id, embedding) VALUES (:id, :emb)"),
                                {"id": rec.id, "emb": json.dumps(vec)},
                            )
                            synced += 1
                    except Exception:
                        continue
                db.commit()
            except Exception as e:
                logger.error(f"Rebuild index error: {e}")
                return {"status": "error", "error": str(e)}

        return {"status": "success", "synced_vectors": synced}

    def migrate_from_in_memory(
        self,
        in_memory_store: InMemoryVectorStore,
        tenant_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> int:
        """Migrate all documents from an InMemoryVectorStore instance into persistent storage."""
        if not in_memory_store or not in_memory_store.documents:
            return 0

        texts = []
        metas = []
        ids = []

        for doc_id, doc_data in in_memory_store.documents.items():
            text = doc_data.get("text", "")
            meta = dict(doc_data.get("metadata", {}))
            if tenant_id and "tenant_id" not in meta:
                meta["tenant_id"] = tenant_id
            if workspace_id and "workspace_id" not in meta:
                meta["workspace_id"] = workspace_id
            texts.append(text)
            metas.append(meta)
            ids.append(doc_id)

        added = self.add_texts(texts=texts, metadatas=metas, ids=ids)
        return len(added)

    @property
    def documents(self) -> Dict[str, Any]:
        """Backwards compatibility property returning document mapping from SQLite."""
        docs: Dict[str, Any] = {}
        with SessionLocal() as db:
            records = db.query(VectorEmbeddingRecord).all()
            for rec in records:
                try:
                    meta = json.loads(rec.metadata_json)
                except Exception:
                    meta = {}
                if rec.parent_id:
                    meta["parent_id"] = rec.parent_id
                if rec.parent_content:
                    meta["parent_content"] = rec.parent_content
                if rec.content_hash:
                    meta["content_hash"] = rec.content_hash
                if rec.tenant_id:
                    meta["tenant_id"] = rec.tenant_id
                if rec.workspace_id:
                    meta["workspace_id"] = rec.workspace_id
                if rec.source_id:
                    meta["video_id"] = rec.source_id
                    meta["source_id"] = rec.source_id
                if rec.start_ms is not None:
                    meta["start"] = rec.start_ms / 1000.0
                if rec.end_ms is not None:
                    meta["end"] = rec.end_ms / 1000.0

                docs[rec.document_id or rec.id] = {
                    "text": rec.text,
                    "metadata": meta,
                }
        return docs


_global_sqlite_store: Optional[PersistentSQLiteVectorStore] = None
_global_inmemory_store: Optional[InMemoryVectorStore] = None


def get_vector_store(store_type: Optional[str] = None) -> BaseVectorStore:
    global _global_sqlite_store, _global_inmemory_store
    from config.settings import get_settings
    settings = get_settings()

    selected_type = (store_type or settings.vector_store_type or "sqlite").strip().lower()

    if selected_type in ("sqlite", "sqlite-vec", "persistent"):
        if _global_sqlite_store is None:
            _global_sqlite_store = PersistentSQLiteVectorStore(
                embedding_provider=LocalDenseEmbeddingProvider()
            )
        return _global_sqlite_store
    elif selected_type == "chroma":
        return ChromaVectorStore()
    else:
        if _global_inmemory_store is None:
            _global_inmemory_store = InMemoryVectorStore()
        return _global_inmemory_store
