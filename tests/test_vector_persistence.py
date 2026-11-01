"""Tests for PersistentSQLiteVectorStore persistence, multi-tenancy, and dimension guards."""

from __future__ import annotations

import uuid
import pytest

from app.database.models import VectorEmbeddingRecord
from app.database.session import SessionLocal
from app.retrieval.multi_vector import MultiVectorDocument
from app.retrieval.vector_store import (
    LocalDenseEmbeddingProvider,
    PersistentSQLiteVectorStore,
)


def test_sqlite_vector_persistence_across_instances():
    embedding_provider = LocalDenseEmbeddingProvider()
    store1 = PersistentSQLiteVectorStore(embedding_provider=embedding_provider)

    doc_id = f"doc_{uuid.uuid4().hex[:8]}"
    text = "Photosynthesis converts light energy into chemical energy stored in glucose."
    meta = {"title": "Biology Lecture 1", "topic": "Photosynthesis", "video_id": "vid_bio_1"}

    doc = MultiVectorDocument(
        id=doc_id,
        content=text,
        metadata=meta,
        tenant_id="tenant_alpha",
        workspace_id="ws_main",
    )

    # 1. Index document into store 1
    generated_ids = store1.index_documents([doc])
    assert len(generated_ids) == 1

    # 2. Query from store 1
    hits1 = store1.search("light energy conversion into glucose", k=3, tenant_id="tenant_alpha")
    assert len(hits1) > 0
    assert hits1[0][0]["topic"] == "Photosynthesis"

    # 3. Simulate process restart by creating a completely fresh store instance
    store2 = PersistentSQLiteVectorStore(embedding_provider=embedding_provider)
    hits2 = store2.search("light energy conversion into glucose", k=3, tenant_id="tenant_alpha")
    assert len(hits2) > 0
    assert hits2[0][0]["topic"] == "Photosynthesis"

    # Verify document exists in compatibility property
    assert doc_id in store2.documents

    # 4. Clean up
    deleted = store2.delete([doc_id])
    assert deleted >= 1


def test_sqlite_vector_tenant_and_workspace_isolation():
    embedding_provider = LocalDenseEmbeddingProvider()
    store = PersistentSQLiteVectorStore(embedding_provider=embedding_provider)

    doc_a_id = f"doc_a_{uuid.uuid4().hex[:6]}"
    doc_b_id = f"doc_b_{uuid.uuid4().hex[:6]}"

    doc_a = MultiVectorDocument(
        id=doc_a_id,
        content="Tenant Alpha proprietary artificial intelligence algorithm.",
        metadata={"title": "Alpha Research"},
        tenant_id="tenant_alpha",
        workspace_id="ws_1",
    )
    doc_b = MultiVectorDocument(
        id=doc_b_id,
        content="Tenant Beta proprietary neural architecture.",
        metadata={"title": "Beta Research"},
        tenant_id="tenant_beta",
        workspace_id="ws_2",
    )

    store.index_documents([doc_a, doc_b])

    # Search as tenant Alpha: should only see Alpha
    results_alpha = store.search("proprietary intelligence architecture", k=5, tenant_id="tenant_alpha")
    alpha_titles = [r[0]["title"] for r in results_alpha]
    assert "Alpha Research" in alpha_titles
    assert "Beta Research" not in alpha_titles

    # Search as tenant Beta: should only see Beta
    results_beta = store.search("proprietary intelligence architecture", k=5, tenant_id="tenant_beta")
    beta_titles = [r[0]["title"] for r in results_beta]
    assert "Beta Research" in beta_titles
    assert "Alpha Research" not in beta_titles

    # Clean up
    store.delete([doc_a_id, doc_b_id])


def test_dimension_mismatch_guard():
    embedding_provider = LocalDenseEmbeddingProvider()
    store = PersistentSQLiteVectorStore(
        embedding_provider=embedding_provider,
        index_dimension=128,  # configured for 128
    )

    # Provider produces dimension matching embedding_provider.dimension (e.g. 128)
    assert store.index_dimension == 128

    # If an incompatible store dimension is configured
    incompatible_store = PersistentSQLiteVectorStore(
        embedding_provider=embedding_provider,
        index_dimension=512,  # mismatch with 128
    )

    with pytest.raises(ValueError) as exc_info:
        incompatible_store.search("test query", k=3)
    assert "EMBEDDING_INDEX_INCOMPATIBLE" in str(exc_info.value)


def test_delete_by_filter():
    embedding_provider = LocalDenseEmbeddingProvider()
    store = PersistentSQLiteVectorStore(embedding_provider=embedding_provider)

    batch = [
        MultiVectorDocument(
            id=f"del_test_{i}_{uuid.uuid4().hex[:6]}",
            content=f"Sample text content number {i}",
            metadata={"category": "temporary_batch", "batch_num": 99},
            tenant_id="test_del_tenant",
        )
        for i in range(3)
    ]

    store.index_documents(batch)
    deleted = store.delete_by_filter({"category": "temporary_batch", "batch_num": 99})
    assert deleted == 3
