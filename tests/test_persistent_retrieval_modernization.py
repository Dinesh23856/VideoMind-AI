"""
Comprehensive Test Suite for P1 Storage & Retrieval Modernization.

Validates:
1. PersistentSQLiteVectorStore sqlite-vec SIMD acceleration (REAL classification).
2. Persistent deterministic SQLite cosine-math fallback (FALLBACK classification).
3. Parent-Child knowledge chunking, hierarchy, and bi-directional provenance.
4. Multimodal knowledge unit chunking (transcripts, OCR, keyframes).
5. Strict multi-tenant and workspace boundary isolation across all retrieval paths.
6. Dimension incompatibility prevention with EMBEDDING_INDEX_INCOMPATIBLE error.
7. Complete vector index lifecycle (upsert, update, count, delete by video/workspace/tenant, health_check, rebuild_index).
8. Seamless migration from InMemoryVectorStore to PersistentSQLiteVectorStore.
9. Automatic corpus hydration across fresh HybridSearchEngine instances without data loss.
"""

from __future__ import annotations

import uuid
import pytest
from sqlalchemy import text as sa_text

from app.database.models import VectorEmbeddingRecord
from app.database.session import SessionLocal
from app.domain.models import TranscriptSegment
from app.embeddings.provider import LocalDenseEmbeddingProvider
from app.rag.engine import RAGEngine
from app.retrieval.hybrid_search import HybridSearchEngine, RetrievalStrategy
from app.retrieval.multi_vector import (
    ModalityType,
    MultiVectorDocument,
    ParentChildChunker,
    ParentDocument,
)
from app.retrieval.vector_store import (
    InMemoryVectorStore,
    PersistentSQLiteVectorStore,
)


@pytest.fixture(autouse=True)
def clean_test_vectors():
    """Clean up test records after tests."""
    yield
    with SessionLocal() as db:
        try:
            db.query(VectorEmbeddingRecord).filter(
                VectorEmbeddingRecord.tenant_id.like("test_%")
                | VectorEmbeddingRecord.tenant_id.in_(["tenant_alpha", "tenant_beta", "migrated_tenant"])
            ).delete(synchronize_session=False)
            db.commit()
        except Exception:
            pass


def test_sqlite_vec_acceleration_and_classification():
    """Verify sqlite-vec virtual table acceleration and truthful REAL classification."""
    provider = LocalDenseEmbeddingProvider()
    store = PersistentSQLiteVectorStore(embedding_provider=provider, index_dimension=128)

    # Truthful classification
    assert store.classification in ("REAL", "FALLBACK")
    if store.is_sqlite_vec:
        assert store.classification == "REAL"
        # Verify virtual table exists in SQLite
        with SessionLocal() as db:
            tbl_check = db.execute(
                sa_text("SELECT name FROM sqlite_master WHERE type='table' AND name='vec_items_128'")
            ).scalar()
            assert tbl_check == "vec_items_128"

    # Add a document and verify search returns high similarity
    doc_id = f"vec_accel_{uuid.uuid4().hex[:8]}"
    doc = MultiVectorDocument(
        id=doc_id,
        content="Backpropagation computes gradients of the loss function with respect to weights using chain rule.",
        metadata={"topic": "Neural Optimization", "title": "Deep Learning Lecture"},
        tenant_id="test_accel_tenant",
        workspace_id="test_ws",
    )
    store.index_documents([doc])

    hits = store.search("backpropagation chain rule gradient computation", k=3, tenant_id="test_accel_tenant")
    assert len(hits) > 0
    assert hits[0][0]["topic"] == "Neural Optimization"
    assert hits[0][1] > 0.25


def test_deterministic_cosine_fallback():
    """Verify deterministic fallback cosine math when sqlite-vec is disabled."""
    provider = LocalDenseEmbeddingProvider()
    store = PersistentSQLiteVectorStore(embedding_provider=provider, index_dimension=128)

    # Force fallback mode
    store.is_sqlite_vec = False
    store.classification = "FALLBACK"

    doc_id = f"vec_fall_{uuid.uuid4().hex[:8]}"
    doc = MultiVectorDocument(
        id=doc_id,
        content="Conjugate gradient method optimizes quadratic forms without explicit matrix inversion.",
        metadata={"topic": "Numerical Linear Algebra"},
        tenant_id="test_fallback_tenant",
        workspace_id="test_ws",
    )
    store.index_documents([doc])

    hits = store.search("conjugate gradient optimization quadratic forms", k=3, tenant_id="test_fallback_tenant")
    assert len(hits) > 0
    assert hits[0][0]["topic"] == "Numerical Linear Algebra"
    assert hits[0][1] > 0.25


def test_parent_child_chunking_and_provenance():
    """Verify parent-child chunking preserves bi-directional hierarchy and search context."""
    segments = [
        TranscriptSegment(id="s1", start=0.0, end=30.0, text="Welcome to the lecture on Attention Mechanisms."),
        TranscriptSegment(id="s2", start=30.0, end=75.0, text="Dot-product attention computes similarity between queries and keys."),
        TranscriptSegment(id="s3", start=75.0, end=115.0, text="Scaling by square root of dimension prevents vanishing softmax gradients."),
    ]

    parents, children = ParentChildChunker.chunk_transcript(
        video_id="vid_attn_101",
        video_title="Attention in Deep Learning",
        segments=segments,
        parent_window_sec=120.0,
        tenant_id="test_parent_child_tenant",
        workspace_id="ws_nlp",
    )

    assert len(parents) == 1
    assert len(children) == 3
    parent = parents[0]
    assert parent.section_type in ("transcript_section", "topic_section")
    assert len(parent.child_ids) == 3

    for child in children:
        assert child.parent_id == parent.id
        assert child.parent_content == parent.content
        assert child.content_hash is not None
        assert len(child.content_hash) == 64

    # Index into HybridSearchEngine
    provider = LocalDenseEmbeddingProvider()
    store = PersistentSQLiteVectorStore(embedding_provider=provider)
    engine = HybridSearchEngine(vector_store=store)

    engine.index_parent_child(parent, children)

    # Retrieve evidence pack
    pack = engine.retrieve_evidence_pack(
        query="scaling by square root of dimension softmax gradients",
        k=3,
        tenant_id="test_parent_child_tenant",
        workspace_id="ws_nlp",
    )

    assert len(pack.hits) > 0
    top_hit = pack.hits[0]
    assert top_hit.parent_id == parent.id
    assert top_hit.parent_snippet is not None
    assert "Attention Mechanisms" in top_hit.parent_snippet

    # Verify context compression includes parent topic framing
    assert "[Topic:" in pack.compressed_context

    # Verify RAGEngine citations carry parent context
    rag = RAGEngine(search_engine=engine)
    rag_resp = rag.answer_query(
        query="Why do we scale dot products in attention?",
        tenant_id="test_parent_child_tenant",
        workspace_id="ws_nlp",
    )
    assert len(rag_resp.citations) > 0
    assert rag_resp.citations[0].parent_id == parent.id
    assert rag_resp.citations[0].parent_snippet is not None


def test_parent_child_multimodal_unit_chunking():
    """Verify multimodal unit decomposition into linked parent-child items."""
    keyframes = [
        {"id": "kf1", "content": "Q = XW_Q, K = XW_K, V = XW_V", "modality": "ocr", "start_ms": 15000, "end_ms": 30000},
        {"id": "kf2", "content": "Softmax(QK^T / sqrt(d_k)) * V", "modality": "formula", "start_ms": 30000, "end_ms": 60000},
        {"id": "kf3", "content": "Projection matrices and attention scores diagram", "modality": "keyframe", "start_ms": 15000, "end_ms": 60000},
    ]

    parent, children = ParentChildChunker.chunk_multimodal_unit(
        video_id="vid_multimodal_1",
        video_title="Deep Learning Architecture",
        section_title="Transformer Projection Math",
        section_content="Projection matrices and scaled dot-product attention formula.",
        items=keyframes,
        start_ms=15000,
        end_ms=60000,
        tenant_id="test_multimodal_tenant",
    )

    assert parent.section_type == "multimodal_unit"
    assert len(children) == 3
    for c in children:
        assert c.parent_id == parent.id
        assert c.tenant_id == "test_multimodal_tenant"
        assert c.modality in (ModalityType.KEYFRAME, ModalityType.OCR, ModalityType.FORMULA)


def test_strict_multi_tenant_and_workspace_isolation():
    """Ensure hard boundary isolation between tenants and workspaces."""
    provider = LocalDenseEmbeddingProvider()
    store = PersistentSQLiteVectorStore(embedding_provider=provider)

    doc_t1_w1 = MultiVectorDocument(
        id=f"doc_t1_w1_{uuid.uuid4().hex[:6]}",
        content="Confidential quantum cryptography key exchange protocol.",
        metadata={"title": "T1 W1 Research"},
        tenant_id="test_tenant_1",
        workspace_id="ws_alpha",
    )
    doc_t1_w2 = MultiVectorDocument(
        id=f"doc_t1_w2_{uuid.uuid4().hex[:6]}",
        content="Internal quantum error correction code benchmark.",
        metadata={"title": "T1 W2 Research"},
        tenant_id="test_tenant_1",
        workspace_id="ws_beta",
    )
    doc_t2_w1 = MultiVectorDocument(
        id=f"doc_t2_w1_{uuid.uuid4().hex[:6]}",
        content="Competitor quantum sensing and magnetometry analysis.",
        metadata={"title": "T2 W1 Research"},
        tenant_id="test_tenant_2",
        workspace_id="ws_alpha",
    )

    store.index_documents([doc_t1_w1, doc_t1_w2, doc_t2_w1])

    # 1. Search Tenant 1, Workspace Alpha
    res1 = store.search("quantum cryptography key exchange", k=5, tenant_id="test_tenant_1", workspace_id="ws_alpha")
    titles1 = [r[0]["title"] for r in res1]
    assert "T1 W1 Research" in titles1
    assert "T1 W2 Research" not in titles1
    assert "T2 W1 Research" not in titles1

    # 2. Search Tenant 1, Workspace Beta
    res2 = store.search("quantum error correction", k=5, tenant_id="test_tenant_1", workspace_id="ws_beta")
    titles2 = [r[0]["title"] for r in res2]
    assert "T1 W2 Research" in titles2
    assert "T1 W1 Research" not in titles2
    assert "T2 W1 Research" not in titles2

    # 3. Search Tenant 2
    res3 = store.search("quantum sensing magnetometry", k=5, tenant_id="test_tenant_2")
    titles3 = [r[0]["title"] for r in res3]
    assert "T2 W1 Research" in titles3
    assert "T1 W1 Research" not in titles3
    assert "T1 W2 Research" not in titles3


def test_dimension_incompatibility_prevention():
    """Verify strict rejection of incompatible vector dimensions."""
    provider = LocalDenseEmbeddingProvider()
    store = PersistentSQLiteVectorStore(embedding_provider=provider, index_dimension=128)

    # Incompatible dimension in upsert
    with pytest.raises(ValueError) as exc:
        store.upsert(
            id="bad_dim_doc",
            text="Incompatible dimension test",
            metadata={},
            vector=[0.1] * 256,  # 256 != 128
        )
    assert "EMBEDDING_INDEX_INCOMPATIBLE" in str(exc.value)

    # Incompatible dimension in update
    with pytest.raises(ValueError) as exc:
        store.update(id="nonexistent", vector=[0.1] * 512)
    assert "EMBEDDING_INDEX_INCOMPATIBLE" in str(exc.value)


def test_complete_vector_lifecycle_management():
    """Verify upsert, update, count, delete variations, health check, and rebuild."""
    provider = LocalDenseEmbeddingProvider()
    store = PersistentSQLiteVectorStore(embedding_provider=provider)
    tid = "test_lifecycle_tenant"
    wid = "ws_ops"

    # 1. Health check
    health = store.health_check()
    assert health["status"] == "healthy"
    assert health["store_type"] == "sqlite_persistent"

    # 2. Upsert
    doc_id = f"life_{uuid.uuid4().hex[:8]}"
    store.upsert(
        id=doc_id,
        text="Initial draft of system documentation.",
        metadata={"tenant_id": tid, "workspace_id": wid, "video_id": "vid_life_1"},
    )
    assert store.count(tenant_id=tid) == 1

    # 3. Update
    store.update(
        id=doc_id,
        text="Updated complete edition of system documentation.",
        metadata={"version": 2},
    )
    hits = store.search("complete edition system documentation", k=1, tenant_id=tid)
    assert len(hits) == 1
    assert hits[0][0]["version"] == 2

    # 4. Count with filter
    assert store.count(tenant_id=tid, workspace_id=wid) == 1

    # 5. Rebuild Index
    rebuild = store.rebuild_index()
    assert rebuild["status"] in ("success", "skipped")

    # 6. Delete by video
    del_vid = store.delete_by_video("vid_life_1", tenant_id=tid)
    assert del_vid == 1
    assert store.count(tenant_id=tid) == 0


def test_migrate_from_in_memory_to_persistent():
    """Verify seamless migration from ephemeral InMemoryVectorStore to PersistentSQLiteVectorStore."""
    in_mem = InMemoryVectorStore()
    in_mem.add_texts(
        texts=[
            "MapReduce processes large datasets with a distributed parallel algorithm.",
            "Resilient Distributed Datasets (RDDs) provide fault-tolerant in-memory computing.",
        ],
        metadatas=[
            {"framework": "Hadoop", "topic": "MapReduce"},
            {"framework": "Spark", "topic": "RDD"},
        ],
        ids=["doc_hadoop", "doc_spark"],
    )
    assert len(in_mem.documents) == 2

    provider = LocalDenseEmbeddingProvider()
    persistent_store = PersistentSQLiteVectorStore(embedding_provider=provider)

    migrated = persistent_store.migrate_from_in_memory(
        in_memory_store=in_mem,
        tenant_id="migrated_tenant",
        workspace_id="ws_bigdata",
    )
    assert migrated == 2

    hits = persistent_store.search("fault-tolerant in-memory computing", k=2, tenant_id="migrated_tenant")
    assert len(hits) > 0
    assert hits[0][0]["framework"] == "Spark"


def test_corpus_hydration_across_fresh_hybrid_search_instances():
    """Verify that a fresh HybridSearchEngine instance automatically hydrates its corpus from SQLite."""
    provider = LocalDenseEmbeddingProvider()
    store1 = PersistentSQLiteVectorStore(embedding_provider=provider)
    engine1 = HybridSearchEngine(vector_store=store1)

    doc_id = f"hydrate_{uuid.uuid4().hex[:8]}"
    doc = MultiVectorDocument(
        id=doc_id,
        content="Paxos consensus protocol guarantees safety in asynchronous networks despite node failures.",
        metadata={"algorithm": "Paxos", "field": "Distributed Systems"},
        tenant_id="test_hydration_tenant",
        workspace_id="ws_dist",
        source_id="vid_paxos",
    )
    engine1.index_document(doc)

    # Simulate fresh app startup with empty in-memory corpus
    store2 = PersistentSQLiteVectorStore(embedding_provider=provider)
    engine2 = HybridSearchEngine(vector_store=store2)
    assert len(engine2.segments_corpus) == 0  # Starts empty in RAM

    # Querying should trigger automatic corpus hydration and return results
    hits = engine2.search(
        query="paxos consensus protocol safety asynchronous networks",
        k=3,
        tenant_id="test_hydration_tenant",
    )
    assert len(hits) > 0
    assert hits[0].metadata.get("algorithm") == "Paxos" or "Paxos" in hits[0].snippet
    assert len(engine2.segments_corpus) > 0  # Successfully hydrated!
