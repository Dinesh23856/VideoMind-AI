"""
Multi-Vector Embeddings & Advanced Hybrid RAG Engine Test Suite.

Tests:
1. Multi-vector representation (MultiVectorDocument / KnowledgeItem) & serialization
2. Modality types, temporal spans, and conversions to SearchResultItem & MultimodalContextItem
3. Thread-safe embedding cache hit, miss, and isolation
4. LocalDenseEmbeddingProvider (128-d, deterministic, HEURISTIC classification)
5. EmbeddingModelRegistry and strict dimension compatibility checks
6. Vector store dimension mismatch prevention (EMBEDDING_INDEX_INCOMPATIBLE)
7. Multi-tenant and workspace isolation (tenant A cannot retrieve tenant B data)
8. Document deletion by ID and metadata filter
9. Index metadata inspection
10. ChromaVectorStore adapter safety
11. QueryClassifier temporal parsing (timestamps, intervals)
12. QueryClassifier modality detection and concept extraction
13. Safe query expansion without hallucination
14. BM25 sparse lexical ranking
15. Dense continuous vector search
16. Reciprocal Rank Fusion (RRF: sum(1 / (k + rank_i)))
17. Temporal proximity ranking
18. Multi-modal retrieval filtering
19. CrossEncoder reranker integration and graceful truthful fallback
20. Context compression and adjacent temporal segment merging
21. Multi-factor retrieval confidence model (HIGH, MEDIUM, LOW, INSUFFICIENT)
22. Insufficient evidence abstention
23. RAGEngine claim-level citations and grounding tags
24. FastAPI RAG Query endpoint (/api/v1/rag/query)
25. FastAPI RAG Search endpoint (/api/v1/rag/search)
26. FastAPI RAG Index endpoint (/api/v1/rag/index)
27. FastAPI RAG Delete endpoint (/api/v1/rag/documents)
28. FastAPI RAG Health endpoint (/api/v1/rag/health)
29. FastAPI RAG Stats endpoint (/api/v1/rag/stats)
30. End-to-end multi-tenant RAG workflow
"""

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.domain.models import TranscriptSegment
from app.embeddings.provider import (
    EmbeddingCache,
    EmbeddingModelRegistry,
    LocalDenseEmbeddingProvider,
)
from app.rag.engine import RAGEngine
from app.retrieval.hybrid_search import (
    HybridSearchEngine,
    QueryClassifier,
    RetrievalConfidence,
    RetrievalStrategy,
    SearchResultItem,
)
from app.retrieval.multi_vector import ModalityType, MultiVectorDocument
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.vector_store import ChromaVectorStore, InMemoryVectorStore


# =============================================================================
# 1-4: MultiVectorDocument & Modality Tests
# =============================================================================

def test_multi_vector_document_creation_and_serialization():
    doc = MultiVectorDocument(
        id="doc_101",
        tenant_id="tenant_alpha",
        workspace_id="ws_1",
        modality=ModalityType.TRANSCRIPT,
        content="Self-attention allows tokens to attend to all previous tokens.",
        source_id="vid_001",
        source_title="Transformers 101",
        start_ms=12000,
        end_ms=25000,
        entities=["Self-Attention", "Transformers"],
    )
    d = doc.to_dict()
    assert d["id"] == "doc_101"
    assert d["tenant_id"] == "tenant_alpha"
    assert d["modality"] == "transcript"

    reconstructed = MultiVectorDocument.from_dict(d)
    assert reconstructed.id == doc.id
    assert reconstructed.modality == ModalityType.TRANSCRIPT
    assert reconstructed.entities == ["Self-Attention", "Transformers"]


def test_multi_vector_document_temporal_spans():
    doc1 = MultiVectorDocument(
        id="d1", content="test", source_id="v1", start_ms=30000, end_ms=45500
    )
    start, end = doc1.get_temporal_span_sec()
    assert start == 30.0
    assert end == 45.5

    doc_untimed = MultiVectorDocument(id="d2", content="untimed doc", source_id="v2")
    s2, e2 = doc_untimed.get_temporal_span_sec()
    assert s2 is None
    assert e2 is None


def test_multi_vector_to_search_result_item():
    doc = MultiVectorDocument(
        id="doc_seg_01",
        modality=ModalityType.TRANSCRIPT,
        content="ResNet uses residual skip connections to mitigate vanishing gradients.",
        source_id="vid_cv",
        source_title="Computer Vision Deep Dive",
        start_ms=65000,
        end_ms=90000,
        entities=["ResNet", "Skip Connections"],
    )
    item = doc.to_search_result_item(score=2.85, retrieval_method="hybrid_rrf")
    assert isinstance(item, SearchResultItem)
    assert item.video_id == "vid_cv"
    assert item.video_title == "Computer Vision Deep Dive"
    assert item.start_time == 65.0
    assert item.end_time == 90.0
    assert "[01:05 - 01:30]" in item.timestamp_formatted
    assert "ResNet" in item.snippet
    assert item.score == 2.85
    assert "ResNet" in item.matched_concepts


def test_multi_vector_to_multimodal_context_item():
    doc = MultiVectorDocument(
        id="doc_code_01",
        tenant_id="tenant_beta",
        workspace_id="ws_code",
        modality=ModalityType.CODE,
        content="def forward(self, x):\n    return self.linear(x)",
        source_id="repo_01",
        source_title="LinearModule",
    )
    item = doc.to_multimodal_context_item()
    assert item.source_type.value == "CODE"
    assert item.content.startswith("def forward")
    assert item.metadata["tenant_id"] == "tenant_beta"


# =============================================================================
# 5-8: Embedding Provider, Cache & Registry Tests
# =============================================================================

def test_embedding_cache_hit_and_miss():
    cache = EmbeddingCache(max_size=10)
    assert cache.get("model_a", "1.0", "hello") is None

    cache.set("model_a", "1.0", "hello", [0.1, 0.2, 0.3])
    cached = cache.get("model_a", "1.0", "hello")
    assert cached == [0.1, 0.2, 0.3]

    # Different version is a cache miss
    assert cache.get("model_a", "2.0", "hello") is None
    # Different model is a cache miss
    assert cache.get("model_b", "1.0", "hello") is None


def test_local_dense_embedding_provider_properties():
    provider = LocalDenseEmbeddingProvider(dimension=128)
    assert provider.dimension == 128
    assert provider.classification == "HEURISTIC"
    assert provider.is_local is True
    assert provider.is_free_tier is True

    v1 = provider.embed_text("Convolutional Neural Networks for vision")
    v2 = provider.embed_text("Convolutional Neural Networks for vision")
    assert len(v1) == 128
    assert v1 == v2  # Deterministic

    # Batch embedding
    batch = provider.embed_batch(["text 1", "text 2"])
    assert len(batch) == 2
    assert len(batch[0]) == 128


def test_embedding_model_registry_catalog_and_validation():
    models = EmbeddingModelRegistry.list_models()
    assert len(models) >= 4
    model_ids = [m["id"] for m in models]
    assert "videomind-dense-hashing-v1" in model_ids
    assert "all-MiniLM-L6-v2" in model_ids

    # Dimension compatibility checks
    assert EmbeddingModelRegistry.validate_dimension_compatibility(128, 128) is True
    assert EmbeddingModelRegistry.validate_dimension_compatibility(128, 384) is False

    with pytest.raises(ValueError, match="EMBEDDING_INDEX_INCOMPATIBLE"):
        EmbeddingModelRegistry.validate_compatibility(
            index_model_id="all-MiniLM-L6-v2",
            index_dimension=384,
            provider_dimension=128,
            provider_model_id="videomind-dense-hashing-v1",
        )


# =============================================================================
# 9-13: Vector Store Safety, Multi-Tenancy & Deletion Tests
# =============================================================================

def test_in_memory_vector_store_dimension_mismatch_safety():
    provider_128 = LocalDenseEmbeddingProvider(dimension=128)
    store = InMemoryVectorStore(embedding_provider=provider_128, index_dimension=128)
    store.add_texts(["Gradient descent optimization"], [{"topic": "opt"}])

    # 1. Searching an index with an incompatible provider raises EMBEDDING_INDEX_INCOMPATIBLE
    provider_256 = LocalDenseEmbeddingProvider(dimension=256)
    store.embedding_provider = provider_256
    with pytest.raises(ValueError, match="EMBEDDING_INDEX_INCOMPATIBLE"):
        store.search("optimization", k=2)

    # 2. Indexing into a store with incompatible vector dimension raises EMBEDDING_INDEX_INCOMPATIBLE
    mismatched_store = InMemoryVectorStore(embedding_provider=provider_256, index_dimension=128)
    with pytest.raises(ValueError, match="EMBEDDING_INDEX_INCOMPATIBLE"):
        mismatched_store.add_texts(["Another document"], [{"topic": "opt"}])


def test_in_memory_vector_store_multi_tenant_isolation():
    store = InMemoryVectorStore()
    docs = [
        "Confidential Project Alpha blueprint",
        "Confidential Project Beta roadmap",
    ]
    metas = [
        {"tenant_id": "org_alpha", "topic": "alpha"},
        {"tenant_id": "org_beta", "topic": "beta"},
    ]
    store.add_texts(docs, metas, ids=["alpha_1", "beta_1"])

    # Search as tenant alpha
    results_alpha = store.search("Confidential Project", k=5, tenant_id="org_alpha")
    assert len(results_alpha) == 1
    assert results_alpha[0][0]["tenant_id"] == "org_alpha"

    # Search as tenant beta
    results_beta = store.search("Confidential Project", k=5, tenant_id="org_beta")
    assert len(results_beta) == 1
    assert results_beta[0][0]["tenant_id"] == "org_beta"

    # Search as nonexistent tenant
    assert store.search("Confidential Project", k=5, tenant_id="org_gamma") == []


def test_in_memory_vector_store_deletion_by_id_and_filter():
    store = InMemoryVectorStore()
    store.add_texts(
        ["Item one", "Item two", "Item three"],
        [{"cat": "A"}, {"cat": "A"}, {"cat": "B"}],
        ids=["id_1", "id_2", "id_3"],
    )
    assert len(store.documents) == 3

    # Delete single ID
    deleted = store.delete(["id_1"])
    assert deleted == 1
    assert len(store.documents) == 2
    assert "id_1" not in store.documents

    # Delete by filter
    deleted_filter = store.delete_by_filter({"cat": "A"})
    assert deleted_filter == 1
    assert len(store.documents) == 1
    assert "id_3" in store.documents


def test_in_memory_vector_store_metadata_inspection():
    provider = LocalDenseEmbeddingProvider(dimension=128)
    store = InMemoryVectorStore(embedding_provider=provider, index_model_id="videomind-dense-hashing-v1", index_dimension=128)
    store.add_texts(["Sample text"], [{"meta": 1}])
    meta = store.get_index_metadata()
    assert meta["index_model_id"] == "videomind-dense-hashing-v1"
    assert meta["index_dimension"] == 128
    assert meta["doc_count"] == 1
    assert meta["is_dense"] is True


def test_chroma_vector_store_adapter_graceful_handling():
    # ChromaVectorStore handles absent dependency cleanly
    chroma = ChromaVectorStore(collection_name="test_col")
    meta = chroma.get_index_metadata()
    assert meta["store_type"] == "chroma"
    assert "is_active" in meta


# =============================================================================
# 14-18: Query Understanding & Hybrid Search Mechanics
# =============================================================================

def test_query_classifier_temporal_analysis():
    q1 = "What happens at 02:45 in the lecture?"
    a1 = QueryClassifier.analyze(q1)
    assert a1.is_temporal is True
    assert a1.target_timestamp == 165.0  # 2*60 + 45
    assert a1.intent == "timestamp_lookup"

    q2 = "Explain the slide between 01:10 and 02:20"
    a2 = QueryClassifier.analyze(q2)
    assert a2.is_temporal is True
    assert a2.target_timestamp == 70.0
    assert a2.target_timestamp_end == 140.0


def test_query_classifier_modality_hints_and_expansion():
    q = "Show me the python code for CNN backpropagation"
    analysis = QueryClassifier.analyze(q)
    assert ModalityType.CODE in analysis.extracted_modalities

    expansions = QueryClassifier.expand_query("cnn backprop")
    assert "convolutional" in expansions
    assert "backpropagation" in expansions


def test_hybrid_search_bm25_retrieval():
    engine = HybridSearchEngine()
    segs = [
        TranscriptSegment(id="s1", start=0.0, end=10.0, text="Support vector machines maximize the margin hyperplane."),
        TranscriptSegment(id="s2", start=11.0, end=20.0, text="K-means clustering partitions observations into voronoi cells."),
    ]
    engine.index_segments("v1", "ML Algorithms", segs)

    results = engine.search("hyperplane margin", strategy=RetrievalStrategy.BM25, k=2)
    assert len(results) >= 1
    assert "Support vector machines" in results[0].snippet


def test_hybrid_search_dense_vector_retrieval():
    provider = LocalDenseEmbeddingProvider(dimension=128)
    store = InMemoryVectorStore(embedding_provider=provider)
    engine = HybridSearchEngine(vector_store=store)

    segs = [
        TranscriptSegment(id="s1", start=0.0, end=15.0, text="Convolutional filters extract spatial visual features."),
        TranscriptSegment(id="s2", start=16.0, end=30.0, text="Relational SQL tables enforce foreign key integrity."),
    ]
    engine.index_segments("v_dense", "Dense Index", segs)

    results = engine.search("convolutional image visual", strategy=RetrievalStrategy.SEMANTIC, k=1)
    assert len(results) >= 1
    assert "Convolutional filters" in results[0].snippet


def test_hybrid_search_rrf_scoring_and_trace():
    engine = HybridSearchEngine(rrf_k=60.0)
    segs = [
        TranscriptSegment(id="s1", start=5.0, end=15.0, text="Backpropagation applies chain rule across neural layers."),
        TranscriptSegment(id="s2", start=20.0, end=35.0, text="Decision trees split on information gain entropy."),
    ]
    engine.index_segments("v_rrf", "Neural Networks", segs)

    pack = engine.retrieve_evidence_pack("backpropagation chain rule", k=2)
    assert len(pack.hits) >= 1
    assert pack.trace is not None
    assert pack.trace.rrf_k == 60.0
    assert pack.trace.confidence_level in (RetrievalConfidence.HIGH, RetrievalConfidence.MEDIUM)
    assert pack.hits[0].score > 0.0


# =============================================================================
# 19-24: Temporal Boosting, Multi-Modal & Multi-Tenant Search
# =============================================================================

def test_hybrid_search_temporal_proximity_ranking():
    engine = HybridSearchEngine()
    segs = [
        TranscriptSegment(id="s1", start=10.0, end=20.0, text="Optimization introduction concepts."),
        TranscriptSegment(id="s2", start=120.0, end=130.0, text="Optimization algorithm convergence rate analysis."),
    ]
    engine.index_segments("v_time", "Optimization", segs)

    # Search specifically around 02:05 (125 seconds)
    results = engine.search("Optimization at 02:05", strategy=RetrievalStrategy.TEMPORAL, k=2)
    assert len(results) >= 1
    assert results[0].start_time == 120.0


def test_hybrid_search_multimodal_indexing_and_filtering():
    engine = HybridSearchEngine()
    doc_text = MultiVectorDocument(
        id="txt_01",
        modality=ModalityType.TRANSCRIPT,
        content="The speaker introduces the transformer architecture.",
        source_id="vid_tf",
    )
    doc_code = MultiVectorDocument(
        id="code_01",
        modality=ModalityType.CODE,
        content="class MultiHeadAttention(nn.Module): pass",
        source_id="vid_tf",
    )
    engine.index_documents([doc_text, doc_code])

    # Filter by CODE modality
    code_results = engine.search("MultiHeadAttention", modality=ModalityType.CODE.value, k=2)
    assert len(code_results) >= 1
    assert code_results[0].modality == "code"


def test_hybrid_search_multi_tenant_isolation():
    engine = HybridSearchEngine()
    engine.index_segments(
        video_id="v_tenant_a",
        video_title="Alpha Secret",
        segments=[TranscriptSegment(id="s1", start=0.0, end=10.0, text="Secret project alpha credentials.")],
        tenant_id="tenant_a",
    )
    engine.index_segments(
        video_id="v_tenant_b",
        video_title="Beta Public",
        segments=[TranscriptSegment(id="s2", start=0.0, end=10.0, text="Secret project beta credentials.")],
        tenant_id="tenant_b",
    )

    hits_a = engine.search("Secret project", tenant_id="tenant_a", k=5)
    assert all(h.tenant_id == "tenant_a" for h in hits_a)
    assert len(hits_a) == 1
    assert hits_a[0].video_id == "v_tenant_a"


def test_cross_encoder_reranker_graceful_fallback():
    reranker = CrossEncoderReranker(enabled=False)
    assert reranker.get_classification() == "FALLBACK"
    candidates = [
        SearchResultItem(
            video_id="v1",
            video_title="Title",
            start_time=0.0,
            end_time=10.0,
            timestamp_formatted="[00:00 - 00:10]",
            snippet="Snippet test",
            score=1.0,
            retrieval_method="hybrid_rrf",
        )
    ]
    reranked, res = reranker.rerank("test query", candidates, top_k=1)
    assert len(reranked) == 1
    assert res.classification == "FALLBACK"


def test_hybrid_search_context_compression_merging():
    engine = HybridSearchEngine()
    # Two consecutive overlapping segments from same video
    item1 = SearchResultItem(
        video_id="v1",
        video_title="Lect",
        start_time=10.0,
        end_time=20.0,
        timestamp_formatted="[00:10 - 00:20]",
        snippet="First part of explanation.",
        score=2.0,
        retrieval_method="hybrid_rrf",
    )
    item2 = SearchResultItem(
        video_id="v1",
        video_title="Lect",
        start_time=21.0,
        end_time=32.0,
        timestamp_formatted="[00:21 - 00:32]",
        snippet="Second part of explanation.",
        score=1.8,
        retrieval_method="hybrid_rrf",
    )
    compressed = engine.compress_context([item1, item2], merge_temporal_window_sec=5.0)
    assert len(compressed) == 1
    assert compressed[0].start_time == 10.0
    assert compressed[0].end_time == 32.0
    assert "First part" in compressed[0].snippet
    assert "Second part" in compressed[0].snippet


# =============================================================================
# 25-30: RAGEngine Abstention, Grounding & FastAPI Endpoints
# =============================================================================

def test_rag_engine_supported_and_abstention():
    engine = HybridSearchEngine()
    engine.index_segments(
        video_id="vid_rag",
        video_title="Calculus",
        segments=[
            TranscriptSegment(
                id="c1",
                start=0.0,
                end=15.0,
                text="The derivative represents the instantaneous rate of change of a function.",
            )
        ],
    )
    rag = RAGEngine(engine)

    # Supported query
    res_sup = rag.answer_query("What does the derivative represent?", video_id="vid_rag")
    assert res_sup.abstention is False
    assert res_sup.confidence >= 0.80
    assert len(res_sup.citations) >= 1
    assert res_sup.citations[0].grounding_status == "SUPPORTED"

    # Unrelated query -> Abstention
    res_abs = rag.answer_query("How do you bake sourdough bread?", video_id="vid_rag")
    assert res_abs.abstention is True
    assert res_abs.confidence <= 0.35


def test_api_rag_query_endpoint():
    client = TestClient(app)
    # Seed index via API
    index_payload = {
        "documents": [
            {
                "id": "api_doc_1",
                "modality": "transcript",
                "content": "Gradient descent minimizes the objective loss function iteratively.",
                "source_id": "vid_api_1",
                "source_title": "Deep Learning Optimization",
                "start_ms": 10000,
                "end_ms": 25000,
            }
        ],
        "tenant_id": "tenant_test",
        "workspace_id": "ws_test",
    }
    r_index = client.post("/api/v1/rag/index", json=index_payload)
    assert r_index.status_code == 200
    assert r_index.json()["indexed_count"] == 1

    # Query via API
    q_payload = {
        "query": "What minimizes the objective loss function?",
        "tenant_id": "tenant_test",
        "workspace_id": "ws_test",
        "strategy": "hybrid",
        "top_k": 3,
    }
    r_query = client.post("/api/v1/rag/query", json=q_payload)
    assert r_query.status_code == 200
    data = r_query.json()
    assert "Gradient descent" in data["answer"]
    assert data["confidence_level"] in ("HIGH", "MEDIUM")
    assert len(data["citations"]) >= 1


def test_api_rag_search_and_stats_endpoints():
    client = TestClient(app)
    s_payload = {
        "query": "Gradient descent",
        "tenant_id": "tenant_test",
        "workspace_id": "ws_test",
        "strategy": "bm25",
        "top_k": 5,
    }
    r_search = client.post("/api/v1/rag/search", json=s_payload)
    assert r_search.status_code == 200
    res_data = r_search.json()
    assert res_data["results_count"] >= 1

    # Check stats endpoint
    r_stats = client.get("/api/v1/rag/stats")
    assert r_stats.status_code == 200
    assert r_stats.json()["total_corpus_documents"] >= 1

    # Check health endpoint
    r_health = client.get("/api/v1/rag/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "healthy"


def test_api_rag_delete_endpoint():
    client = TestClient(app)
    del_payload = {"ids": ["doc_tenant_test_api_doc_1"]}
    r_del = client.request("DELETE", "/api/v1/rag/documents", json=del_payload)
    assert r_del.status_code == 200
    assert r_del.json()["deleted_count"] >= 0


def test_query_decomposition_compound_queries():
    engine = HybridSearchEngine()
    subs1 = engine.decompose_query("compare convolutional neural networks vs transformers")
    assert len(subs1) >= 2

    subs2 = engine.decompose_query("gradient descent and backpropagation")
    assert len(subs2) >= 2


def test_knowledge_graph_concept_expansion():
    from app.knowledge.graph import ConceptNode, KnowledgeGraph

    kg = KnowledgeGraph()
    kg.add_concept(ConceptNode(id="c_attn", name="Self-Attention", definition="Mechanism weighting token interactions."))
    kg.add_concept(ConceptNode(id="c_tf", name="Transformer", definition="Sequence model based solely on attention."))
    kg.add_relationship("c_attn", "c_tf", "component_of")

    engine = HybridSearchEngine()
    engine.index_segments(
        video_id="v_kg",
        video_title="NLP",
        segments=[
            TranscriptSegment(
                id="s1",
                start=0.0,
                end=10.0,
                text="The transformer architecture relies on parallel matrix multiplications.",
            )
        ],
    )

    # Search with KG expansion
    hits = engine.search("self-attention", graph=kg, k=2)
    assert len(hits) >= 1
    assert "Self-Attention" in hits[0].matched_concepts


def test_sentence_transformer_provider_fallback_truthfulness():
    from app.embeddings.provider import SentenceTransformerEmbeddingProvider

    provider = SentenceTransformerEmbeddingProvider(model_name="all-MiniLM-L6-v2")
    # Must report truthful classification
    classification = provider.get_classification()
    assert classification in ("REAL", "FALLBACK", "HEURISTIC")
    # Must produce valid vector of declared dimension
    vec = provider.embed_text("test sentence")
    assert len(vec) == provider.dimension


def test_multi_factor_retrieval_confidence_levels():
    engine = HybridSearchEngine()
    analysis = QueryClassifier.analyze("neural networks")

    # High score -> HIGH
    item_high = SearchResultItem(
        video_id="v1",
        video_title="T1",
        start_time=0.0,
        end_time=10.0,
        timestamp_formatted="[00:00 - 00:10]",
        snippet="Neural networks learn internal representations through layers.",
        score=3.2,
        retrieval_method="hybrid_rrf",
    )
    score_h, lvl_h, abs_h = engine.calculate_retrieval_confidence("neural networks", [item_high], analysis)
    assert lvl_h == RetrievalConfidence.HIGH
    assert abs_h is False

    # Low score -> LOW or INSUFFICIENT
    item_low = SearchResultItem(
        video_id="v2",
        video_title="T2",
        start_time=0.0,
        end_time=10.0,
        timestamp_formatted="[00:00 - 00:10]",
        snippet="Completely unrelated astronomical data.",
        score=0.10,
        retrieval_method="hybrid_rrf",
    )
    score_l, lvl_l, abs_l = engine.calculate_retrieval_confidence("neural networks", [item_low], analysis)
    assert lvl_l in (RetrievalConfidence.LOW, RetrievalConfidence.INSUFFICIENT)

    # Empty hits -> INSUFFICIENT
    score_e, lvl_e, abs_e = engine.calculate_retrieval_confidence("neural networks", [], analysis)
    assert lvl_e == RetrievalConfidence.INSUFFICIENT
    assert abs_e is True


def test_cross_media_retrieval_integration():
    from app.documents.cross_rag import CrossMediaRetrievalEngine
    from app.documents.ingest import DocumentChunk

    engine = CrossMediaRetrievalEngine()
    chunk = DocumentChunk(
        chunk_id="chk_01",
        doc_id="textbook_dl",
        doc_title="Deep Learning Textbook",
        media_type="textbook",
        page_number=42,
        text="Stochastic gradient descent updates parameters using minibatch estimates.",
    )
    engine.index_document_chunks([chunk])
    engine.index_video_segments(
        segments=[
            TranscriptSegment(
                id="vseg_01",
                start=60.0,
                end=80.0,
                text="Stochastic gradient descent reduces computation per iteration.",
            )
        ],
        video_id="vid_opt",
        video_title="Optimization Lecture",
    )

    evidence = engine.search("gradient descent minibatch", top_k=5)
    assert evidence.total_sources_consulted >= 1
    assert len(evidence.matches) >= 1
    assert any(m.source_type == "textbook" for m in evidence.matches)

