from app.domain.models import TranscriptSegment, TutorMode
from app.rag.engine import RAGEngine
from app.retrieval.hybrid_search import HybridSearchEngine
from app.tutor.engine import AITutorEngine


def test_hybrid_search_and_rag():
    engine = HybridSearchEngine()
    segments = [
        TranscriptSegment(
            id="seg_01",
            start=12.0,
            end=30.0,
            text="Cross-validation prevents overfitting by evaluating on held-out folds.",
        ),
        TranscriptSegment(
            id="seg_02",
            start=32.0,
            end=65.0,
            text="Gradient descent updates model parameters using the negative gradient of the loss.",
        ),
    ]
    engine.index_segments("vid_test", "Machine Learning 101", segments)

    # Search for overfitting
    hits = engine.search("overfitting cross-validation", k=3)
    assert len(hits) >= 1
    assert "Cross-validation" in hits[0].snippet
    assert "[00:12 - 00:30]" in hits[0].timestamp_formatted

    # Test RAG synthesis
    rag = RAGEngine(engine)
    rag_resp = rag.answer_query("How does cross-validation prevent overfitting?", video_id="vid_test")
    assert rag_resp.confidence > 0.7
    assert len(rag_resp.citations) >= 1
    assert "00:12" in rag_resp.citations[0].formatted

    # Test AI Tutor modes and progressive hints
    tutor = AITutorEngine(rag)
    tutor_resp = tutor.interact("Explain cross-validation", mode=TutorMode.BEGINNER, video_id="vid_test")
    assert "Analogy" in tutor_resp.tutor_reply
    assert len(tutor_resp.suggested_follow_ups) > 0

    hint_resp = tutor.interact("Give me a hint on cross-validation", mode=TutorMode.EXAM, context_hints_given=0)
    assert hint_resp.hint is not None
    assert "Hint 1" in hint_resp.hint


def test_dense_vector_store_integration():
    from app.embeddings.provider import LocalDenseEmbeddingProvider
    from app.retrieval.vector_store import InMemoryVectorStore

    provider = LocalDenseEmbeddingProvider(dimension=128)
    store = InMemoryVectorStore(embedding_provider=provider)
    texts = [
        "Backpropagation computes gradients of loss with respect to neural network weights.",
        "Convolutional layers apply spatial filters over input image feature maps.",
        "Relational databases use B-tree indexing for logarithmic lookups.",
    ]
    metas = [{"topic": "backprop"}, {"topic": "cnn"}, {"topic": "db"}]
    doc_ids = store.add_texts(texts, metas)
    assert len(doc_ids) == 3

    # Semantic search with query
    results = store.search("gradient backpropagation network", k=1)
    assert len(results) >= 1
    assert results[0][0]["topic"] == "backprop"
    assert results[0][1] > 0.20
