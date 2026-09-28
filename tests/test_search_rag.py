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
