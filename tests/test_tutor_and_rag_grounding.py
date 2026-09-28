"""Verification of RAG Grounding (4 question types), AI Tutor (8 modes + hint ladder), and Course Generation."""

import pytest
from app.domain.models import ConceptNode, QuizQuestion, TranscriptSegment, TutorMode
from app.knowledge.graph import KnowledgeGraph
from app.learning.course import CourseGenerator, LearningPathEngine
from app.rag.engine import RAGEngine
from app.retrieval.hybrid_search import HybridSearchEngine
from app.tutor.engine import AITutorEngine


@pytest.fixture
def indexed_engine():
    search_eng = HybridSearchEngine()
    segments = [
        TranscriptSegment(
            id="seg_0001",
            start=0.0,
            end=12.5,
            speaker="Professor",
            text="Today we cover Gradient Descent. The learning rate alpha controls step size along the negative gradient of the loss function.",
            confidence=0.98,
            concepts=["Gradient Descent", "Learning Rate"],
        ),
        TranscriptSegment(
            id="seg_0002",
            start=13.0,
            end=28.0,
            speaker="Professor",
            text="If alpha is too large, the optimization diverges. If alpha is too small, convergence is exceedingly slow.",
            confidence=0.95,
            concepts=["Learning Rate", "Divergence"],
        ),
        TranscriptSegment(
            id="seg_0003",
            start=29.0,
            end=45.0,
            speaker="Professor",
            text="Backpropagation calculates the analytical gradient of the loss with respect to all layer weights using the multivariable chain rule.",
            confidence=0.99,
            concepts=["Backpropagation", "Chain Rule"],
        ),
    ]
    search_eng.index_segments(
        video_id="vid_opt_101",
        video_title="Optimization in Neural Networks",
        segments=segments,
    )
    return search_eng


def test_rag_grounding_four_question_types(indexed_engine):
    rag = RAGEngine(indexed_engine)

    # Question A: Fully Supported
    res_a = rag.answer_query("What does the learning rate alpha control?", video_id="vid_opt_101")
    assert res_a.confidence >= 0.85
    assert len(res_a.citations) > 0
    assert "00:00" in res_a.citations[0].formatted
    assert res_a.hallucination_risk == "LOW"
    assert "step size" in res_a.answer.lower()

    # Question B: Partially Supported
    res_b = rag.answer_query("What happens when alpha is too large or too small?", video_id="vid_opt_101")
    assert res_b.confidence >= 0.80
    assert len(res_b.citations) > 0
    assert any("diverg" in c.snippet.lower() for c in res_b.citations)

    # Question C: Unsupported (Not covered in lecture)
    res_c = rag.answer_query("How do quantum supercomputers factor prime numbers using Shor's algorithm?", video_id="vid_opt_101")
    assert res_c.confidence <= 0.3
    assert len(res_c.citations) == 0
    assert res_c.hallucination_risk == "HIGH"
    assert "no relevant video content" in res_c.answer.lower()

    # Question D: Contradictory Premise ("Why does a large alpha always guarantee faster convergence?")
    res_d = rag.answer_query("Why does a large alpha always guarantee faster convergence?", video_id="vid_opt_101")
    assert len(res_d.citations) > 0
    # The grounded evidence clearly notes that large alpha diverges
    assert any("diverg" in c.snippet.lower() for c in res_d.citations)


def test_ai_tutor_all_eight_modes(indexed_engine):
    rag = RAGEngine(indexed_engine)
    tutor = AITutorEngine(rag)

    modes = [
        TutorMode.BEGINNER,
        TutorMode.INTERMEDIATE,
        TutorMode.ADVANCED,
        TutorMode.EXPERT,
        TutorMode.INTERVIEW,
        TutorMode.EXAM,
        TutorMode.RESEARCH,
        TutorMode.DEVELOPER,
    ]

    responses = {}
    for m in modes:
        res = tutor.interact(
            user_message="Explain gradient descent",
            mode=m,
            video_id="vid_opt_101",
        )
        assert res.mode == m
        assert len(res.tutor_reply) > 50
        assert len(res.suggested_follow_ups) > 0
        responses[m.value.lower()] = res.tutor_reply

    # Verify that modes produce pedagogically distinct personas and outputs
    assert "analogy" in responses["beginner"].lower() or "break this down" in responses["beginner"].lower()
    assert "```python" in responses["developer"]
    assert "interview" in responses["interview"].lower()
    assert "exam" in responses["exam"].lower()
    assert "asymptotic" in responses["advanced"].lower() or "theoretical" in responses["advanced"].lower()


def test_ai_tutor_hint_progression_ladder(indexed_engine):
    rag = RAGEngine(indexed_engine)
    tutor = AITutorEngine(rag)

    # Step 1: Hint 1 (Nudge)
    h1 = tutor.interact("I am stuck on this problem, give me a hint", mode=TutorMode.INTERMEDIATE, context_hints_given=0)
    assert h1.hint is not None
    assert "Hint 1" in h1.hint
    assert not h1.solution_revealed

    # Step 2: Hint 2 (Direction)
    h2 = tutor.interact("Still need help", mode=TutorMode.INTERMEDIATE, context_hints_given=1)
    assert h2.hint is not None
    assert "Hint 2" in h2.hint
    assert not h2.solution_revealed

    # Step 3: Hint 3 (Framework)
    h3 = tutor.interact("I need another clue", mode=TutorMode.INTERMEDIATE, context_hints_given=2)
    assert h3.hint is not None
    assert "Hint 3" in h3.hint
    assert not h3.solution_revealed

    # Step 4: Student submits solution attempt
    sol = tutor.interact(
        "Optimization problem",
        mode=TutorMode.INTERMEDIATE,
        student_solution="We take the derivative and subtract alpha times the gradient.",
    )
    assert sol.solution_revealed
    assert "Review of Your Solution" in sol.tutor_reply


def test_knowledge_graph_influences_search(indexed_engine):
    # Construct a KnowledgeGraph and verify concept expansion
    kg = KnowledgeGraph()
    c1 = ConceptNode(id="c_gd", name="Gradient Descent", definition="Iterative first-order optimization.")
    c2 = ConceptNode(id="c_lr", name="Learning Rate", definition="Step size hyperparameter.")
    kg.add_node(c1)
    kg.add_node(c2)
    kg.add_edge("c_gd", "c_lr", "USES_HYPERPARAMETER")

    # Search with graph concept expansion
    results_graph = indexed_engine.search("Gradient Descent", k=3, graph=kg)

    assert len(results_graph) > 0
    assert "Gradient Descent" in results_graph[0].matched_concepts


def test_course_and_adaptive_learning():
    cg = CourseGenerator()
    videos = [
        {"id": "v1", "title": "Linear Algebra Basics", "duration": 1800.0, "concepts": ["Vectors", "Matrices"]},
        {"id": "v2", "title": "Calculus for Deep Learning", "duration": 2400.0, "concepts": ["Derivatives", "Chain Rule"]},
        {"id": "v3", "title": "Neural Networks 101", "duration": 3600.0, "concepts": ["Perceptrons", "Activation Functions"]},
    ]
    course = cg.compile_course("Deep Learning Foundations", "Comprehensive curriculum", videos)
    assert course.title == "Deep Learning Foundations"
    assert len(course.modules) >= 1
    assert course.estimated_duration_hours >= 2.0

    # Test Adaptive Learning Path Engine
    engine = LearningPathEngine()
    from app.domain.models import LearnerProfile
    profile = LearnerProfile(
        user_id="learner_1",
        mastered_concepts=["Vectors"],
        weak_concepts=["Chain Rule"],
    )
    concepts = [
        ConceptNode(id="1", name="Vectors", definition="Vector spaces and geometric linear combinations."),
        ConceptNode(id="2", name="Chain Rule", definition="Multivariable derivative composition."),
        ConceptNode(id="3", name="Perceptrons", definition="Binary threshold classification neuron."),
    ]
    path = engine.generate_personalized_path(profile, concepts)
    assert len(path) == 3
    # Weak concept prioritized
    assert path[0]["concept"] == "Chain Rule"
    assert path[0]["status"] == "NEEDS_REVISION"
    assert path[0]["priority"] == 1

