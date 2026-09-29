"""
Comprehensive Integration Tests for Next-Gen Video Knowledge OS Subsystems:
- Multi-Video Collections & Cross-Video Contradiction Detection
- Knowledge Reasoning Engine (Comparison, Prerequisites, Study Plans)
- Learner Mastery Model & SM-2 Spaced Repetition Scheduler
- Study Material Pack Generator (Summaries, Flashcards, Mindmaps)
- Code Intelligence & Passive Static Analysis
- Mathematical Formula & LaTeX Derivation Intelligence
- Quota Management & Enterprise Data Governance (GDPR)
- FastAPI Next-Gen REST Endpoints
"""

import uuid
import pytest
from starlette.testclient import TestClient

from app.api.main import app
from app.collections.manager import MultiVideoCollectionManager
from app.reasoning.engine import ReasoningEngine
from app.learning.mastery import LearnerMasteryTracker
from app.learning.spaced_repetition import SpacedRepetitionScheduler
from app.learning.study_material import StudyMaterialGenerator
from app.code.analyzer import CodeIntelligenceEngine
from app.math.equations import MathematicalIntelligenceEngine
from app.admin.quotas import QuotaManager, QuotaExceededError
from app.security.governance import DataGovernanceManager
from app.memory.personal_memory import PersonalMemoryManager
from app.knowledge.graph import KnowledgeGraph
from app.domain.models import (
    ConceptNode,
    KnowledgeEdgeType,
    LearnerMasteryState,
    TranscriptSegment,
)
from app.retrieval.hybrid_search import HybridSearchEngine, SearchResultItem


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def auth_headers(client):
    uid = f"adm_{uuid.uuid4().hex[:6]}"
    reg = client.post(
        "/api/v1/auth/register",
        json={"email": f"{uid}@videomind.ai", "password": "AdminPassword123!", "role": "ADMIN"},
    )
    assert reg.status_code == 201
    data = reg.json()
    token = data["token"]
    user_id = data["user_id"]
    return {"Authorization": f"Bearer {token}", "User-Id": user_id}


@pytest.fixture
def student_auth_headers(client):
    uid = f"stu_{uuid.uuid4().hex[:6]}"
    reg = client.post(
        "/api/v1/auth/register",
        json={"email": f"{uid}@videomind.ai", "password": "StudentPassword123!", "role": "STUDENT"},
    )
    assert reg.status_code == 201
    data = reg.json()
    token = data["token"]
    user_id = data["user_id"]
    return {"Authorization": f"Bearer {token}", "User-Id": user_id}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Multi-Video Collections & Cross-Video Contradiction Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_multi_video_collections():
    search_engine = HybridSearchEngine()
    manager = MultiVideoCollectionManager(search_engine)

    coll = manager.create_collection(
        user_id="user_1",
        title="Deep Learning Curriculum",
        description="Collection of neural network lectures",
        video_ids=["vid_101", "vid_102"],
    )
    assert coll.id.startswith("col_")
    assert coll.title == "Deep Learning Curriculum"
    assert len(coll.video_ids) == 2

    # Add video
    manager.add_video_to_collection(coll.id, "vid_103")
    retrieved = manager.get_collection(coll.id)
    assert "vid_103" in retrieved.video_ids

    # Search collection
    results = manager.search_collection(coll.id, "neural networks")
    assert isinstance(results, list)


def test_cross_video_contradiction_detection():
    class MockContradictionSearch(HybridSearchEngine):
        def search(self, query, k=5, video_filter=None):
            if video_filter == "vid_lecture_2020":
                return [
                    SearchResultItem(
                        segment_id="s1",
                        video_id="vid_lecture_2020",
                        video_title="Lecture 2020",
                        timestamp_formatted="00:12",
                        start_time=12.0,
                        end_time=18.0,
                        text="Recurrent neural networks are optimal for sequence processing.",
                        score=0.9,
                        match_type="dense",
                        retrieval_method="dense",
                        snippet="Recurrent neural networks are optimal for sequence processing.",
                    )
                ]
            elif video_filter == "vid_lecture_2024":
                return [
                    SearchResultItem(
                        segment_id="s2",
                        video_id="vid_lecture_2024",
                        video_title="Lecture 2024",
                        timestamp_formatted="00:30",
                        start_time=30.0,
                        end_time=36.0,
                        text="Recurrent neural networks are no longer efficient and disproven by Transformers.",
                        score=0.92,
                        match_type="dense",
                        retrieval_method="dense",
                        snippet="Recurrent neural networks are no longer efficient and disproven by Transformers.",
                    )
                ]
            return []

    manager = MultiVideoCollectionManager(MockContradictionSearch())
    coll = manager.create_collection(
        user_id="u1",
        title="Evolution of NLP",
        video_ids=["vid_lecture_2020", "vid_lecture_2024"],
    )

    contradictions = manager.detect_cross_video_contradictions(coll.id, "Recurrent neural networks")
    assert len(contradictions) >= 1
    assert contradictions[0]["concept"] == "Recurrent neural networks"
    assert "vid_lecture_2020" in (contradictions[0]["video_a"], contradictions[0]["video_b"])


# ─────────────────────────────────────────────────────────────────────────────
# 2. Knowledge Reasoning Engine Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_knowledge_reasoning_comparison_and_prerequisites():
    kg = KnowledgeGraph()
    c1 = ConceptNode(id="c_python", name="Python Basics", definition="Core python language fundamentals", category="Programming", importance=0.9)
    c2 = ConceptNode(id="c_pytorch", name="PyTorch", definition="Deep learning framework for tensor computation", category="Framework", importance=0.85)
    c3 = ConceptNode(id="c_nlp", name="Transformers", definition="Self-attention based neural network model", category="Architecture", importance=0.95)

    kg.add_concept(c1)
    kg.add_concept(c2)
    kg.add_concept(c3)

    kg.add_edge(c1.id, c2.id, KnowledgeEdgeType.PREREQUISITE_OF, confidence=1.0)
    kg.add_edge(c2.id, c3.id, KnowledgeEdgeType.PREREQUISITE_OF, confidence=0.9)

    engine = ReasoningEngine(graph=kg)

    # 1. Compare concepts
    comp = engine.compare_concepts("Python Basics", "PyTorch")
    assert comp.concept_a == "Python Basics"
    assert comp.concept_b == "PyTorch"
    assert comp.direct_relationship is not None

    # 2. Prerequisite readiness (not ready)
    readiness_not_ready = engine.analyze_prerequisite_readiness("Transformers", mastered_concepts=[])
    assert readiness_not_ready.is_ready is False
    assert readiness_not_ready.readiness_percentage < 1.0

    # 3. Prerequisite readiness (ready)
    readiness_ready = engine.analyze_prerequisite_readiness("Transformers", mastered_concepts=["PyTorch", "Python Basics"])
    assert readiness_ready.is_ready is True
    assert readiness_ready.readiness_percentage == 100.0

    # 4. Study plan synthesis
    plan = engine.synthesize_study_plan("Transformers", mastered_concepts=["Python Basics"], available_time_minutes=90)
    assert plan["target_concept"] == "Transformers"
    assert len(plan["learning_steps"]) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 3. Learner Mastery & Spaced Repetition Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_learner_mastery_state_transitions():
    tracker = LearnerMasteryTracker()
    user_id = "user_mastery_test"
    concept_id = "c_backprop"

    # Initial state
    rec = tracker.get_or_create_record(user_id, concept_id, "Backpropagation")
    assert rec.mastery_state == LearnerMasteryState.NOT_STARTED

    # First attempt: correct -> state transitions to LEARNING
    rec = tracker.record_attempt(user_id, concept_id, "Backpropagation", is_correct=True)
    assert rec.mastery_state == LearnerMasteryState.LEARNING

    # Subsequent correct attempts -> PRACTICING then MASTERED
    tracker.record_attempt(user_id, concept_id, "Backpropagation", is_correct=True)
    tracker.record_attempt(user_id, concept_id, "Backpropagation", is_correct=True)
    tracker.record_attempt(user_id, concept_id, "Backpropagation", is_correct=True)
    rec = tracker.record_attempt(user_id, concept_id, "Backpropagation", is_correct=True)

    assert rec.attempt_count == 5
    assert rec.correct_count == 5
    assert rec.mastery_state == LearnerMasteryState.MASTERED


def test_sm2_spaced_repetition_scheduler():
    scheduler = SpacedRepetitionScheduler()
    user_id = "user_sm2_test"

    # Schedule item
    item = scheduler.schedule_initial_review(
        user_id=user_id,
        concept_id="c_entropy",
        question_text="What is Cross-Entropy loss?",
    )
    assert item.item_id.startswith("rev_")
    assert item.interval_days == 1
    assert item.repetition_number == 0

    # Process successful recall (quality 5)
    updated = scheduler.process_review_result(item.item_id, quality=5)
    assert updated.repetition_number == 1
    assert updated.interval_days == 1

    # Second successful recall (quality 4)
    updated = scheduler.process_review_result(item.item_id, quality=4)
    assert updated.repetition_number == 2
    assert updated.interval_days == 6

    # Test failure resets repetition (quality 1)
    failed = scheduler.process_review_result(item.item_id, quality=1)
    assert failed.repetition_number == 0
    assert failed.interval_days == 1


# ─────────────────────────────────────────────────────────────────────────────
# 4. Study Material Pack Generator Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_study_material_pack_generation():
    generator = StudyMaterialGenerator()
    segments = [
        TranscriptSegment(id="s1", start=0.0, end=10.0, text="Welcome to Linear Regression.", importance=0.8),
        TranscriptSegment(id="s2", start=10.0, end=25.0, text="We minimize the mean squared error loss.", importance=0.9),
    ]
    concepts = [
        ConceptNode(id="c_mse", name="Mean Squared Error", definition="The average squared difference between predictions and targets.", category="LossFunction", importance=0.9),
        ConceptNode(id="c_lr", name="Linear Regression", definition="A linear approach to modeling the relationship between variables.", category="Model", importance=0.85),
    ]

    pack = generator.generate_pack("vid_101", "Linear Regression 101", segments, concepts)
    assert pack.video_id == "vid_101"
    assert "Linear Regression 101" in pack.executive_summary
    assert len(pack.cheat_sheet) == 2
    assert len(pack.glossary) == 2
    assert len(pack.flashcards) == 2
    assert len(pack.mcqs) == 2
    assert "mindmap" in pack.mind_map_mermaid


# ─────────────────────────────────────────────────────────────────────────────
# 5. Code & Mathematical Intelligence Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_code_intelligence_passive_analysis():
    code_engine = CodeIntelligenceEngine()

    python_code = """
def compute_fibonacci(n):
    # compute fibonacci sequence
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
"""
    res = code_engine.detect_and_analyze(python_code, video_timestamp=45.5)
    assert res is not None
    assert res.language == "python"
    assert "compute_fibonacci" in res.functions
    assert res.complexity == "O(n)"
    assert res.explanation != ""


def test_mathematical_intelligence_structuring():
    math_engine = MathematicalIntelligenceEngine()

    gradient_descent_text = r"The update rule is \theta_{t+1} = \theta_t - \alpha \nabla L(\theta_t) for convex loss function."
    derivation = math_engine.detect_and_structure(gradient_descent_text, timestamp=120.0)

    assert derivation is not None
    assert "alpha" in derivation.variables or "α (alpha)" in derivation.variables
    assert "Loss surface is strictly or locally convex." in derivation.assumptions
    assert len(derivation.derivation_steps) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# 6. Quota Management & GDPR Data Governance Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_quota_management_tier_limits():
    qm = QuotaManager()
    quota = qm.get_or_create_quota("student_user_1", role="STUDENT")

    assert quota.role == "STUDENT"
    assert quota.max_video_duration_minutes == 120

    # Consume queries within limit
    qm.check_and_increment_query("student_user_1", role="STUDENT")
    assert quota.current_month_queries == 1

    # Over quota triggers QuotaExceededError
    quota.current_month_queries = quota.max_monthly_queries
    with pytest.raises(QuotaExceededError):
        qm.check_and_increment_query("student_user_1", role="STUDENT")


def test_data_governance_gdpr_lifecycle():
    mem_mgr = PersonalMemoryManager()
    gov_mgr = DataGovernanceManager(mem_mgr)

    user_id = "gdpr_test_user"
    mem_mgr.record_question(user_id, "How does backpropagation work?")

    class MockQuery:
        def filter(self, *args, **kwargs):
            return self

        def first(self):
            class MockUser:
                id = user_id
                email = "user@gdpr.org"
                role = "STUDENT"
            return MockUser()

        def all(self):
            return []

        def delete(self, *args, **kwargs):
            return 0

    class MockDB:
        def query(self, entity):
            return MockQuery()

        def delete(self, obj):
            pass

        def commit(self):
            pass

    export_pkg = gov_mgr.export_user_data(user_id, MockDB())
    assert export_pkg["user"]["id"] == user_id
    assert len(export_pkg["learning_memory"]["recent_questions"]) == 1

    # Right to be forgotten (Erasure)
    del_res = gov_mgr.delete_user_data(user_id, MockDB())
    assert del_res["status"] == "PURGED"
    assert len(mem_mgr.get_memory(user_id).recent_questions) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 7. Next-Gen FastAPI HTTP Endpoints Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_api_collections_lifecycle(client, auth_headers):
    # 1. Create collection
    resp = client.post(
        "/api/v1/collections",
        json={"title": "ML Foundation", "description": "Core ML Lectures", "video_ids": ["v1", "v2"]},
        headers={"Authorization": auth_headers["Authorization"]},
    )
    assert resp.status_code == 200
    col_data = resp.json()
    assert col_data["title"] == "ML Foundation"
    col_id = col_data["id"]

    # 2. List collections
    list_resp = client.get("/api/v1/collections", headers={"Authorization": auth_headers["Authorization"]})
    assert list_resp.status_code == 200
    assert any(c["id"] == col_id for c in list_resp.json()["collections"])

    # 3. Add video to collection
    add_resp = client.post(
        f"/api/v1/collections/{col_id}/videos",
        json={"video_id": "v3"},
        headers={"Authorization": auth_headers["Authorization"]},
    )
    assert add_resp.status_code == 200
    assert "v3" in add_resp.json()["video_ids"]


def test_api_reasoning_and_learning_endpoints(client, auth_headers):
    # Concept comparison
    comp_resp = client.post(
        "/api/v1/reasoning/compare",
        json={"concept_a": "CNN", "concept_b": "RNN"},
        headers={"Authorization": auth_headers["Authorization"]},
    )
    assert comp_resp.status_code == 200
    assert comp_resp.json()["concept_a"] == "CNN"

    # Prerequisite check
    prereq_resp = client.post(
        "/api/v1/reasoning/prerequisites",
        json={"target_concept": "Attention", "mastered_concepts": ["Linear Algebra"]},
        headers={"Authorization": auth_headers["Authorization"]},
    )
    assert prereq_resp.status_code == 200
    assert "is_ready" in prereq_resp.json()

    # Code intelligence endpoint
    code_resp = client.post(
        "/api/v1/code/analyze",
        json={"text": "def calculate_loss(y_pred, y_true): return (y_pred - y_true)**2"},
    )
    assert code_resp.status_code == 200
    assert code_resp.json()["is_code"] is True

    # Mathematical intelligence endpoint
    math_resp = client.post(
        "/api/v1/math/equations",
        json={"text": "gradient descent formula L(theta) = sum(y - hat_y)^2"},
    )
    assert math_resp.status_code == 200
    assert math_resp.json()["is_math"] is True


def test_api_mastery_and_spaced_repetition(client, auth_headers):
    # Calibrate mastery
    cal_resp = client.post(
        "/api/v1/learning/mastery/calibrate",
        json={"concept_id": "c_transformer", "concept_name": "Transformer", "is_correct": True},
        headers={"Authorization": auth_headers["Authorization"]},
    )
    assert cal_resp.status_code == 200
    assert cal_resp.json()["correct_count"] == 1

    # Schedule review
    sch_resp = client.post(
        "/api/v1/learning/reviews/schedule",
        json={"concept_id": "c_transformer", "question_text": "What is self-attention?"},
        headers={"Authorization": auth_headers["Authorization"]},
    )
    assert sch_resp.status_code == 200
    rev_id = sch_resp.json()["item_id"]

    # Grade review
    grade_resp = client.post(
        f"/api/v1/learning/reviews/{rev_id}/grade",
        json={"quality": 4},
        headers={"Authorization": auth_headers["Authorization"]},
    )
    assert grade_resp.status_code == 200
    assert grade_resp.json()["repetition_number"] == 1


def test_api_admin_and_governance(client, auth_headers, student_auth_headers):
    # Admin telemetry (ADMIN only)
    admin_resp = client.get("/api/v1/admin/telemetry", headers={"Authorization": auth_headers["Authorization"]})
    assert admin_resp.status_code == 200
    assert "timestamp" in admin_resp.json()

    # Admin telemetry blocked for STUDENT
    student_resp = client.get("/api/v1/admin/telemetry", headers={"Authorization": student_auth_headers["Authorization"]})
    assert student_resp.status_code == 403

    # User Quota check
    student_id = student_auth_headers["User-Id"]
    quota_resp = client.get(f"/api/v1/admin/quotas/{student_id}", headers={"Authorization": student_auth_headers["Authorization"]})
    assert quota_resp.status_code == 200
    assert quota_resp.json()["user_id"] == student_id

    # GDPR Privacy Export
    export_resp = client.get("/api/v1/governance/export", headers={"Authorization": auth_headers["Authorization"]})
    assert export_resp.status_code == 200
    assert "export_timestamp" in export_resp.json()
