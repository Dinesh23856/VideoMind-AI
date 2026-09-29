"""Tests for Active Learning, Multimodal Uncertainty, Adaptive Compute, Explanations, and Counterfactuals."""

import pytest
from app.active.uncertainty import ModalityUncertaintyVector, MultimodalUncertaintyEngine
from app.active.controller import AdaptiveComputeAction, AdaptiveComputeController
from app.reasoning.explainer import EvidenceBasedExplainer
from app.reasoning.counterfactual import CounterfactualSimulator
from app.domain.models import ConceptNode, KnowledgeEdgeType, RelationshipEdge


def test_multimodal_uncertainty_engine():
    engine = MultimodalUncertaintyEngine()

    u_vec = engine.evaluate_segment_uncertainty(
        transcription_confidence=0.98,
        ocr_confidence=0.40,
        frame_blur_metric=0.85,
        retrieval_top1_score=0.9,
        retrieval_top2_score=0.4,
        unsupported_claims_ratio=0.1,
    )

    assert u_vec.audio_uncertainty < 0.05
    assert u_vec.visual_uncertainty == 0.85
    assert u_vec.ocr_uncertainty == 0.60
    assert u_vec.primary_bottleneck == "visual"
    assert u_vec.composite_uncertainty > 0.30


def test_adaptive_compute_controller():
    controller = AdaptiveComputeController(uncertainty_threshold=0.35, max_escalations_per_video=3)

    # Low uncertainty -> NO_ACTION
    low_u = ModalityUncertaintyVector(visual_uncertainty=0.1, ocr_uncertainty=0.1)
    dec1 = controller.decide_action(low_u)
    assert dec1.action == AdaptiveComputeAction.NO_ACTION

    # High visual uncertainty -> RESAMPLE_FRAMES
    high_vis = ModalityUncertaintyVector(visual_uncertainty=0.9)
    dec2 = controller.decide_action(high_vis)
    assert dec2.action == AdaptiveComputeAction.RESAMPLE_FRAMES
    assert dec2.target_modality == "visual"
    assert dec2.allowed is True


def test_evidence_based_explainer():
    explainer = EvidenceBasedExplainer()

    citations = [
        {"source_id": "vid_lec1", "timestamp": 124.0, "snippet": "ReLU derivative is 1 for positive inputs.", "confidence": 0.98, "trust": 0.95},
        {"source_id": "vid_lec1", "timestamp": 180.0, "snippet": "Non-saturating gradients prevent vanishing.", "confidence": 0.92, "trust": 0.90},
    ]

    exp = explainer.explain_answer(
        question="Why does ReLU solve vanishing gradients?",
        answer="ReLU maintains constant derivative of 1 for positive inputs, preventing exponential decay in backprop.",
        supporting_citations=citations,
        uncertainty_score=0.08,
    )

    assert len(exp.premises) == 2
    assert exp.confidence_interval[0] > 0.8
    assert "vid_lec1" in exp.premises[0].source_id
    assert "grounded in 2 distinct evidence anchors" in exp.audit_trail


def test_counterfactual_simulator():
    c_math = ConceptNode(id="c_math", name="Linear Algebra", definition="Vector spaces and matrices")
    c_calc = ConceptNode(id="c_calc", name="Multivariate Calculus", definition="Partial derivatives and gradients")
    c_backprop = ConceptNode(id="c_backprop", name="Backpropagation", definition="Gradient computation in DAGs")
    c_transformers = ConceptNode(id="c_trans", name="Transformers", definition="Self-attention networks")

    edges = [
        RelationshipEdge(source_id="c_math", target_id="c_backprop", relation=KnowledgeEdgeType.PREREQUISITE_OF),
        RelationshipEdge(source_id="c_calc", target_id="c_backprop", relation=KnowledgeEdgeType.PREREQUISITE_OF),
        RelationshipEdge(source_id="c_backprop", target_id="c_trans", relation=KnowledgeEdgeType.PREREQUISITE_OF),
    ]

    sim = CounterfactualSimulator([c_math, c_calc, c_backprop, c_transformers], edges)

    # If Backpropagation is omitted, Transformers is immediately lost
    impact = sim.simulate_omission("c_backprop")
    assert "Transformers" in impact.unlocked_concepts_lost
    assert impact.impact_score > 0.2
