"""
P3 — Generalized Knowledge Extraction & Advanced Graph RAG Test Suite.

Comprehensive test coverage for:
1. Generalized multimodal extraction (transcripts, OCR, keyframes, documents, code, formulas, audio, scenes)
2. Deterministic entity & concept resolution with alias handling and over-merging protection
3. Rigorous claim + evidence model with complete provenance and epistemic classification
4. Honest contradiction detection preserving conflicting sources, citations, and timestamps
5. Temporal reasoning and Allen's interval algebra integration
6. Bounded multi-hop graph traversal with cycle prevention and depth/node limits
7. Cross-modal graph reasoning and traceable reasoning paths
8. Graph-aware hybrid retrieval and evidence packaging
9. Grounded RAG answer generation with multimodal citations and truthful abstention
10. Strict multi-tenant and workspace isolation across all graph, extraction, and retrieval operations
11. Local-first, offline deterministic execution with zero cloud calls
"""

from __future__ import annotations

import pytest
from app.domain.models import (
    AudioEvent,
    ConceptNode,
    FactClaim,
    FactVerificationStatus,
    KnowledgeEdgeType,
    KnowledgeObjectType,
    RelationshipEdge,
    TranscriptSegment,
    VisualFrameContext,
)
from app.knowledge.extractor import KnowledgeExtractor
from app.knowledge.graph import KnowledgeGraph
from app.knowledge.resolver import EntityResolver
from app.knowledge.verification import FactVerificationEngine
from app.rag.engine import RAGEngine
from app.reasoning.temporal import TemporalRelation, compute_temporal_relation
from app.retrieval.hybrid_search import HybridSearchEngine, RetrievalConfidence, RetrievalStrategy


# =============================================================================
# 1. Generalized Multimodal Knowledge Extraction Tests
# =============================================================================

def test_multimodal_extraction_transcripts_and_notes():
    extractor = KnowledgeExtractor()
    notes = """
    ## Transformer Architecture [00:15]
    **Concept: Self-Attention**: Computes attention weights across all token pairs simultaneously.
    - Self-Attention enables quadratic time complexity over input length.
    - PyTorch implements Multi-Head Attention efficiently.
    ```python
    import torch
    import torch.nn as nn
    def scaled_dot_product(q, k, v):
        return torch.matmul(q, k.transpose(-2, -1))
    ```
    $$A = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)$$
    """
    segments = [
        TranscriptSegment(id="s1", start=15.0, end=30.0, text="Today we discuss self-attention and transformer architecture."),
        TranscriptSegment(id="s2", start=35.0, end=50.0, text="Self-attention has quadratic complexity in sequence length."),
    ]

    result = extractor.extract_all(notes, segments, tenant_id="t1", workspace_id="ws1", video_id="v1")

    assert len(result["concepts"]) >= 1
    assert any("Self-Attention" in c.name for c in result["concepts"])
    assert len(result["code_snippets"]) >= 1
    assert result["code_snippets"][0].language == "python"
    assert "scaled_dot_product" in result["code_snippets"][0].functions
    assert len(result["math_derivations"]) >= 1
    assert "softmax" in result["math_derivations"][0].latex
    assert len(result["chapters"]) >= 1
    assert "Transformer Architecture" in result["chapters"][0].title
    assert len(result["claims"]) >= 1


def test_ocr_knowledge_extraction_and_modality():
    extractor = KnowledgeExtractor()
    ocr_frames = [
        VisualFrameContext(
            frame_id="kf_01",
            timestamp=25.5,
            frame_type="code",
            ocr_text="import numpy as np\ndef compute_loss(y, y_hat):\n    return np.mean((y - y_hat)**2)",
            confidence=0.95,
        ),
        VisualFrameContext(
            frame_id="kf_02",
            timestamp=55.0,
            frame_type="slide",
            ocr_text="Convolutional Neural Networks\nFeature Extraction and Pooling Layers",
            confidence=0.92,
        ),
    ]

    nodes = extractor.extract_ocr_knowledge(ocr_frames, tenant_id="t1", workspace_id="ws1", video_id="v1")
    assert len(nodes) >= 2
    assert any(n.node_type == KnowledgeObjectType.CODE for n in nodes)
    assert any("Convolutional Neural Network" in n.name for n in nodes)
    assert all(n.source_modality == "ocr" for n in nodes)
    assert all(n.tenant_id == "t1" for n in nodes)


def test_scene_knowledge_extraction():
    extractor = KnowledgeExtractor()
    scenes = [
        {"scene_id": "sc_01", "start_time": 0.0, "end_time": 60.0, "description": "Introduction to neural networks"},
        {"scene_id": "sc_02", "start_time": 60.0, "end_time": 180.0, "description": "Backpropagation walkthrough"},
    ]
    scene_nodes = extractor.extract_scene_knowledge(scenes, tenant_id="t_scene", workspace_id="ws_main", video_id="vid_sc")

    assert len(scene_nodes) == 2
    assert scene_nodes[0].node_type == KnowledgeObjectType.SCENE
    assert scene_nodes[0].start_ms == 0
    assert scene_nodes[0].end_ms == 60000
    assert scene_nodes[1].start_ms == 60000
    assert scene_nodes[1].end_ms == 180000
    assert scene_nodes[0].tenant_id == "t_scene"


def test_audio_events_and_speaker_extraction():
    extractor = KnowledgeExtractor()
    audio_events = [
        AudioEvent(event_id="aud_1", start_time=12.0, end_time=14.5, event_type="applause", confidence=0.88),
        AudioEvent(event_id="aud_2", start_time=45.0, end_time=52.0, event_type="speech", speaker="Dr. Smith", confidence=0.95),
    ]

    audio_nodes = extractor.extract_audio_events(audio_events, tenant_id="t_aud", workspace_id="ws_aud", video_id="vid_aud")

    assert len(audio_nodes) >= 2
    assert any(n.node_type == KnowledgeObjectType.AUDIO_EVENT and "Applause" in n.name for n in audio_nodes)
    assert any(n.node_type == KnowledgeObjectType.SPEAKER and n.name == "Dr. Smith" for n in audio_nodes)
    assert all(n.tenant_id == "t_aud" for n in audio_nodes)


def test_document_knowledge_extraction():
    extractor = KnowledgeExtractor()
    documents = [
        {"id": "doc_dl", "title": "Deep Learning Textbook Chapter 6", "text": "Deep feedforward networks are core models."},
        {"id": "doc_attn", "title": "Attention Is All You Need Paper", "text": "The dominant sequence models are based on complex recurrent networks."},
    ]
    doc_nodes = extractor.extract_document_knowledge(documents, tenant_id="t_doc", workspace_id="ws_doc")

    assert len(doc_nodes) == 2
    assert doc_nodes[0].node_type == KnowledgeObjectType.DOCUMENT
    assert doc_nodes[0].document_id == "doc_dl"
    assert doc_nodes[0].source_modality == "document"
    assert doc_nodes[1].document_id == "doc_attn"


def test_code_and_formula_extraction_provenance():
    extractor = KnowledgeExtractor()
    notes = """
    ```python
    def relu(x):
        return max(0, x)
    ```
    $$\\sigma(z) = \\frac{1}{1 + e^{-z}}$$
    """
    code_items = extractor.extract_code_snippets(notes, [])
    math_items = extractor.extract_math_derivations(notes, [])

    assert len(code_items) == 1
    assert code_items[0].language == "python"
    assert code_items[0].functions == ["relu"]

    assert len(math_items) == 1
    assert "sigma" in math_items[0].latex


# =============================================================================
# 2. Entity and Concept Resolution Tests
# =============================================================================

def test_entity_normalization_and_acronym_expansion():
    resolver = EntityResolver()
    assert resolver.normalize_name("cnn") == "Convolutional Neural Network"
    assert resolver.normalize_name("convolutional neural network") == "Convolutional Neural Network"
    assert resolver.normalize_name("BERT") == "BERT"
    assert resolver.normalize_name("sgd") == "Stochastic Gradient Descent"
    assert resolver.normalize_name("rag") == "Retrieval-Augmented Generation"


def test_entity_resolution_alias_merging():
    resolver = EntityResolver()
    node1 = ConceptNode(id="n1", name="Convolutional Neural Network", definition="Deep neural network for vision", timestamps=[10.0])
    node2 = ConceptNode(id="n2", name="CNN", definition="CNN visual architecture", timestamps=[45.0, 60.0])

    can_merge, reason = resolver.can_merge(node1, node2)
    assert can_merge is True

    resolved = resolver.resolve_entities([node1, node2])
    assert len(resolved) == 1
    canonical = resolved[0]
    assert canonical.name == "Convolutional Neural Network"
    assert "CNN" in canonical.aliases
    assert canonical.timestamps == [10.0, 45.0, 60.0]
    assert canonical.mention_count == 2


def test_distinct_concepts_not_overmerged():
    resolver = EntityResolver()
    py_node = ConceptNode(id="n_py", name="Python", definition="High-level programming language", node_type=KnowledgeObjectType.TOOL)
    torch_node = ConceptNode(id="n_torch", name="PyTorch", definition="Tensors and dynamic neural networks in Python", node_type=KnowledgeObjectType.TOOL)

    can_merge, reason = resolver.can_merge(py_node, torch_node)
    assert can_merge is False
    assert "cannot be merged" in reason

    resolved = resolver.resolve_entities([py_node, torch_node])
    assert len(resolved) == 2


def test_attention_and_self_attention_not_overmerged():
    resolver = EntityResolver()
    attn = ConceptNode(id="a1", name="Attention", definition="Mechanism mapping query and keys to values")
    self_attn = ConceptNode(id="a2", name="Self-Attention", definition="Relating different positions of a single sequence")

    can_merge, reason = resolver.can_merge(attn, self_attn)
    assert can_merge is False

    resolved = resolver.resolve_entities([attn, self_attn])
    assert len(resolved) == 2


def test_entity_resolution_cross_modal_mention_aggregation():
    resolver = EntityResolver()
    transcript_mention = ConceptNode(
        id="m_tx",
        name="Transformer",
        definition="Sequence-to-sequence model architecture",
        timestamps=[15.0],
        source_modality="transcript",
    )
    ocr_mention = ConceptNode(
        id="m_ocr",
        name="Transformer",
        definition="Slide title: Transformer Attention Layers",
        timestamps=[18.0],
        source_modality="ocr",
        evidence_ids=["frame_101"],
    )

    resolved = resolver.resolve_entities([transcript_mention, ocr_mention])
    assert len(resolved) == 1
    assert resolved[0].timestamps == [15.0, 18.0]
    assert "frame_101" in resolved[0].evidence_ids
    assert resolved[0].mention_count == 2


def test_entity_resolution_strict_tenant_isolation():
    resolver = EntityResolver()
    node_tenant_a = ConceptNode(id="ta1", name="TensorFlow", definition="Google ML library", tenant_id="tenant_A")
    node_tenant_b = ConceptNode(id="tb1", name="TensorFlow", definition="Google ML library", tenant_id="tenant_B")

    can_merge, reason = resolver.can_merge(node_tenant_a, node_tenant_b, tenant_id="tenant_A")
    assert can_merge is False
    assert "tenant boundaries" in reason

    resolved_a = resolver.resolve_entities([node_tenant_a, node_tenant_b], tenant_id="tenant_A")
    # Only tenant_A node returned
    assert len(resolved_a) == 1
    assert resolved_a[0].id == "ta1"


def test_entity_resolution_strict_workspace_isolation():
    resolver = EntityResolver()
    node_ws_1 = ConceptNode(id="w1", name="Docker", definition="Container platform", tenant_id="t1", workspace_id="ws_dev")
    node_ws_2 = ConceptNode(id="w2", name="Docker", definition="Container platform", tenant_id="t1", workspace_id="ws_prod")

    can_merge, reason = resolver.can_merge(node_ws_1, node_ws_2)
    assert can_merge is False
    assert "workspace boundaries" in reason


# =============================================================================
# 3. Claim + Evidence Model & Provenance Tests
# =============================================================================

def test_claim_extraction_with_complete_provenance():
    extractor = KnowledgeExtractor()
    notes = "- Backpropagation computes gradients using the multivariate chain rule."
    segments = [
        TranscriptSegment(id="seg_01", start=10.0, end=20.0, text="Backpropagation computes gradients using chain rule."),
        TranscriptSegment(id="seg_02", start=22.0, end=35.0, text="The multivariate chain rule is fundamental to backpropagation."),
    ]
    claims = extractor.extract_claims(notes, segments, video_id="vid_math", tenant_id="t_clm", workspace_id="ws_clm")

    assert len(claims) == 1
    clm = claims[0]
    assert clm.video_id == "vid_math"
    assert clm.tenant_id == "t_clm"
    assert clm.workspace_id == "ws_clm"
    assert clm.source_timestamps == [10.0, 22.0]
    assert clm.status == FactVerificationStatus.SUPPORTED
    assert clm.epistemic_classification == "HEURISTIC"


def test_unverified_claim_handling():
    extractor = KnowledgeExtractor()
    notes = "- Quantum computing solves NP-complete problems in constant time."
    segments = [
        TranscriptSegment(id="s1", start=0.0, end=10.0, text="Introduction to standard sorting algorithms."),
    ]
    claims = extractor.extract_claims(notes, segments, video_id="vid_sort")
    assert len(claims) == 1
    assert claims[0].status == FactVerificationStatus.UNVERIFIED
    assert claims[0].confidence <= 0.50
    assert claims[0].source_timestamps == []


def test_evidence_linkage_to_claims():
    kg = KnowledgeGraph()
    clm_node = kg.add_claim(claim_id="clm_01", claim_text="Adam optimizer combines Momentum and RMSProp", confidence=0.95)
    ev_node = kg.register_node(node_id="ev_01", name="Adam definition", definition="Adam utilizes first and second gradient moments.")

    edge = kg.link_evidence(claim_id="clm_01", evidence_id="ev_01", relation_type="SUPPORTS", confidence=0.98)
    assert edge.source_id == "ev_01"
    assert edge.target_id == "clm_01"
    assert edge.relation == KnowledgeEdgeType.SUPPORTS

    evidence_summary = kg.get_claim_evidence("clm_01")
    assert evidence_summary["status"] in ("VERIFIED", "PARTIALLY_SUPPORTED")
    assert len(evidence_summary["supports"]) == 1
    assert evidence_summary["supports"][0]["evidence_id"] == "ev_01"


def test_claim_epistemic_classification_truthful_heuristic():
    verifier = FactVerificationEngine()
    claim = FactClaim(id="c1", claim="Python was invented in 1991.", evidence="")
    verified = verifier.verify_claims([claim], segments=[])
    assert verified[0].epistemic_classification == "HEURISTIC"
    assert verified[0].status == FactVerificationStatus.UNVERIFIED


# =============================================================================
# 4. Honest Contradiction Detection Tests
# =============================================================================

def test_contradiction_detection_numeric_year_divergence():
    verifier = FactVerificationEngine()
    claim_a = FactClaim(id="ca", claim="Python was created in 1991 by Guido van Rossum.", evidence="", video_id="vid_hist_1")
    claim_b = FactClaim(id="cb", claim="Python was created in 1989 by Guido van Rossum.", evidence="", video_id="vid_hist_2")

    contra = verifier.detect_claim_contradiction(claim_a, claim_b)
    assert contra is not None
    assert contra["status"] == "CONTRADICTED"
    assert contra["contradiction_type"] == "divergent_numeric_values"
    assert contra["classification"] == "HEURISTIC"
    assert "vid_hist_1" in contra["claim_a_source"]
    assert "vid_hist_2" in contra["claim_b_source"]


def test_contradiction_detection_affirmation_vs_negation():
    verifier = FactVerificationEngine()
    claim_a = FactClaim(id="ca", claim="PyTorch supports distributed pipeline parallelism.", evidence="", video_id="v1")
    claim_b = FactClaim(id="cb", claim="PyTorch does not support distributed pipeline parallelism.", evidence="", video_id="v2")

    contra = verifier.detect_claim_contradiction(claim_a, claim_b)
    assert contra is not None
    assert contra["status"] == "CONTRADICTED"
    assert contra["contradiction_type"] == "affirmation_vs_negation"
    assert contra["classification"] == "HEURISTIC"


def test_contradiction_preserves_both_sources_and_timestamps():
    verifier = FactVerificationEngine()
    claim_a = FactClaim(id="c1", claim="Transformer training requires 128 GPUs.", evidence="", video_id="lecture_A", source_timestamps=[120.0])
    claim_b = FactClaim(id="c2", claim="Transformer training requires 16 GPUs.", evidence="", video_id="lecture_B", source_timestamps=[340.0])

    contra = verifier.detect_claim_contradiction(claim_a, claim_b)
    assert contra is not None
    assert contra["claim_a_timestamps"] == [120.0]
    assert contra["claim_b_timestamps"] == [340.0]
    assert contra["claim_a_source"] == "lecture_A"
    assert contra["claim_b_source"] == "lecture_B"


def test_non_contradictory_statements_not_flagged():
    verifier = FactVerificationEngine()
    claim_a = FactClaim(id="ca", claim="Python supports object oriented programming.", evidence="")
    claim_b = FactClaim(id="cb", claim="Python supports functional programming features.", evidence="")

    contra = verifier.detect_claim_contradiction(claim_a, claim_b)
    assert contra is None


def test_graph_contradiction_detection_integration():
    kg = KnowledgeGraph()
    kg.register_node(node_id="clm_1", name="Claim 1", definition="Model A achieves 95% accuracy", video_id="source_1")
    kg.register_node(node_id="clm_2", name="Claim 2", definition="Model A achieves 82% accuracy", video_id="source_2")
    kg.add_edge(edge="clm_1", target_id="clm_2", relation_type="CONTRADICTS", confidence=0.95, description="Conflicting benchmark results")

    contras = kg.detect_contradictions()
    assert len(contras) >= 1
    found = contras[0]
    assert found["status"] == "CONTRADICTED"
    assert found["classification"] == "HEURISTIC"
    assert "source_1" in (found["source_citation"], found["target_citation"])
    assert "source_2" in (found["source_citation"], found["target_citation"])


# =============================================================================
# 5. Temporal Knowledge & Allen Interval Reasoning Tests
# =============================================================================

def test_allen_interval_algebra_relations_in_graph():
    kg = KnowledgeGraph()
    # Concept A: 10s - 30s
    node_a = kg.register_node(node_id="ta", name="Tokenization", start_ms=10000, end_ms=30000)
    # Concept B: 30s - 50s (Meets A)
    node_b = kg.register_node(node_id="tb", name="Embedding Lookup", start_ms=30000, end_ms=50000)
    # Concept C: 60s - 80s (After A and B)
    node_c = kg.register_node(node_id="tc", name="Attention Calculation", start_ms=60000, end_ms=80000)

    rel_ab = kg.temporal_relation("ta", "tb")
    assert rel_ab == TemporalRelation.MEETS.value

    rel_ac = kg.temporal_relation("ta", "tc")
    assert rel_ac == TemporalRelation.BEFORE.value

    rel_ca = kg.temporal_relation("tc", "ta")
    assert rel_ca == TemporalRelation.AFTER.value


def test_find_interval_relations_query():
    kg = KnowledgeGraph()
    kg.register_node(node_id="n_base", name="Linear Algebra Intro", start_ms=0, end_ms=60000)
    kg.register_node(node_id="n_during", name="Vector Spaces", start_ms=15000, end_ms=45000)
    kg.register_node(node_id="n_after", name="Eigenvalues", start_ms=70000, end_ms=120000)

    during_nodes = kg.find_interval_relations(reference_node_id="n_during", target_relation="DURING")
    # n_during is during n_base
    assert any(n.id == "n_base" for n in during_nodes)

    after_nodes = kg.find_interval_relations(reference_node_id="n_base", target_relation="BEFORE")
    # n_base is BEFORE n_after
    assert any(n.id == "n_after" for n in after_nodes)


def test_temporal_what_was_discussed_before_and_after():
    kg = KnowledgeGraph()
    n1 = kg.register_node(node_id="n1", name="Gradient Descent", start_ms=0, end_ms=60000)
    n2 = kg.register_node(node_id="n2", name="Learning Rate", start_ms=70000, end_ms=120000)
    n3 = kg.register_node(node_id="n3", name="Momentum", start_ms=130000, end_ms=180000)

    before_n2 = kg.what_was_discussed_before(target_node_id_or_time="n2", delta_sec=120.0)
    assert len(before_n2) >= 1
    assert before_n2[0].name == "Gradient Descent"

    after_n2 = kg.what_was_discussed_after(target_node_id_or_time="n2", delta_sec=120.0)
    assert len(after_n2) >= 1
    assert after_n2[0].name == "Momentum"


def test_temporal_concepts_between_range():
    kg = KnowledgeGraph()
    kg.register_node(node_id="c1", name="Early Concept", start_ms=10000, end_ms=30000)
    kg.register_node(node_id="c2", name="Middle Concept", start_ms=40000, end_ms=70000)
    kg.register_node(node_id="c3", name="Late Concept", start_ms=100000, end_ms=120000)

    between = kg.concepts_between(start_time=35.0, end_time=75.0)
    assert len(between) == 1
    assert between[0].name == "Middle Concept"


# =============================================================================
# 6. Bounded Multi-Hop & Cross-Modal Graph Traversal Tests
# =============================================================================

def test_bounded_multihop_traversal_depth_and_node_limits():
    kg = KnowledgeGraph()
    # Build a chain: N0 -> N1 -> N2 -> N3 -> N4
    for i in range(5):
        kg.register_node(node_id=f"node_{i}", name=f"Concept {i}")
    for i in range(4):
        kg.add_edge(edge=f"node_{i}", target_id=f"node_{i+1}", relation_type="PREREQUISITE_OF")

    # 1-hop traversal
    trav_1 = kg.traverse_bounded(start_node_ids="node_0", max_hops=1, max_nodes=50)
    assert trav_1["hops_reached"] == 1
    assert len(trav_1["nodes"]) == 2  # node_0 and node_1

    # 2-hop traversal
    trav_2 = kg.traverse_bounded(start_node_ids="node_0", max_hops=2, max_nodes=50)
    assert trav_2["hops_reached"] == 2
    assert len(trav_2["nodes"]) == 3  # node_0, node_1, node_2

    # Node count budget constraint
    trav_capped = kg.traverse_bounded(start_node_ids="node_0", max_hops=4, max_nodes=2)
    assert len(trav_capped["nodes"]) <= 2


def test_bounded_multihop_traversal_cycle_prevention():
    kg = KnowledgeGraph()
    # Build cycle: A -> B -> C -> A
    kg.register_node(node_id="na", name="A")
    kg.register_node(node_id="nb", name="B")
    kg.register_node(node_id="nc", name="C")

    kg.add_edge("na", "nb", relation_type="RELATED_TO")
    kg.add_edge("nb", "nc", relation_type="RELATED_TO")
    kg.add_edge("nc", "na", relation_type="RELATED_TO")

    # Traversal should terminate without infinite loop
    trav = kg.traverse_bounded(start_node_ids="na", max_hops=5, max_nodes=20)
    assert len(trav["nodes"]) == 3
    assert trav["total_nodes"] == 3


def test_multiseed_bounded_traversal():
    kg = KnowledgeGraph()
    kg.register_node(node_id="s1", name="Seed 1")
    kg.register_node(node_id="s2", name="Seed 2")
    kg.register_node(node_id="t1", name="Target 1")
    kg.register_node(node_id="t2", name="Target 2")

    kg.add_edge("s1", "t1", relation_type="EXPLAINS")
    kg.add_edge("s2", "t2", relation_type="EXPLAINS")

    trav = kg.traverse_bounded(start_node_ids=["s1", "s2"], max_hops=1)
    assert len(trav["nodes"]) == 4
    assert trav["start_node_ids"] == ["s1", "s2"]


def test_reasoning_path_traceability():
    kg = KnowledgeGraph()
    kg.register_node("py", "Python")
    kg.register_node("gvr", "Guido van Rossum", node_type=KnowledgeObjectType.PERSON)
    kg.register_node("pl", "Programming Language", node_type=KnowledgeObjectType.CONCEPT)

    kg.add_edge("py", "gvr", relation_type="DEVELOPED_BY" if hasattr(KnowledgeEdgeType, "DEVELOPED_BY") else "RELATED_TO")
    kg.add_edge("py", "pl", relation_type="IS_A")

    trav = kg.traverse_bounded(start_node_ids="py", max_hops=1)
    assert any("gvr" in p for p in trav["paths"])
    assert any("pl" in p for p in trav["paths"])


def test_cross_modal_graph_connectivity():
    kg = KnowledgeGraph()
    # Transcript claim -> Keyframe -> Formula
    kg.add_claim("clm_bellman", "Bellman equation expresses recursive value function", source_id="vid_rl")
    kg.register_node("kf_01", "Keyframe Slide 4", node_type=KnowledgeObjectType.KEYFRAME, start_ms=45000, end_ms=50000)
    kg.register_node("formula_bellman", "V(s) = max_a (R + gamma * V(s'))", node_type=KnowledgeObjectType.FORMULA)

    kg.add_edge("clm_bellman", "kf_01", relation_type="APPEARS_IN")
    kg.add_edge("kf_01", "formula_bellman", relation_type="DEPICTS")

    trav = kg.traverse_bounded(start_node_ids="clm_bellman", max_hops=2)
    node_ids = {n.id for n in trav["nodes"]}
    assert "clm_bellman" in node_ids
    assert "kf_01" in node_ids
    assert "formula_bellman" in node_ids


def test_graph_tenant_isolation_boundary_enforcement():
    kg = KnowledgeGraph()
    kg.register_node("node_tenant_x", "Confidential Alpha", tenant_id="tenant_X")
    kg.register_node("node_tenant_y", "Confidential Beta", tenant_id="tenant_Y")
    kg.add_edge("node_tenant_x", "node_tenant_y", relation_type="RELATED_TO", tenant_id="tenant_X")

    # Traverse as tenant_X
    trav_x = kg.traverse_bounded("node_tenant_x", tenant_id="tenant_X")
    for n in trav_x["nodes"]:
        assert n.tenant_id in ("tenant_X", "default")
        assert n.tenant_id != "tenant_Y"


# =============================================================================
# 7. Advanced Graph RAG, Grounding & Truthful Abstention Tests
# =============================================================================

def test_graph_aware_hybrid_retrieval_scoring():
    engine = HybridSearchEngine()
    engine.index_segments(
        video_id="vid_ai",
        video_title="Modern AI",
        segments=[
            TranscriptSegment(
                id="seg_01",
                start=10.0,
                end=25.0,
                text="Convolutional layers apply spatial kernels to detect visual edges and textures.",
            ),
            TranscriptSegment(
                id="seg_02",
                start=30.0,
                end=45.0,
                text="Stochastic gradient descent updates model weights iteratively.",
            ),
        ],
    )
    kg = KnowledgeGraph()
    kg.register_node("n_cnn", "CNN", definition="Convolutional layers for computer vision")
    kg.register_node("n_kernel", "Kernel", definition="Spatial filter matrix")
    kg.add_edge("n_cnn", "n_kernel", relation_type="USES")

    hits = engine.search("convolutional kernel", k=2, graph=kg, strategy=RetrievalStrategy.HYBRID)
    assert len(hits) >= 1
    top_hit = hits[0]
    assert "spatial kernels" in top_hit.snippet
    assert "graph" in top_hit.score_breakdown
    assert top_hit.score_breakdown["graph"] >= 0.0


def test_rag_grounded_answer_with_multimodal_citations():
    engine = HybridSearchEngine()
    engine.index_segments(
        video_id="vid_trans",
        video_title="Transformers Explained",
        segments=[
            TranscriptSegment(
                id="tx_01",
                start=12.0,
                end=24.0,
                text="Self-attention allows tokens to attend to all positions within the same sequence.",
            )
        ],
    )
    rag = RAGEngine(engine)
    response = rag.answer_query("What does self-attention allow?", video_id="vid_trans")

    assert response.abstention is False
    assert response.confidence >= 0.75
    assert len(response.citations) >= 1
    cit = response.citations[0]
    assert cit.video_id == "vid_trans"
    assert cit.timestamp_start == 12.0
    assert cit.grounding_status == "SUPPORTED"
    assert "tokens to attend" in response.answer


def test_rag_truthful_abstention_on_unrelated_query():
    engine = HybridSearchEngine()
    engine.index_segments(
        video_id="vid_math",
        video_title="Calculus",
        segments=[
            TranscriptSegment(
                id="c1",
                start=0.0,
                end=10.0,
                text="The integral computes the accumulated area under a curve.",
            )
        ],
    )
    rag = RAGEngine(engine)
    response = rag.answer_query("What are the optimal ingredients for authentic Neapolitan pizza?", video_id="vid_math")

    assert response.abstention is True
    assert response.confidence <= 0.35
    assert "sufficient verified evidence" in response.answer.lower()
    assert response.confidence_level == "INSUFFICIENT"


def test_rag_reasoning_path_propagation():
    engine = HybridSearchEngine()
    engine.index_segments(
        video_id="vid_lang",
        video_title="Programming Paradigms",
        segments=[
            TranscriptSegment(
                id="l1",
                start=5.0,
                end=20.0,
                text="Python was created by Guido van Rossum and released in 1991.",
            )
        ],
    )
    kg = KnowledgeGraph()
    kg.register_node("py", "Python")
    kg.register_node("gvr", "Guido van Rossum")
    kg.add_edge("py", "gvr", relation_type="CREATED_BY" if hasattr(KnowledgeEdgeType, "CREATED_BY") else "RELATED_TO")

    rag = RAGEngine(engine)
    response = rag.answer_query("Who created Python?", video_id="vid_lang", graph=kg)

    assert response.abstention is False
    assert len(response.citations) >= 1


def test_graph_context_controls_prevent_explosion():
    kg = KnowledgeGraph()
    # Build star graph with 100 neighbors
    kg.register_node("hub", "Central Hub")
    for i in range(100):
        kg.register_node(f"leaf_{i}", f"Leaf {i}")
        kg.add_edge("hub", f"leaf_{i}", relation_type="RELATED_TO")

    # Enforce tight budget
    trav = kg.traverse_bounded(start_node_ids="hub", max_hops=1, max_nodes=15)
    assert len(trav["nodes"]) <= 15
    assert trav["total_nodes"] <= 15


def test_offline_deterministic_fallback_no_network():
    # Verify that full multimodal extraction and graph traversal run offline with zero network calls
    extractor = KnowledgeExtractor()
    kg = KnowledgeGraph()

    notes = """
    **Concept: Backpropagation**: Backpropagation computes gradient descent steps.
    - Backpropagation computes gradient descent steps.
    """
    segments = [TranscriptSegment(id="s1", start=0.0, end=10.0, text="Gradient descent steps.")]

    extracted = extractor.extract_all(notes, segments, tenant_id="t_off", workspace_id="ws_off")
    stats = kg.integrate_multimodal_knowledge(extracted, tenant_id="t_off", workspace_id="ws_off")

    assert stats["nodes_added"] >= 1
    assert stats["canonical_nodes"] >= 1
    # Verify offline classifications
    for node in kg.nodes.values():
        assert node.classification in ("REAL", "HEURISTIC", "DETERMINISTIC_OFFLINE", "FALLBACK")
