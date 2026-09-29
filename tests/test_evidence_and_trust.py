"""Tests for Multimodal Memory, Independent Evidence Graph, Claim Resolution, and Source Trust."""

import pytest
from app.memory.multimodal_memory import (
    AudioAnchor,
    CodeMathAnchor,
    MultimodalMemoryItem,
    MultimodalMemoryStore,
    VisualAnchor,
)
from app.evidence.graph import (
    EvidenceEdgeType,
    EvidenceGraph,
    EvidenceNode,
    EvidenceNodeType,
)
from app.evidence.resolver import ClaimResolutionEngine, RawClaim
from app.evidence.trust import SourceMetadata, SourceTrustModel


def test_multimodal_memory_store():
    store = MultimodalMemoryStore()

    item1 = MultimodalMemoryItem(
        primary_text="Convolutional filters apply sliding dot products across 2D feature maps.",
        video_id="vid_cnn",
        timestamp=45.2,
        concept_tags=["CNN", "Convolution", "Feature Map"],
        visual_anchor=VisualAnchor(
            video_id="vid_cnn",
            timestamp=45.2,
            ocr_text="Kernel 3x3 Stride 1 Padding 0",
        ),
        code_math_anchor=CodeMathAnchor(
            latex_equation=r"y(i, j) = \sum_m \sum_n x(i+m, j+n) w(m, n)",
        ),
    )
    store.add_item(item1)

    assert len(store) == 1
    # Query by tag
    by_concept = store.find_by_concept("CNN")
    assert len(by_concept) == 1
    assert by_concept[0].memory_id == item1.memory_id

    # Query multimodal filter
    results = store.query_multimodal("Stride", has_visual=True)
    assert len(results) == 1
    assert results[0].visual_anchor.ocr_text == "Kernel 3x3 Stride 1 Padding 0"


def test_evidence_graph_and_verification():
    graph = EvidenceGraph()

    claim = EvidenceNode(
        node_type=EvidenceNodeType.CLAIM,
        content="ResNet uses skip connections to prevent gradient vanishing.",
        confidence=1.0,
    )
    ev_transcript = EvidenceNode(
        node_type=EvidenceNodeType.TRANSCRIPT_SPAN,
        content="He explains that residual shortcuts let gradients flow unimpeded.",
        confidence=0.95,
    )
    ev_code = EvidenceNode(
        node_type=EvidenceNodeType.CODE_SNIPPET,
        content="return x + self.conv(x)",
        confidence=1.0,
    )

    c_id = graph.add_node(claim)
    t_id = graph.add_node(ev_transcript)
    code_id = graph.add_node(ev_code)

    graph.add_edge(c_id, t_id, EvidenceEdgeType.SUPPORTED_BY, weight=1.0)
    graph.add_edge(c_id, code_id, EvidenceEdgeType.CORROBORATES, weight=0.8)

    assert len(graph) == 3
    supports = graph.get_supporting_evidence(c_id)
    assert len(supports) == 2

    verif = graph.compute_claim_verification(c_id)
    assert verif["status"] == "VERIFIED"
    assert verif["support_score"] > 1.5


def test_evidence_graph_contradiction():
    graph = EvidenceGraph()

    claim = EvidenceNode(
        node_type=EvidenceNodeType.CLAIM,
        content="Transformer self-attention has linear time complexity.",
        confidence=1.0,
    )
    counter_ev = EvidenceNode(
        node_type=EvidenceNodeType.TRANSCRIPT_SPAN,
        content="Standard full self-attention scales quadratically with O(N^2).",
        confidence=1.0,
    )

    c_id = graph.add_node(claim)
    cnt_id = graph.add_node(counter_ev)

    graph.add_edge(c_id, cnt_id, EvidenceEdgeType.CONTRADICTED_BY, weight=1.0)

    verif = graph.compute_claim_verification(c_id)
    assert verif["status"] == "CONTRADICTED"
    assert verif["contradiction_score"] == 1.0


def test_claim_resolution_engine():
    resolver = ClaimResolutionEngine(similarity_threshold=0.5)

    claims = [
        RawClaim(source_id="v1", text="Adam optimizer combines Momentum and RMSProp."),
        RawClaim(source_id="v2", text="Adam combines momentum with RMSprop."),
        RawClaim(source_id="v3", text="SGD uses a fixed learning rate throughout training."),
    ]

    resolved = resolver.resolve_claims(claims, total_sources=3)
    assert len(resolved) == 2  # Adam cluster, SGD cluster

    adam_cluster = next(r for r in resolved if "Adam" in r.canonical_text)
    assert len(adam_cluster.source_ids) == 2
    assert adam_cluster.consensus_status == "CONSENSUS"
    assert adam_cluster.contradiction_detected is False


def test_source_trust_model():
    model = SourceTrustModel(current_year=2026)

    meta_paper = SourceMetadata(
        source_id="paper_1",
        source_type="academic_paper",
        publisher="NeurIPS",
        publication_year=2024,
        peer_reviewed=True,
        domain_authority_score=0.9,
    )
    res_paper = model.compute_trust_score(meta_paper)
    assert res_paper["trust_score"] >= 0.95

    meta_old_video = SourceMetadata(
        source_id="vid_old",
        source_type="video",
        publication_year=2010,
        peer_reviewed=False,
        domain_authority_score=0.4,
    )
    res_old = model.compute_trust_score(meta_old_video)
    # Should have freshness penalty and lower base
    assert res_old["trust_score"] < res_paper["trust_score"]

    # User override
    meta_override = SourceMetadata(
        source_id="user_doc",
        source_type="user_note",
        user_trust_override=0.88,
    )
    res_override = model.compute_trust_score(meta_override)
    assert res_override["trust_score"] == 0.88
