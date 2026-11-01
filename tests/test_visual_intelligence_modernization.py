"""
Test Suite: P2 Visual & Video Intelligence Modernization.

Verifies:
1. Deterministic perceptual hashing (dHash, aHash, pHash, Hamming distance, deduplication)
2. Temporal scene boundary detection & representative keyframe selection
3. Enhanced OCR engine with text normalization & duplicate suppression
4. Visual content type classification (slide, diagram, code, formula, terminal)
5. Local VLM interface with truthful NOT_AVAILABLE fallback
6. Multimodal video event extraction with millisecond-precision alignment
7. Multimodal Timeline 2.0 temporal fusion
8. Vector Store parent-child chunking & visual hybrid retrieval
9. Knowledge Graph visual intelligence integration
10. Multi-tenant and workspace isolation across visual artifacts
"""

import os
import shutil
import tempfile
import numpy as np
import pytest
from PIL import Image, ImageDraw

from app.domain.models import (
    AudioEvent,
    ConceptNode,
    KnowledgeEdgeType,
    KnowledgeObjectType,
    TranscriptSegment,
    VisualFrameContext,
)
from app.knowledge.graph import KnowledgeGraph
from app.retrieval.hybrid_search import HybridSearchEngine, RetrievalStrategy
from app.retrieval.multi_vector import ModalityType, MultiVectorDocument, ParentChildChunker
from app.retrieval.vector_store import InMemoryVectorStore
from app.vision.classifier import VisualClassificationResult, VisualContentClassifier
from app.vision.events import MultimodalEventExtractor, MultimodalVideoEvent
from app.vision.extractor import DifferentialVisionExtractor, VideoKeyframeExtractor
from app.vision.ocr import EnhancedOCREngine, OCRBoundingBox, OCRResult
from app.vision.perceptual import PerceptualHasher
from app.vision.scene_detector import SceneBoundary, TemporalSceneDetector
from app.vision.timeline import (
    align_visual_frames_to_transcript,
    build_multimodal_timeline,
    find_multimodal_context_at,
)
from app.vision.vlm import LocalVLMInterface, VLMResponse


# =============================================================================
# Helper Fixtures & Utilities
# =============================================================================

@pytest.fixture
def sample_blank_image():
    """Generates a plain white 320x240 image."""
    return Image.new("RGB", (320, 240), color="white")


@pytest.fixture
def sample_code_image():
    """Generates an image resembling a dark code editor."""
    img = Image.new("RGB", (320, 240), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)
    # Draw horizontal text-like lines
    for y in range(30, 200, 16):
        draw.line([(30, y), (200, y)], fill=(220, 220, 220), width=3)
    return img


@pytest.fixture
def sample_diagram_image():
    """Generates an image resembling a diagram with boxes and connecting lines."""
    img = Image.new("RGB", (320, 240), color=(250, 250, 250))
    draw = ImageDraw.Draw(img)
    # Draw boxes
    draw.rectangle([40, 40, 120, 90], outline=(0, 100, 200), width=3)
    draw.rectangle([200, 40, 280, 90], outline=(0, 100, 200), width=3)
    draw.rectangle([120, 150, 200, 200], outline=(200, 100, 0), width=3)
    # Draw connecting lines
    draw.line([(120, 65), (200, 65)], fill=(50, 50, 50), width=2)
    draw.line([(80, 90), (160, 150)], fill=(50, 50, 50), width=2)
    draw.line([(240, 90), (160, 150)], fill=(50, 50, 50), width=2)
    return img


# =============================================================================
# 1. Perceptual Hashing & Similarity Tests
# =============================================================================

def test_perceptual_hasher_truth_classification():
    hasher = PerceptualHasher()
    assert hasher.get_classification() == "REAL"
    assert hasher.get_semantic_classification() == "HEURISTIC"


def test_dhash_identical_and_different_images(sample_blank_image, sample_code_image):
    hasher = PerceptualHasher()

    h_blank1 = hasher.compute_dhash(sample_blank_image)
    h_blank2 = hasher.compute_dhash(sample_blank_image.copy())
    h_code = hasher.compute_dhash(sample_code_image)

    # Identical images must produce 0 Hamming distance
    assert hasher.hamming_distance(h_blank1, h_blank2) == 0
    assert hasher.perceptual_similarity(h_blank1, h_blank2) == 1.0
    assert hasher.is_duplicate_frame(sample_blank_image, sample_blank_image.copy()) is True

    # Visually distinct images must produce large Hamming distance
    dist = hasher.hamming_distance(h_blank1, h_code)
    assert dist > 10
    assert hasher.is_duplicate_frame(sample_blank_image, sample_code_image) is False


def test_ahash_and_phash_computation(sample_code_image, sample_diagram_image):
    hasher = PerceptualHasher()

    ahash_code = hasher.compute_ahash(sample_code_image)
    phash_code = hasher.compute_phash(sample_code_image)
    phash_diag = hasher.compute_phash(sample_diagram_image)

    assert len(ahash_code) == 16  # 64 bits = 16 hex chars
    assert len(phash_code) == 16

    # Frequency-domain DCT pHash should distinguish code editor from white diagram
    assert hasher.hamming_distance(phash_code, phash_diag) > 10


# =============================================================================
# 2. Enhanced OCR Engine Tests
# =============================================================================

def test_ocr_text_normalization():
    ocr = EnhancedOCREngine()

    raw_text = "This is a demon-\nstration of text normal-\nization in Video-\nMind."
    normalized = ocr.normalize_text(raw_text)

    assert "demonstration" in normalized
    assert "normalization" in normalized
    assert "VideoMind" in normalized
    assert "-\n" not in normalized


def test_ocr_duplicate_suppression():
    ocr = EnhancedOCREngine()

    t1 = "Convolutional Neural Networks and Feature Maps"
    t2 = "Convolutional Neural Networks and Feature Maps"
    t3 = "Recurrent Neural Networks and Sequence Modeling"

    assert ocr.is_duplicate_ocr(t1, t2) is True
    assert ocr.is_duplicate_ocr(t1, t3) is False


def test_ocr_truthful_classification(sample_blank_image):
    ocr = EnhancedOCREngine()
    res = ocr.extract_ocr(sample_blank_image)

    assert isinstance(res, OCRResult)
    assert res.classification in ("REAL", "NOT_AVAILABLE")
    if not ocr.is_available:
        assert res.classification == "NOT_AVAILABLE"
        assert res.text == ""  # Never fabricates text


# =============================================================================
# 3. Visual Content Type Classification Tests
# =============================================================================

def test_visual_classifier_truth_classification():
    classifier = VisualContentClassifier()
    assert classifier.get_classification() == "HEURISTIC"


def test_visual_classifier_categories(sample_code_image, sample_diagram_image, sample_blank_image):
    classifier = VisualContentClassifier()

    # Code classification
    res_code = classifier.classify(
        sample_code_image,
        ocr_text="def forward(self, x):\n    return self.linear(x)\nimport torch",
    )
    assert res_code.frame_type == "code"
    assert res_code.confidence >= 0.8
    assert res_code.classification == "HEURISTIC"

    # Formula classification
    res_formula = classifier.classify(
        sample_blank_image,
        ocr_text="L = \\frac{1}{N} \\sum_{i=1}^N (y_i - f(x_i))^2\n\\partial L / \\partial w",
    )
    assert res_formula.frame_type == "formula"

    # Terminal classification
    res_term = classifier.classify(
        sample_code_image,
        ocr_text="$ pip install torch torchvision\n$ python train.py --epochs 10",
    )
    assert res_term.frame_type == "terminal"

    # Diagram classification
    res_diag = classifier.classify(
        sample_diagram_image,
        ocr_text="Architecture: Encoder -> Bottleneck -> Decoder",
    )
    assert res_diag.frame_type in ("diagram", "slide")


# =============================================================================
# 4. Temporal Scene Boundary Detection Tests
# =============================================================================

def test_temporal_scene_detector():
    detector = TemporalSceneDetector(transition_threshold=0.20, min_scene_duration_sec=2.0)
    assert detector.get_classification() == "REAL"

    # Create sequence of frames:
    # 0.0s - 4.0s: Slide 1 (white)
    # 5.0s - 9.0s: Slide 2 (dark code)
    # 10.0s - 14.0s: Slide 3 (diagram)
    slide1 = Image.new("RGB", (160, 120), color="white")
    slide2 = Image.new("RGB", (160, 120), color="black")
    slide3 = Image.new("RGB", (160, 120), color="blue")

    candidates = [
        (0.0, None, slide1),
        (2.0, None, slide1),
        (4.0, None, slide1),
        (5.0, None, slide2),  # Scene cut here
        (7.0, None, slide2),
        (9.0, None, slide2),
        (10.0, None, slide3), # Scene cut here
        (12.0, None, slide3),
        (14.0, None, slide3),
    ]

    scenes = detector.detect_scenes(candidates, video_id="test_vid")

    assert len(scenes) >= 2
    assert scenes[0].start_time == 0.0
    assert scenes[-1].end_time >= 14.0
    assert all(isinstance(s, SceneBoundary) for s in scenes)
    assert all(s.classification == "REAL" for s in scenes)


# =============================================================================
# 5. Local VLM Interface Tests
# =============================================================================

def test_local_vlm_not_available_fallback(sample_code_image):
    vlm = LocalVLMInterface(endpoint_url=None)

    assert vlm.is_available is False
    assert vlm.get_classification() == "NOT_AVAILABLE"

    resp = vlm.describe_image(sample_code_image)
    assert isinstance(resp, VLMResponse)
    assert resp.is_available is False
    assert resp.classification == "NOT_AVAILABLE"
    assert "not configured" in resp.text.lower()

    resp_query = vlm.answer_visual_query(sample_code_image, "What language is shown?")
    assert resp_query.classification == "NOT_AVAILABLE"


# =============================================================================
# 6. Multimodal Video Event Extraction Tests
# =============================================================================

def test_multimodal_event_extraction():
    extractor = MultimodalEventExtractor()
    assert extractor.get_classification() == "HEURISTIC"

    scenes = [
        SceneBoundary(
            scene_id="scn_1",
            video_id="vid_demo",
            start_time=0.0,
            end_time=15.0,
            start_ms=0,
            end_ms=15000,
            duration_sec=15.0,
        ),
        SceneBoundary(
            scene_id="scn_2",
            video_id="vid_demo",
            start_time=15.0,
            end_time=30.0,
            start_ms=15000,
            end_ms=30000,
            duration_sec=15.0,
        ),
    ]

    frames = [
        VisualFrameContext(
            frame_id="frm_1",
            timestamp=2.0,
            frame_type="code",
            description="PyTorch training loop",
            ocr_text="import torch\ndef train(model):",
            scene_id="scn_1",
        ),
        VisualFrameContext(
            frame_id="frm_2",
            timestamp=16.0,
            frame_type="slide",
            description="Exercise slide",
            ocr_text="Question: What is the gradient with respect to w?",
            scene_id="scn_2",
        ),
    ]

    segments = [
        TranscriptSegment(
            id="seg_1",
            start=1.0,
            end=5.0,
            speaker="Dr. Smith",
            text="Here we implement the training loop in PyTorch.",
        ),
        TranscriptSegment(
            id="seg_2",
            start=15.5,
            end=20.0,
            speaker="Dr. Smith",
            text="Let's examine this practice question on the slide.",
        ),
    ]

    events = extractor.extract_events(
        video_id="vid_demo",
        scenes=scenes,
        frames=frames,
        segments=segments,
    )

    event_types = [e.event_type for e in events]
    assert "screen_changes" in event_types
    assert "code_appears" in event_types
    assert "question_appears_on_slide" in event_types
    assert "speaker_explains_slide" in event_types

    # Verify millisecond precision
    for e in events:
        assert isinstance(e.start_ms, int)
        assert isinstance(e.end_ms, int)
        assert e.end_ms >= e.start_ms
        assert e.classification == "HEURISTIC"


# =============================================================================
# 7. Multimodal Timeline 2.0 Tests
# =============================================================================

def test_multimodal_timeline_fusion():
    segments = [
        TranscriptSegment(id="s1", start=0.0, end=5.0, text="Welcome to the lecture", speaker="Speaker 1"),
        TranscriptSegment(id="s2", start=5.0, end=10.0, text="Look at this diagram", speaker="Speaker 1"),
    ]
    frames = [
        VisualFrameContext(frame_id="f1", timestamp=2.0, frame_type="slide", ocr_text="Lecture Overview"),
        VisualFrameContext(frame_id="f2", timestamp=6.0, frame_type="diagram", ocr_text="Transformer Architecture"),
    ]
    scenes = [
        SceneBoundary(scene_id="sc1", video_id="v1", start_time=0.0, end_time=5.0, start_ms=0, end_ms=5000, duration_sec=5.0),
        SceneBoundary(scene_id="sc2", video_id="v1", start_time=5.0, end_time=10.0, start_ms=5000, end_ms=10000, duration_sec=5.0),
    ]

    timeline = build_multimodal_timeline(segments=segments, frames=frames, scenes=scenes)

    assert len(timeline) >= 4
    # Check that points contain aligned visual and scene IDs
    p_diagram = next((p for p in timeline if p.visual_frame_id == "f2"), None)
    assert p_diagram is not None
    assert p_diagram.frame_type == "diagram"

    context = find_multimodal_context_at(timeline, timestamp=6.0, window_seconds=2.0)
    assert "f2" in context["frame_ids"]
    assert "Transformer Architecture" in context["text"] or "diagram" in context["text"] or len(context["frame_ids"]) > 0


# =============================================================================
# 8. Visual Parent-Child Vector Indexing & Hybrid Retrieval Tests
# =============================================================================

def test_visual_parent_child_chunking_and_retrieval():
    scenes = [
        SceneBoundary(
            scene_id="scn_resnet",
            video_id="vid_deeplearning",
            start_time=10.0,
            end_time=30.0,
            start_ms=10000,
            end_ms=30000,
            duration_sec=20.0,
        )
    ]
    frames = [
        VisualFrameContext(
            frame_id="frm_skip_conn",
            timestamp=15.0,
            frame_type="code",
            description="Residual connection implementation in PyTorch",
            ocr_text="class ResidualBlock(nn.Module):\n    def forward(self, x):\n        return x + self.conv(x)",
            scene_id="scn_resnet",
        )
    ]

    parents, children = ParentChildChunker.chunk_visual_scenes_and_keyframes(
        scenes=scenes,
        frames=frames,
        video_id="vid_deeplearning",
        video_title="Deep Learning Architecture",
        tenant_id="tenant_ai",
        workspace_id="ws_vision",
    )

    assert len(parents) == 1
    assert len(children) == 1
    assert parents[0].section_type == "visual_scene"
    assert children[0].parent_id == parents[0].id
    assert children[0].modality == ModalityType.CODE
    assert "ResidualBlock" in children[0].content

    # Index into HybridSearchEngine
    store = InMemoryVectorStore()
    search = HybridSearchEngine(vector_store=store)

    search.index_visual_content(
        scenes=scenes,
        frames=frames,
        video_id="vid_deeplearning",
        video_title="Deep Learning Architecture",
        tenant_id="tenant_ai",
        workspace_id="ws_vision",
    )

    # Query specifically for visual code
    pack = search.retrieve_evidence_pack(
        query="ResidualBlock skip connection PyTorch",
        strategy=RetrievalStrategy.HYBRID,
        tenant_id="tenant_ai",
        workspace_id="ws_vision",
    )

    assert len(pack.hits) > 0
    top_hit = pack.hits[0]
    assert "ResidualBlock" in top_hit.snippet
    assert top_hit.modality in ("code", "keyframe", "ocr")
    assert top_hit.parent_id == parents[0].id


# =============================================================================
# 9. Knowledge Graph Visual Intelligence Integration Tests
# =============================================================================

def test_knowledge_graph_visual_intelligence():
    kg = KnowledgeGraph()

    scenes = [
        SceneBoundary(
            scene_id="scn_att",
            video_id="vid_nlp",
            start_time=0.0,
            end_time=20.0,
            start_ms=0,
            end_ms=20000,
            duration_sec=20.0,
        )
    ]
    frames = [
        VisualFrameContext(
            frame_id="kf_att_formula",
            timestamp=5.0,
            frame_type="formula",
            description="Scaled dot-product attention formula",
            ocr_text="Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V",
            scene_id="scn_att",
        )
    ]
    events = [
        MultimodalVideoEvent(
            event_id="evt_att_formula",
            event_type="formula_appears",
            start_time=5.0,
            end_time=8.0,
            start_ms=5000,
            end_ms=8000,
            video_id="vid_nlp",
            scene_id="scn_att",
            frame_id="kf_att_formula",
            description="Scaled dot-product formula displayed",
        )
    ]

    res = kg.integrate_visual_intelligence(
        scenes=scenes,
        frames=frames,
        events=events,
        video_id="vid_nlp",
        video_title="Attention Is All You Need",
        tenant_id="tenant_nlp",
        workspace_id="ws_transformers",
    )

    assert res["scenes"] == 1
    assert res["keyframes"] == 1
    assert res["events"] == 1

    # Verify node types in graph
    node_types = {n.node_type for n in kg.get_nodes(tenant_id="tenant_nlp", workspace_id="ws_transformers")}
    assert KnowledgeObjectType.SCENE in node_types
    assert KnowledgeObjectType.KEYFRAME in node_types
    assert KnowledgeObjectType.FORMULA in node_types
    assert KnowledgeObjectType.OCR_REGION in node_types
    assert KnowledgeObjectType.EVENT in node_types

    # Verify edge relations
    edge_types = {e.relation for e in kg.get_edges(tenant_id="tenant_nlp", workspace_id="ws_transformers")}
    assert KnowledgeEdgeType.CONTAINS_SCENE in edge_types
    assert KnowledgeEdgeType.VISUALLY_SHOWN_IN in edge_types
    assert KnowledgeEdgeType.DEPICTS in edge_types
    assert KnowledgeEdgeType.HAS_OCR in edge_types
    assert KnowledgeEdgeType.REFERENCES in edge_types

    # Exporters should run cleanly
    cyto = kg.export_cytoscape_elements(tenant_id="tenant_nlp", workspace_id="ws_transformers")
    assert len(cyto) >= 5

    mermaid = kg.export_mermaid_diagram(tenant_id="tenant_nlp", workspace_id="ws_transformers")
    assert "graph TD" in mermaid
    assert "kf_node_vid_nlp_kf_att_formula" in mermaid


# =============================================================================
# 10. Multi-Tenant and Workspace Isolation Tests
# =============================================================================

def test_visual_tenant_isolation():
    kg = KnowledgeGraph()

    scenes_t1 = [SceneBoundary(scene_id="s_t1", video_id="v_t1", start_time=0.0, end_time=10.0, start_ms=0, end_ms=10000, duration_sec=10.0)]
    frames_t1 = [VisualFrameContext(frame_id="f_t1", timestamp=2.0, frame_type="slide", ocr_text="Secret Tenant 1 Slide", scene_id="s_t1")]

    scenes_t2 = [SceneBoundary(scene_id="s_t2", video_id="v_t2", start_time=0.0, end_time=10.0, start_ms=0, end_ms=10000, duration_sec=10.0)]
    frames_t2 = [VisualFrameContext(frame_id="f_t2", timestamp=2.0, frame_type="slide", ocr_text="Secret Tenant 2 Slide", scene_id="s_t2")]

    kg.integrate_visual_intelligence(scenes_t1, frames_t1, video_id="v_t1", tenant_id="tenant_alpha", workspace_id="ws_main")
    kg.integrate_visual_intelligence(scenes_t2, frames_t2, video_id="v_t2", tenant_id="tenant_beta", workspace_id="ws_main")

    nodes_t1 = kg.get_nodes(tenant_id="tenant_alpha", workspace_id="ws_main")
    nodes_t2 = kg.get_nodes(tenant_id="tenant_beta", workspace_id="ws_main")

    # Strict isolation: tenant_alpha must never see tenant_beta nodes
    t1_names = [n.name for n in nodes_t1]
    t2_names = [n.name for n in nodes_t2]

    assert any("f_t1" in name for name in t1_names)
    assert not any("f_t2" in name for name in t1_names)

    assert any("f_t2" in name for name in t2_names)
    assert not any("f_t1" in name for name in t2_names)


# =============================================================================
# 11. RAG Visual Grounding & Citations
# =============================================================================

def test_rag_visual_grounding_and_citations():
    from app.rag.engine import RAGEngine

    store = InMemoryVectorStore()
    search = HybridSearchEngine(vector_store=store)

    scenes = [
        SceneBoundary(
            scene_id="scn_backprop",
            video_id="vid_nn",
            start_time=60.0,
            end_time=90.0,
            start_ms=60000,
            end_ms=90000,
            duration_sec=30.0,
        )
    ]
    frames = [
        VisualFrameContext(
            frame_id="frm_chain_rule",
            timestamp=65.0,
            frame_type="formula",
            description="Chain rule partial derivative derivation",
            ocr_text="dLoss/dw = dLoss/dy * dy/dz * dz/dw",
            scene_id="scn_backprop",
        )
    ]

    search.index_visual_content(
        scenes=scenes,
        frames=frames,
        video_id="vid_nn",
        video_title="Neural Networks 101",
        tenant_id="tenant_ai",
        workspace_id="ws_main",
    )

    rag = RAGEngine(search_engine=search)
    resp = rag.answer_query(
        query="Chain rule partial derivative derivation in neural networks",
        tenant_id="tenant_ai",
        workspace_id="ws_main",
    )

    assert resp.abstention is False
    assert len(resp.citations) > 0
    top_citation = resp.citations[0]
    assert top_citation.keyframe_id == "frm_chain_rule"
    assert top_citation.scene_id == "scn_backprop"
    assert top_citation.modality in ("formula", "ocr", "keyframe")
    assert "dLoss" in resp.evidence or "derivative" in resp.evidence


# =============================================================================
# 12. Differential Vision Extractor Budget & Edge Cases
# =============================================================================

def test_differential_vision_extractor_edge_cases(tmp_path):
    extractor = DifferentialVisionExtractor()

    # Non-existent video returns empty list safely
    empty_res = extractor.extract_keyframes("non_existent_video_path.mp4", str(tmp_path))
    assert empty_res == []

    scenes, kfs = extractor.extract_scenes_and_keyframes("non_existent_video_path.mp4", str(tmp_path))
    assert scenes == []
    assert kfs == []
