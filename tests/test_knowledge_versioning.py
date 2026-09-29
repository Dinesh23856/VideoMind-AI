"""Tests for Knowledge Version Control, Content Fingerprinting, and Dependency Invalidation."""

import pytest
from app.domain.models import ConceptNode
from app.knowledge.versioning import KnowledgeVersionControl
from app.pipeline.fingerprint import ContentFingerprintEngine
from app.pipeline.invalidation import DependencyInvalidationGraph, PipelineStage


def test_knowledge_version_control_commit_and_diff():
    kvc = KnowledgeVersionControl()

    # Commit 1
    c1 = ConceptNode(id="cnc_relu", name="ReLU", definition="Rectified Linear Unit", importance=0.9)
    c2 = ConceptNode(id="cnc_sigmoid", name="Sigmoid", definition="Sigmoid activation function", importance=0.85)

    rev1 = kvc.commit(concepts=[c1, c2], message="Initial activations")
    assert rev1.revision_id.startswith("rev_")
    assert rev1.parent_id is None
    assert len(rev1.concepts) == 2

    # Commit 2: Modify ReLU, remove Sigmoid, add GELU
    c1_mod = ConceptNode(id="cnc_relu", name="ReLU", definition="f(x) = max(0, x)", importance=0.95)
    c3 = ConceptNode(id="cnc_gelu", name="GELU", definition="Gaussian Error Linear Unit", importance=0.9)

    rev2 = kvc.commit(concepts=[c1_mod, c3], message="Update activations: add GELU, drop Sigmoid")
    assert rev2.parent_id == rev1.revision_id

    # Compute Diff
    diff = kvc.diff(rev1.revision_id, rev2.revision_id)
    assert "cnc_gelu" in diff.added_concepts
    assert "cnc_sigmoid" in diff.removed_concepts
    assert len(diff.modified_concepts) == 1
    assert diff.modified_concepts[0]["concept_id"] == "cnc_relu"
    assert "GELU" in diff.summary or "+1" in diff.summary


def test_knowledge_branching():
    kvc = KnowledgeVersionControl()
    c1 = ConceptNode(id="cnc_1", name="Concept 1", definition="Test definition")
    kvc.commit(concepts=[c1], branch="main", message="Base")

    # Create experimental branch
    kvc.branch("experiment", from_branch="main")
    head_exp = kvc.get_head("experiment")
    head_main = kvc.get_head("main")
    assert head_exp.revision_id == head_main.revision_id


def test_content_fingerprint_engine():
    data = b"Multimodal test payload with audio & video frames"
    fp1 = ContentFingerprintEngine.fingerprint_artifact("audio", data, {"samplerate": 16000})
    fp2 = ContentFingerprintEngine.fingerprint_artifact("audio", data, {"samplerate": 16000})
    fp3 = ContentFingerprintEngine.fingerprint_artifact("audio", data, {"samplerate": 44100})

    assert fp1.sha256 == fp2.sha256
    assert fp1.combined_hash == fp2.combined_hash
    # Changing parameter changes combined hash
    assert fp1.combined_hash != fp3.combined_hash
    assert fp1.size_bytes == len(data)


def test_dependency_invalidation_graph():
    graph = DependencyInvalidationGraph()

    # If only OCR is modified, downstream is only KNOWLEDGE_EXTRACTION -> EMBEDDING_GENERATION -> RAG_INDEXING
    # AUDIO_EXTRACT and TRANSCRIPTION should NOT be invalidated!
    invalidated = graph.compute_invalidated_stages([PipelineStage.OCR])

    assert PipelineStage.OCR in invalidated
    assert PipelineStage.KNOWLEDGE_EXTRACTION in invalidated
    assert PipelineStage.EMBEDDING_GENERATION in invalidated
    assert PipelineStage.RAG_INDEXING in invalidated
    assert PipelineStage.TRANSCRIPTION not in invalidated
    assert PipelineStage.AUDIO_EXTRACT not in invalidated

    assert graph.can_skip_stage(PipelineStage.TRANSCRIPTION, invalidated) is True
    assert graph.can_skip_stage(PipelineStage.OCR, invalidated) is False

    # If MEDIA_INGEST changed, everything downstream must re-run
    all_inv = graph.compute_invalidated_stages([PipelineStage.MEDIA_INGEST])
    assert len(all_inv) == len(PipelineStage)
