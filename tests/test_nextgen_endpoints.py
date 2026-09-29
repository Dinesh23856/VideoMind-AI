"""Integration tests for all 10 next-gen REST endpoints."""

import pytest
from starlette.testclient import TestClient
from app.api.main import app, global_event_stream, global_kvc, global_lineage_tracker
from app.events.stream import EventModality, EventType, MultimodalEvent
from app.domain.models import ConceptNode
from app.lineage.tracker import LineageNode, LineageNodeType


@pytest.fixture
def client():
    return TestClient(app)


def test_api_events_stream(client):
    ev = MultimodalEvent(
        event_type=EventType.CONCEPT_INTRODUCED,
        modality=EventModality.TEXT,
        start_time=1.0,
        end_time=5.0,
        video_id="v_test_ep",
        payload={"concept": "Self-Attention"},
    )
    global_event_stream.add_event(ev)

    res = client.get("/api/v1/events/stream?video_id=v_test_ep")
    assert res.status_code == 200
    data = res.json()
    assert data["total"] >= 1
    assert any(e["payload"].get("concept") == "Self-Attention" for e in data["events"])


def test_api_evidence_verify(client):
    req_data = {
        "claim_text": "Residual connections allow identity mappings across deep neural networks.",
        "supporting_snippets": ["Skip connections pass inputs directly to deeper layers."],
        "contradicting_snippets": [],
    }
    res = client.post("/api/v1/evidence/verify", json=req_data)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] in ("VERIFIED", "PARTIALLY_SUPPORTED")
    assert data["support_score"] >= 1.0


def test_api_knowledge_diff(client):
    c1 = ConceptNode(id="cnc_a", name="A", definition="Def A")
    c2 = ConceptNode(id="cnc_b", name="B", definition="Def B")

    r1 = global_kvc.commit([c1], message="Initial")
    r2 = global_kvc.commit([c1, c2], message="Add B")

    res = client.post(
        "/api/v1/knowledge/diff",
        json={"from_revision": r1.revision_id, "to_revision": r2.revision_id},
    )
    assert res.status_code == 200
    data = res.json()
    assert "cnc_b" in data["added_concepts"]


def test_api_agents_research(client):
    req_data = {
        "topic": "Diffusion Models",
        "passages": [
            {"source_id": "v1", "title": "Diffusion 101", "text": "Denoising score matching reverses Gaussian diffusion noise.", "timestamp": 10.0}
        ],
    }
    res = client.post("/api/v1/agents/research", json=req_data)
    assert res.status_code == 200
    data = res.json()
    assert data["topic"] == "Diffusion Models"
    assert len(data["resolved_claims"]) >= 1


def test_api_agents_project(client):
    req_data = {
        "topic": "Convolutional Neural Networks",
        "concept_names": ["Convolution", "Pooling", "Backprop"],
    }
    res = client.post("/api/v1/agents/project", json=req_data)
    assert res.status_code == 200
    data = res.json()
    assert len(data["milestones"]) == 3
    assert len(data["rubric"]) == 4


def test_api_documents_and_cross_search(client):
    ingest_data = {
        "doc_id": "doc_test_1",
        "doc_title": "Notes on Attention",
        "content": "# Transformers\n\nAttention mechanisms compute query key value dot products.",
        "media_type": "markdown",
    }
    res_ingest = client.post("/api/v1/documents/ingest", json=ingest_data)
    assert res_ingest.status_code == 200
    assert res_ingest.json()["chunks_indexed"] >= 1

    res_search = client.post(
        "/api/v1/documents/cross-search",
        json={"query": "Attention dot products", "top_k": 5},
    )
    assert res_search.status_code == 200
    data = res_search.json()
    assert len(data["matches"]) >= 1


def test_api_privacy_redact(client):
    res = client.post(
        "/api/v1/privacy/redact",
        json={"text": "Contact user at john.doe@example.com or 415-555-0199"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["was_modified"] is True
    assert "[REDACTED_EMAIL]" in data["redacted_text"]


def test_api_observability_slo(client):
    res = client.get("/api/v1/observability/slo")
    assert res.status_code == 200
    data = res.json()
    assert "slo" in data
    assert "cost" in data
    assert "feature_flags" in data


def test_api_lineage_trace(client):
    n1 = LineageNode(node_id="n_parent", node_type=LineageNodeType.MEDIA_FILE, name="video.mp4", content_hash="hash1")
    n2 = LineageNode(node_id="n_child", node_type=LineageNodeType.TRANSCRIPT_SEGMENT, name="seg1", content_hash="hash2")
    global_lineage_tracker.register_node(n1)
    global_lineage_tracker.register_node(n2)
    global_lineage_tracker.record_derivation("n_parent", "n_child", "transcription")

    res = client.get("/api/v1/lineage/trace?node_id=n_child")
    assert res.status_code == 200
    data = res.json()
    ancestor_ids = [a["node_id"] for a in data["ancestors"]]
    assert "n_parent" in ancestor_ids
