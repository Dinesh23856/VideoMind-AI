"""Tests for modularized API router contracts, health checks, and backward compatibility."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.api.main as main_mod
from app.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_routes(client):
    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"
    assert r_health.json()["service"] == "videomind-ai"

    r_live = client.get("/live")
    assert r_live.status_code == 200
    assert r_live.json()["status"] == "live"

    r_ready = client.get("/ready")
    assert r_ready.status_code == 200
    assert "status" in r_ready.json()

    r_metrics = client.get("/metrics")
    assert r_metrics.status_code == 200
    assert "videomind" in r_metrics.text or "process" in r_metrics.text or "python" in r_metrics.text


def test_ui_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "html" in resp.headers.get("content-type", "").lower()


def test_openapi_contains_core_router_paths():
    openapi_spec = app.openapi()
    paths = openapi_spec.get("paths", {})

    expected_routes = [
        "/health",
        "/live",
        "/ready",
        "/metrics",
        "/api/v1/auth/register",
        "/api/v1/auth/login",
        "/api/v1/auth/me",
        "/api/v1/storage/{file_path}",
        "/api/v1/jobs/url",
        "/api/v1/jobs/upload",
        "/api/v1/jobs",
        "/api/v1/videos",
        "/api/v1/search",
        "/api/v1/rag/query",
        "/api/v1/rag/search",
        "/api/v1/rag/health",
        "/api/v1/knowledge/graph",
        "/api/v1/knowledge/entities",
        "/api/v1/knowledge/temporal",
        "/api/v1/tutor/chat",
        "/api/v1/courses",
        "/api/ai/chat",
        "/api/v1/ai/chat",
        "/api/v1/bookmarks",
        "/api/v1/collections",
        "/api/v1/reasoning/compare",
    ]

    for route in expected_routes:
        assert route in paths, f"Expected route {route} missing from registered OpenAPI schema"


def test_backward_compatible_main_exports():
    """Verify that symbols imported by legacy tests remain present in app.api.main."""
    assert hasattr(main_mod, "global_knowledge_graph")
    assert hasattr(main_mod, "global_search_engine")
    assert hasattr(main_mod, "global_rag_engine")
    assert hasattr(main_mod, "global_tutor_engine")
    assert hasattr(main_mod, "global_event_stream")
    assert hasattr(main_mod, "global_kvc")
    assert hasattr(main_mod, "global_lineage_tracker")
    assert hasattr(main_mod, "JobAccepted")
    assert hasattr(main_mod, "UrlJobRequest")
    assert hasattr(main_mod, "TutorChatRequest")
