from fastapi.testclient import TestClient
from app.api.main import app

client = TestClient(app, raise_server_exceptions=False)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "VideoMind AI" in response.text


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "service" in data


def test_jobs_url_invalid_input():
    response = client.post("/api/v1/jobs/url", json={})
    assert response.status_code == 422


def test_artifacts_nonexistent_job():
    response = client.get("/api/v1/jobs/fake-id-12345/artifacts/markdown")
    assert response.status_code == 404


def test_spa_direct_refresh_routes():
    routes = ["/dashboard", "/video/vid_123", "/search", "/tutor", "/knowledge", "/settings"]
    for route in routes:
        res = client.get(route)
        assert res.status_code == 200
        assert "<html" in res.text.lower()
        assert "VideoMind AI" in res.text


def test_api_nonexistent_returns_404():
    res = client.get("/api/v1/nonexistent_route")
    assert res.status_code == 404

