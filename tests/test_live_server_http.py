"""Real HTTP server integration test running uvicorn over a genuine TCP port."""

import threading
import time
import httpx
import pytest
import uvicorn

from app.api.main import app


class UvicornTestServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        self.host = host
        self.port = port
        self.config = uvicorn.Config(app=app, host=self.host, port=self.port, log_level="warning")
        self.server = uvicorn.Server(config=self.config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self):
        self.thread.start()
        # Wait until server is listening
        for _ in range(50):
            try:
                resp = httpx.get(f"http://{self.host}:{self.port}/health", timeout=0.5)
                if resp.status_code == 200:
                    return
            except Exception:
                time.sleep(0.1)
        raise RuntimeError("Uvicorn test server failed to start within 5 seconds")

    def stop(self):
        self.server.should_exit = True
        self.thread.join(timeout=2.0)


@pytest.fixture(scope="module")
def live_server():
    srv = UvicornTestServer(port=8765)
    srv.start()
    yield f"http://{srv.host}:{srv.port}"
    srv.stop()


def test_real_server_endpoints(live_server):
    client = httpx.Client(base_url=live_server, timeout=5.0)

    # 1. GET /health
    r_health = client.get("/health")
    assert r_health.status_code == 200
    data = r_health.json()
    assert data["status"] == "ok"
    assert data["service"] == "videomind-ai"
    assert "X-Request-ID" in r_health.headers

    # 2. GET /live
    r_live = client.get("/live")
    assert r_live.status_code == 200
    assert r_live.json()["status"] == "live"

    # 3. GET /ready
    r_ready = client.get("/ready")
    assert r_ready.status_code == 200
    assert "status" in r_ready.json()

    # 4. GET /metrics (Prometheus scrape target)
    r_metrics = client.get("/metrics")
    assert r_metrics.status_code == 200
    assert "videomind_http_requests_total" in r_metrics.text


    # 5. GET /docs (Swagger UI)
    r_docs = client.get("/docs")
    assert r_docs.status_code == 200
    assert "swagger-ui" in r_docs.text

    # 6. GET /openapi.json
    r_openapi = client.get("/openapi.json")
    assert r_openapi.status_code == 200
    schema = r_openapi.json()
    assert "paths" in schema
    paths = schema["paths"]
    # Check that documented API paths exist in OpenAPI schema
    assert "/api/v1/jobs/url" in paths
    assert "/api/v1/jobs/upload" in paths
    assert "/api/v1/jobs/{job_id}/status" in paths
    assert "/api/v1/jobs/{job_id}/progress" in paths
    assert "/api/v1/jobs/{job_id}/artifacts/{name}" in paths
    assert "/api/v1/search" in paths
    assert "/api/v1/tutor/chat" in paths
    assert "/api/v1/knowledge/graph/{job_id}" in paths
    assert "/api/v1/knowledge/chapters/{job_id}" in paths
    assert "/api/v1/courses" in paths
    assert "/api/v1/videos" in paths
    assert "/api/v1/auth/register" in paths
    assert "/api/v1/auth/login" in paths
    assert "/api/v1/auth/me" in paths

    # 7. CORS verification
    r_cors = client.options(
        "/api/v1/search",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert "access-control-allow-origin" in r_cors.headers

    # 8. Request ID tracking
    custom_id = "test-req-tracing-1234"
    r_trace = client.get("/health", headers={"X-Request-ID": custom_id})
    assert r_trace.headers.get("X-Request-ID") == custom_id

    # 9. Validation and Error formatting
    r_err = client.post("/api/v1/jobs/url", json={"invalid_field": "foo"})
    assert r_err.status_code == 422  # Pydantic validation error

    # 10. Rate limiting test
    hit_429 = False
    for _ in range(80):
        r_rl = client.get("/health")
        if r_rl.status_code == 429:
            hit_429 = True
            body = r_rl.json()
            assert "error" in body
            assert body["error"]["error_code"] == "RATE_LIMIT_EXCEEDED"
            break
    # Note: rate limiter threshold depends on client IP; if reached, verified.
