"""
VideoMind-AI Python SDK Client: Programmatic access to the Video-to-Knowledge OS.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import urllib.request
import urllib.parse
import json


class VideoMindClient:
    """
    Official Python SDK client for VideoMind-AI REST API.
    Provides complete programmatic access to ingestion, search, RAG, AI tutor,
    knowledge graphs, courses, and system telemetry.
    """

    def __init__(self, base_url: str = "http://localhost:8000", access_token: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.access_token = access_token

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        return headers

    def _request(self, method: str, path: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        req_data = json.dumps(data).encode("utf-8") if data else None
        req = urllib.request.Request(url, data=req_data, headers=self._headers(), method=method)

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as err:
            err_body = err.read().decode("utf-8")
            try:
                err_json = json.loads(err_body)
                msg = err_json.get("detail", err_body)
            except Exception:
                msg = err_body
            raise RuntimeError(f"API Error {err.code}: {msg}") from err

    def login(self, email: str, password: str) -> str:
        """Authenticates and sets active JWT bearer token."""
        res = self._request("POST", "/api/v1/auth/login", {"email": email, "password": password})
        self.access_token = res.get("access_token")
        return self.access_token or ""

    def submit_video_url(self, url: str, title: Optional[str] = None) -> Dict[str, Any]:
        """Submits video URL for processing."""
        return self._request("POST", "/api/v1/jobs/url", {"url": url, "title": title})

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Fetches status and metadata for a job."""
        return self._request("GET", f"/api/v1/jobs/{job_id}")

    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """Cancels an in-flight background job."""
        return self._request("POST", f"/api/v1/jobs/{job_id}/cancel")

    def search(self, query: str, k: int = 5, video_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Executes hybrid search across video libraries."""
        path = f"/api/v1/search?q={urllib.parse.quote(query)}&k={k}"
        if video_id:
            path += f"&video_id={urllib.parse.quote(video_id)}"
        return self._request("GET", path).get("results", [])

    def ask_tutor(self, message: str, mode: str = "Intermediate", video_id: Optional[str] = None) -> Dict[str, Any]:
        """Queries the Socratic AI Tutor."""
        payload = {"message": message, "mode": mode, "video_id": video_id}
        return self._request("POST", "/api/v1/tutor/chat", payload)

    def get_knowledge_graph(self, job_id: str) -> Dict[str, Any]:
        """Retrieves Cytoscape graph nodes and edges."""
        return self._request("GET", f"/api/v1/knowledge/graph/{job_id}")

    def create_course(self, title: str, video_ids: List[str], description: str = "") -> Dict[str, Any]:
        """Compiles a structured curriculum from video IDs."""
        return self._request("POST", "/api/v1/courses", {
            "title": title,
            "video_ids": video_ids,
            "description": description,
        })

    def get_health(self) -> Dict[str, Any]:
        """Checks API health and service metadata."""
        return self._request("GET", "/health")
