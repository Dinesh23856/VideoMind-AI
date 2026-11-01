"""Tests for secure media storage streaming, HTTP Range requests, path traversal, tenant/workspace isolation, and ownership enforcement."""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.database.models import JobRecord, User, VideoRecord
from app.database.session import SessionLocal
from app.security.auth import create_access_token
from app.storage.provider import LocalStorageProvider, get_storage_provider


@pytest.fixture
def client():
    return TestClient(app)


def test_public_file_accessible_anonymously(client):
    """Public files under public/ can be accessed without authentication."""
    storage = get_storage_provider()
    file_key = f"public/samples/{uuid.uuid4().hex}/intro.mp4"
    dummy_data = b"HEADER\x00\x01" + b"P" * 512
    storage.save(file_key, dummy_data)

    response = client.get(f"/api/v1/storage/{file_key}")
    assert response.status_code == 200
    assert response.content == dummy_data
    assert "video/mp4" in response.headers.get("content-type", "")


def test_private_file_anonymous_access_denied(client):
    """Requirement 5: Anonymous access to private stored files is denied with 401."""
    storage = get_storage_provider()
    file_key = f"private_docs/{uuid.uuid4().hex}/confidential.pdf"
    storage.save(file_key, b"Confidential financial statement")

    response = client.get(f"/api/v1/storage/{file_key}")
    assert response.status_code == 401
    assert "Authentication required" in response.json().get("detail", "")


def test_large_file_streaming_memory_safety(client):
    """
    Requirements 1, 4, 9 (P0-A):
    Proves that a large file is NOT loaded into RAM via storage.read().
    Streams in bounded chunks and verifies storage.read is never called.
    """
    storage = get_storage_provider()
    file_key = f"public/large_media/{uuid.uuid4().hex}/test_stream.mp4"

    # Create a 2MB chunked media file
    chunk_size = 64 * 1024
    total_size = 2 * 1024 * 1024  # 2 MB
    total_chunks = total_size // chunk_size
    dest_path = storage._resolve(file_key)
    with open(dest_path, "wb") as f:
        for i in range(total_chunks):
            f.write(bytes([i % 256]) * chunk_size)

    # Patch storage.read to ensure it is NEVER called during streaming download
    with patch.object(storage, "read", side_effect=AssertionError("storage.read() must NOT be called for streaming!")):
        with client.stream("GET", f"/api/v1/storage/{file_key}") as response:
            assert response.status_code == 200
            assert "video/mp4" in response.headers.get("content-type", "")
            received_bytes = 0
            for chunk in response.iter_bytes(chunk_size=chunk_size):
                received_bytes += len(chunk)
            assert received_bytes == total_size


def test_http_range_requests_support(client):
    """Requirement 10, 11 (P0-A): HTTP Range requests return 206 Partial Content with bounded slice."""
    storage = get_storage_provider()
    file_key = f"public/range_media/{uuid.uuid4().hex}/lecture.mp4"
    data = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    storage.save(file_key, data)

    # Request bytes 0 to 9 (10 bytes)
    response = client.get(f"/api/v1/storage/{file_key}", headers={"Range": "bytes=0-9"})
    assert response.status_code == 206
    assert response.content == b"0123456789"
    assert "bytes 0-9/" in response.headers.get("content-range", "")

    # Request bytes 10 to 19 (10 bytes)
    response2 = client.get(f"/api/v1/storage/{file_key}", headers={"Range": "bytes=10-19"})
    assert response2.status_code == 206
    assert response2.content == b"ABCDEFGHIJ"


def test_head_request_support(client):
    """Requirement 12 (P0-A): HEAD request returns 200 OK with correct headers and empty body."""
    storage = get_storage_provider()
    file_key = f"public/head_test/{uuid.uuid4().hex}/video.mp4"
    dummy_data = b"X" * 2048
    storage.save(file_key, dummy_data)

    response = client.head(f"/api/v1/storage/{file_key}")
    assert response.status_code == 200
    assert response.headers.get("content-length") == "2048"
    assert "video/mp4" in response.headers.get("content-type", "")
    assert len(response.content) == 0


def test_path_traversal_rejection(client):
    """Requirement 6 (P0-C) & 7, 9 (P0-A): Rejects ../, encoded traversal, and root escape attempts."""
    # Dot-dot slash (Client-side URL normalization turns this into /api/etc/passwd -> 404, or server catches -> 400/403)
    resp1 = client.get("/api/v1/storage/../../etc/passwd")
    assert resp1.status_code in (400, 403, 404)

    # Encoded dot-dot slash
    resp2 = client.get("/api/v1/storage/%2e%2e/%2e%2e/windows/system32/cmd.exe")
    assert resp2.status_code in (400, 403)

    # Absolute path / drive letter
    resp3 = client.get("/api/v1/storage/C:/boot.ini")
    assert resp3.status_code in (400, 403)

    # Leading slash
    resp4 = client.get("/api/v1/storage//etc/shadow")
    assert resp4.status_code in (400, 403)


def test_symlink_escape_rejection(client, tmp_path):
    """Requirement 7 (P0-C): Symlink pointing outside the storage base dir is strictly rejected."""
    storage = get_storage_provider()
    outside_file = tmp_path / "outside_secret.txt"
    outside_file.write_text("SUPER_SECRET_EXTERNAL_DATA")

    symlink_key = f"public/links/symlink_{uuid.uuid4().hex}.txt"
    symlink_path = storage.base_dir / symlink_key
    symlink_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        os.symlink(str(outside_file), str(symlink_path))
        resp = client.get(f"/api/v1/storage/{symlink_key}")
        assert resp.status_code in (403, 404)
    except (OSError, NotImplementedError):
        # On Windows without developer symlink privileges, test provider resolution guard directly
        with pytest.raises(PermissionError):
            storage._resolve("../../outside_secret.txt")


def test_nonexistent_file_safe_404(client):
    """Requirement 8 (P0-C): Requesting nonexistent file returns safe 404."""
    user_id = f"usr_{uuid.uuid4().hex[:8]}"
    token = create_access_token(user_id, role="STUDENT")

    response = client.get(
        f"/api/v1/storage/public/nonexistent_{uuid.uuid4().hex}.dat",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 404
    assert "Storage object not found" in response.json().get("detail", "")


def test_user_ownership_isolation(client):
    """
    Requirements 1 & 2 (P0-C):
    User A accesses User A file -> 200 OK.
    User A accesses User B file -> 403 Forbidden.
    """
    user_a = f"usr_a_{uuid.uuid4().hex[:8]}"
    user_b = f"usr_b_{uuid.uuid4().hex[:8]}"
    job_a = f"job_a_{uuid.uuid4().hex[:8]}"
    job_b = f"job_b_{uuid.uuid4().hex[:8]}"

    with SessionLocal() as db:
        u_a = User(id=user_a, email=f"{user_a}@test.com", hashed_password="pw", role="STUDENT")
        u_b = User(id=user_b, email=f"{user_b}@test.com", hashed_password="pw", role="STUDENT")
        j_a = JobRecord(id=job_a, user_id=user_a, status="COMPLETED")
        j_b = JobRecord(id=job_b, user_id=user_b, status="COMPLETED")
        db.add_all([u_a, u_b, j_a, j_b])
        db.commit()

    storage = get_storage_provider()
    key_a = f"uploads/{job_a}_video.mp4"
    key_b = f"uploads/{job_b}_video.mp4"
    storage.save(key_a, b"User A Media Content")
    storage.save(key_b, b"User B Media Content")

    token_a = create_access_token(user_a, role="STUDENT")
    token_b = create_access_token(user_b, role="STUDENT")

    # 1. User A accesses User A file -> 200 OK
    resp_a_owns = client.get(f"/api/v1/storage/{key_a}", headers={"Authorization": f"Bearer {token_a}"})
    assert resp_a_owns.status_code == 200
    assert resp_a_owns.content == b"User A Media Content"

    # 2. User A accesses User B file -> 403 Forbidden
    resp_a_steals = client.get(f"/api/v1/storage/{key_b}", headers={"Authorization": f"Bearer {token_a}"})
    assert resp_a_steals.status_code == 403
    assert "ownership access" in resp_a_steals.json().get("detail", "").lower()

    # 3. User B accesses User B file -> 200 OK
    resp_b_owns = client.get(f"/api/v1/storage/{key_b}", headers={"Authorization": f"Bearer {token_b}"})
    assert resp_b_owns.status_code == 200
    assert resp_b_owns.content == b"User B Media Content"


def test_tenant_boundary_isolation(client):
    """Requirement 3 (P0-C): Tenant A accesses Tenant B file -> 403 Forbidden."""
    user_tenant_alpha = f"usr_{uuid.uuid4().hex[:8]}"
    user_tenant_beta = f"usr_{uuid.uuid4().hex[:8]}"

    with SessionLocal() as db:
        u_alpha = User(id=user_tenant_alpha, email=f"{user_tenant_alpha}@test.com", hashed_password="pw", role="STUDENT")
        u_beta = User(id=user_tenant_beta, email=f"{user_tenant_beta}@test.com", hashed_password="pw", role="STUDENT")
        db.add_all([u_alpha, u_beta])
        db.commit()

    storage = get_storage_provider()
    file_key_beta = "tenants/tenant_beta/workspaces/ws_main/data.json"
    storage.save(file_key_beta, b'{"tenant": "beta_data"}')

    token_alpha = create_access_token(user_tenant_alpha, role="STUDENT", tenant_id="tenant_alpha", workspace_id="ws_main")
    token_beta = create_access_token(user_tenant_beta, role="STUDENT", tenant_id="tenant_beta", workspace_id="ws_main")

    # Tenant Alpha user tries to access Tenant Beta's file -> 403 Forbidden
    resp_forbidden = client.get(f"/api/v1/storage/{file_key_beta}", headers={"Authorization": f"Bearer {token_alpha}"})
    assert resp_forbidden.status_code == 403
    assert "tenant" in resp_forbidden.json().get("detail", "").lower()

    # Tenant Beta user accesses their own file -> 200 OK
    resp_allowed = client.get(f"/api/v1/storage/{file_key_beta}", headers={"Authorization": f"Bearer {token_beta}"})
    assert resp_allowed.status_code == 200
    assert resp_allowed.content == b'{"tenant": "beta_data"}'


def test_workspace_boundary_isolation(client):
    """Requirement 4 (P0-C): Workspace A accesses Workspace B file -> 403 Forbidden."""
    user_ws1 = f"usr_{uuid.uuid4().hex[:8]}"
    user_ws2 = f"usr_{uuid.uuid4().hex[:8]}"

    with SessionLocal() as db:
        u_ws1 = User(id=user_ws1, email=f"{user_ws1}@test.com", hashed_password="pw", role="STUDENT")
        u_ws2 = User(id=user_ws2, email=f"{user_ws2}@test.com", hashed_password="pw", role="STUDENT")
        db.add_all([u_ws1, u_ws2])
        db.commit()

    storage = get_storage_provider()
    file_key_ws2 = "tenants/tenant_alpha/workspaces/ws_finance/report.pdf"
    storage.save(file_key_ws2, b"%PDF-1.4 financial figures")

    token_ws1 = create_access_token(user_ws1, role="STUDENT", tenant_id="tenant_alpha", workspace_id="ws_engineering")
    token_ws2 = create_access_token(user_ws2, role="STUDENT", tenant_id="tenant_alpha", workspace_id="ws_finance")

    # User in engineering workspace tries to access finance workspace file -> 403 Forbidden
    resp_forbidden = client.get(f"/api/v1/storage/{file_key_ws2}", headers={"Authorization": f"Bearer {token_ws1}"})
    assert resp_forbidden.status_code == 403
    assert "workspace" in resp_forbidden.json().get("detail", "").lower()

    # User in finance workspace accesses their own workspace file -> 200 OK
    resp_allowed = client.get(f"/api/v1/storage/{file_key_ws2}", headers={"Authorization": f"Bearer {token_ws2}"})
    assert resp_allowed.status_code == 200
    assert resp_allowed.content == b"%PDF-1.4 financial figures"
