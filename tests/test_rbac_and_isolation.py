"""Comprehensive RBAC, JWT lifecycle, and Cross-User Resource Isolation Tests."""

import uuid
import pytest
from starlette.testclient import TestClient

from app.api.main import app
from app.database.session import SessionLocal
from app.database.models import BookmarkRecord, CourseRecord, JobRecord, User, VideoRecord
from app.security.auth import create_access_token


@pytest.fixture
def client():
    return TestClient(app)


def test_auth_and_cross_user_isolation(client):
    uid_a = f"test_a_{uuid.uuid4().hex[:6]}"
    uid_b = f"test_b_{uuid.uuid4().hex[:6]}"
    uid_admin = f"test_admin_{uuid.uuid4().hex[:6]}"

    # 1. Register User A, User B, and Admin
    reg_a = client.post(
        "/api/v1/auth/register",
        json={"email": f"{uid_a}@example.com", "password": "PasswordA123!", "role": "USER"},
    )
    assert reg_a.status_code == 201
    user_a = reg_a.json()
    token_a = user_a["token"]

    reg_b = client.post(
        "/api/v1/auth/register",
        json={"email": f"{uid_b}@example.com", "password": "PasswordB123!", "role": "USER"},
    )
    assert reg_b.status_code == 201
    user_b = reg_b.json()
    token_b = user_b["token"]

    reg_adm = client.post(
        "/api/v1/auth/register",
        json={"email": f"{uid_admin}@example.com", "password": "AdminPassword123!", "role": "ADMIN"},
    )
    assert reg_adm.status_code == 201
    admin_user = reg_adm.json()
    token_admin = admin_user["token"]

    # 2. Login verification
    login_a = client.post(
        "/api/v1/auth/login",
        json={"email": f"{uid_a}@example.com", "password": "PasswordA123!"},
    )
    assert login_a.status_code == 200
    assert "token" in login_a.json()

    # Invalid login
    login_fail = client.post(
        "/api/v1/auth/login",
        json={"email": f"{uid_a}@example.com", "password": "WrongPassword"},
    )
    assert login_fail.status_code == 401

    # 3. GET /api/v1/auth/me
    me_a = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token_a}"})
    assert me_a.status_code == 200
    assert me_a.json()["id"] == user_a["user_id"]

    # Invalid token rejected
    me_invalid = client.get("/api/v1/auth/me", headers={"Authorization": "Bearer invalid.jwt.token"})
    assert me_invalid.status_code == 401

    # Expired token rejected
    expired_token = create_access_token(user_a["user_id"], expires_delta_seconds=-10)
    me_expired = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {expired_token}"})
    assert me_expired.status_code == 401

    # 4. Populate resources owned by User A in database
    vid_a_id = f"vid_{uuid.uuid4().hex[:8]}"
    job_a_id = f"job_{uuid.uuid4().hex[:8]}"
    course_a_id = f"crs_{uuid.uuid4().hex[:8]}"

    with SessionLocal() as db:
        video_a = VideoRecord(
            id=vid_a_id,
            user_id=user_a["user_id"],
            title="User A Private Lecture",
            duration_seconds=300.0,
        )
        job_a = JobRecord(
            id=job_a_id,
            user_id=user_a["user_id"],
            video_id=vid_a_id,
            status="COMPLETED",
            progress=100.0,
            message="Done",
        )
        course_a = CourseRecord(
            id=course_a_id,
            user_id=user_a["user_id"],
            title="User A Private Course",
            structure_json='{"id": "' + course_a_id + '", "modules": []}',
        )
        db.add_all([video_a, job_a, course_a])
        db.commit()

    # 5. User A accesses own resources -> 200 OK
    res_a = client.get(f"/api/v1/videos/{vid_a_id}", headers={"Authorization": f"Bearer {token_a}"})
    assert res_a.status_code == 200
    assert res_a.json()["id"] == vid_a_id

    # 6. User B attempts unauthorized access to User A's resources -> 403 Forbidden!
    # Read video
    res_b_vid = client.get(f"/api/v1/videos/{vid_a_id}", headers={"Authorization": f"Bearer {token_b}"})
    assert res_b_vid.status_code == 403

    # Delete video
    res_b_del_vid = client.delete(f"/api/v1/videos/{vid_a_id}", headers={"Authorization": f"Bearer {token_b}"})
    assert res_b_del_vid.status_code == 403

    # Read video transcript
    res_b_trans = client.get(f"/api/v1/videos/{vid_a_id}/transcript", headers={"Authorization": f"Bearer {token_b}"})
    assert res_b_trans.status_code == 403

    # Access job status
    res_b_job = client.get(f"/api/v1/jobs/{job_a_id}/status", headers={"Authorization": f"Bearer {token_b}"})
    assert res_b_job.status_code == 403

    # Access job artifacts
    res_b_art = client.get(f"/api/v1/jobs/{job_a_id}/artifacts/markdown", headers={"Authorization": f"Bearer {token_b}"})
    assert res_b_art.status_code == 403

    # Access knowledge graph
    res_b_kg = client.get(f"/api/v1/knowledge/graph/{job_a_id}", headers={"Authorization": f"Bearer {token_b}"})
    assert res_b_kg.status_code == 403

    # Access tutor chat for User A's video
    res_b_tutor = client.post(
        "/api/v1/tutor/chat",
        json={"message": "Explain this lecture", "video_id": vid_a_id},
        headers={"Authorization": f"Bearer {token_b}"},
    )
    assert res_b_tutor.status_code == 403

    # Read course
    res_b_crs = client.get(f"/api/v1/courses/{course_a_id}", headers={"Authorization": f"Bearer {token_b}"})
    assert res_b_crs.status_code == 403

    # Delete course
    res_b_del_crs = client.delete(f"/api/v1/courses/{course_a_id}", headers={"Authorization": f"Bearer {token_b}"})
    assert res_b_del_crs.status_code == 403

    # 7. Unauthenticated requests to user-owned resources rejected
    assert client.get(f"/api/v1/videos/{vid_a_id}").status_code == 401
    assert client.get(f"/api/v1/jobs/{job_a_id}/status").status_code == 401
    assert client.get(f"/api/v1/courses/{course_a_id}").status_code == 401

    # 8. Admin role bypasses ownership and can inspect/manage resources
    adm_vid = client.get(f"/api/v1/videos/{vid_a_id}", headers={"Authorization": f"Bearer {token_admin}"})
    assert adm_vid.status_code == 200

    adm_job = client.get(f"/api/v1/jobs/{job_a_id}/status", headers={"Authorization": f"Bearer {token_admin}"})
    assert adm_job.status_code == 200

    adm_crs = client.get(f"/api/v1/courses/{course_a_id}", headers={"Authorization": f"Bearer {token_admin}"})
    assert adm_crs.status_code == 200

    # 9. Bookmarks isolation
    bm_res = client.post(
        "/api/v1/bookmarks",
        json={"video_id": vid_a_id, "timestamp": 42.0, "title": "Secret Note", "note_text": "Key formula"},
        headers={"Authorization": f"Bearer {token_a}"},
    )
    assert bm_res.status_code == 200
    bm_id = bm_res.json()["bookmark_id"]

    # User B cannot read User A's bookmarks on video A -> 403 (since video A belongs to User A)
    b_bms = client.get(f"/api/v1/bookmarks/{vid_a_id}", headers={"Authorization": f"Bearer {token_b}"})
    assert b_bms.status_code == 403

    # User B cannot delete User A's bookmark -> 403
    b_del_bm = client.delete(f"/api/v1/bookmarks/{bm_id}", headers={"Authorization": f"Bearer {token_b}"})
    assert b_del_bm.status_code == 403

    # User A can read and delete their own bookmark
    a_bms = client.get(f"/api/v1/bookmarks/{vid_a_id}", headers={"Authorization": f"Bearer {token_a}"})
    assert a_bms.status_code == 200
    assert len(a_bms.json()["bookmarks"]) == 1

    a_del_bm = client.delete(f"/api/v1/bookmarks/{bm_id}", headers={"Authorization": f"Bearer {token_a}"})
    assert a_del_bm.status_code == 200
