"""Secure storage retrieval router with streaming chunk delivery and authorization."""

from __future__ import annotations

import mimetypes
import urllib.parse
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import FileResponse, StreamingResponse

from app.database.models import JobRecord, VideoRecord
from app.database.session import SessionLocal
from app.security.auth import get_optional_current_user, verify_resource_ownership
from app.storage.provider import get_storage_provider

router = APIRouter(prefix="/api/v1/storage", tags=["Storage"])


def _verify_storage_ownership(file_path: str, current_user: Optional[Dict[str, Any]]) -> None:
    """
    Enforces the complete authorization model:
    Authenticated user -> Tenant boundary -> Workspace boundary -> Resource ownership.
    Anonymous access is strictly prohibited except for files explicitly located in 'public/'.
    """
    clean = file_path.replace("\\", "/").strip("/")

    # Public files are explicitly accessible without authentication
    if clean.startswith("public/"):
        return

    # All private files strictly require an authenticated user
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required to access private storage.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Admin role has global administrative clearance across all tenants/workspaces
    if current_user.get("role") == "ADMIN":
        return

    user_tenant = current_user.get("tenant_id") or "default"
    user_workspace = current_user.get("workspace_id") or "default"
    user_id = current_user.get("id")

    parts = [p for p in clean.split("/") if p]

    # 1. Multi-tenant and multi-workspace boundary checking
    path_tenant: Optional[str] = None
    path_workspace: Optional[str] = None

    if len(parts) >= 4 and parts[0] == "tenants" and parts[2] == "workspaces":
        path_tenant = parts[1]
        path_workspace = parts[3]
    elif len(parts) >= 3 and parts[0] not in ("uploads", "videos", "transcripts", "public", "jobs", "users"):
        # e.g. {tenant_id}/{workspace_id}/{filename}
        path_tenant = parts[0]
        path_workspace = parts[1]

    if path_tenant and path_tenant != user_tenant:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Forbidden: Cross-tenant storage access denied. Tenant '{path_tenant}' is not accessible.",
        )
    if path_workspace and path_workspace != user_workspace:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Forbidden: Cross-workspace storage access denied. Workspace '{path_workspace}' is not accessible.",
        )

    # 2. User folder scoping: e.g. users/{user_id}/...
    if parts and parts[0] == "users" and len(parts) > 1:
        target_uid = parts[1]
        if target_uid != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: You do not have ownership access to this user storage.",
            )
        return

    # 3. Job and Video database resource ownership
    first_part = parts[0] if parts else ""
    potential_id: Optional[str] = None

    if first_part == "uploads" and len(parts) > 1:
        upload_name = parts[1]
        potential_id = upload_name.split("_")[0] if "_" in upload_name else upload_name
    elif first_part in ("videos", "transcripts") and len(parts) > 1:
        potential_id = parts[1].split(".")[0]
    elif first_part == "jobs" and len(parts) > 1:
        potential_id = parts[1]
    else:
        potential_id = first_part

    if potential_id:
        with SessionLocal() as db:
            # Check JobRecord
            job = db.query(JobRecord).filter(JobRecord.id == potential_id).first()
            if not job and first_part == "uploads" and len(parts) > 1:
                upload_name = parts[1]
                for j in db.query(JobRecord).all():
                    if upload_name.startswith(j.id):
                        job = j
                        break

            if job and job.user_id:
                if job.user_id != user_id:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Forbidden: You do not have ownership access to this storage file.",
                    )
                return

            # Check VideoRecord
            video = db.query(VideoRecord).filter(VideoRecord.id == potential_id).first()
            if video and video.user_id:
                if video.user_id != user_id:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Forbidden: You do not have ownership access to this storage file.",
                    )
                return


@router.get("/{file_path:path}")
@router.head("/{file_path:path}")
async def download_storage_file(
    file_path: str,
    request: Request,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    """
    Secure direct storage retrieval.
    Streams media in bounded chunks via FileResponse or StreamingResponse to prevent OOM.
    Supports HTTP Range requests and HEAD metadata requests.
    Enforces path traversal protection and resource ownership.
    """
    # 1. Path traversal hardening (including decoded variants)
    decoded_path = urllib.parse.unquote(file_path)
    if (
        ".." in file_path
        or ".." in decoded_path
        or file_path.startswith("/")
        or decoded_path.startswith("/")
        or "\\" in file_path
        or "\\" in decoded_path
        or (len(file_path) > 1 and file_path[1] == ":")
        or (len(decoded_path) > 1 and decoded_path[1] == ":")
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Path traversal forbidden",
        )

    storage = get_storage_provider()

    # 2. Authorization / ownership verification BEFORE existence to prevent enumeration
    _verify_storage_ownership(file_path, current_user)

    # 3. Check file existence in storage
    try:
        if not storage.exists(file_path):
            raise HTTPException(status_code=404, detail="Storage object not found")
    except (PermissionError, ValueError) as err:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Storage access forbidden: {err}",
        )

    # 4. Detect MIME type
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type:
        mime_type = "application/octet-stream"

    filename = Path(file_path).name

    # 5. Stream media in bounded chunks (never read full file into memory)
    local_path = storage.get_local_path(file_path)
    if local_path and local_path.is_file():
        # FileResponse natively handles HTTP Range, 206 Partial Content, Accept-Ranges, and HEAD requests
        return FileResponse(
            path=local_path,
            media_type=mime_type,
            filename=filename,
            content_disposition_type="inline",
        )

    # Fallback to bounded chunk generator for remote / non-local storage
    file_size = storage.get_size(file_path)
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(file_size),
        "Content-Disposition": f'inline; filename="{filename}"',
    }

    # Range request support for streaming response
    range_header = request.headers.get("range")
    if range_header and range_header.startswith("bytes="):
        try:
            range_spec = range_header[6:].strip()
            start_str, end_str = range_spec.split("-", 1)
            start = int(start_str) if start_str else 0
            end = int(end_str) if end_str else file_size - 1
            if start < file_size and end >= start:
                end = min(end, file_size - 1)
                content_len = end - start + 1
                headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
                headers["Content-Length"] = str(content_len)

                def range_generator():
                    with open(storage._resolve(file_path), "rb") as f:
                        f.seek(start)
                        remaining = content_len
                        chunk_size = 64 * 1024
                        while remaining > 0:
                            n = min(remaining, chunk_size)
                            chunk = f.read(n)
                            if not chunk:
                                break
                            remaining -= len(chunk)
                            yield chunk

                return StreamingResponse(
                    range_generator(),
                    status_code=206,
                    media_type=mime_type,
                    headers=headers,
                )
        except Exception:
            pass

    return StreamingResponse(
        storage.stream(file_path, chunk_size=64 * 1024),
        media_type=mime_type,
        headers=headers,
    )
