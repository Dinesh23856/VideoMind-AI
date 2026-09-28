"""JWT Authentication, secure password hashing, and Role-Based Access Control (RBAC)."""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from typing import Any, Dict, List, Optional
import jwt
from fastapi import Header, HTTPException, status

from app.core.errors import AuthenticationError, AuthorizationError

from app.domain.models import UserRole
from config.settings import get_settings

settings = get_settings()


def hash_password(password: str) -> str:
    """PBKDF2-HMAC-SHA256 password hashing with random 16-byte salt."""
    salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    return f"{salt.hex()}:{key.hex()}"


def verify_password(password: str, hashed: str) -> bool:
    """Constant-time verification of PBKDF2 hashed password."""
    try:
        salt_hex, key_hex = hashed.split(":")
        salt = bytes.fromhex(salt_hex)
        expected_key = bytes.fromhex(key_hex)
        actual_key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
        return hmac.compare_digest(actual_key, expected_key)
    except Exception:
        return False


def create_access_token(
    user_id: str,
    role: str = "USER",
    expires_delta_seconds: Optional[int] = None,
) -> str:
    now = time.time()
    expire = now + (expires_delta_seconds or (settings.jwt_access_token_expire_minutes * 60))
    payload: Dict[str, Any] = {
        "sub": user_id,
        "role": role,
        "iat": int(now),
        "exp": int(expire),
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> Dict[str, Any]:
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise AuthenticationError("Authentication token has expired.")
    except jwt.PyJWTError:
        raise AuthenticationError("Invalid authentication token.")


def check_role_permission(user_role: str, allowed_roles: List[UserRole | str]) -> bool:
    allowed_values = [r.value if isinstance(r, UserRole) else str(r) for r in allowed_roles]
    if user_role == UserRole.ADMIN.value:
        return True
    return user_role in allowed_values


def get_current_user(
    authorization: Optional[str] = Header(None),
) -> Any:
    """FastAPI dependency to extract and validate authenticated user from JWT Bearer token."""
    from fastapi import HTTPException, status
    from app.database.session import SessionLocal
    from app.database.models import User

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header. Expected 'Bearer <token>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = authorization[7:].strip()
    try:
        payload = decode_access_token(token)
    except AuthenticationError as auth_err:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(auth_err),
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token subject payload.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    with SessionLocal() as db:
        user = db.query(User).filter(User.id == user_id).first()
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account not found or disabled.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        # Detach/clone attribute values for use across requests
        return {
            "id": user.id,
            "email": user.email,
            "role": user.role,
            "full_name": user.full_name,
            "is_active": user.is_active,
        }


def get_optional_current_user(
    authorization: Optional[str] = Header(None),
) -> Optional[Dict[str, Any]]:
    """FastAPI dependency for endpoints that accept both authenticated and guest access."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    try:
        return get_current_user(authorization=authorization)
    except Exception:
        return None


def require_role(allowed_roles: List[UserRole | str]):
    """Returns a dependency function that enforces user role RBAC."""
    from fastapi import HTTPException, status

    def role_checker(current_user: Dict[str, Any] = get_current_user) -> Dict[str, Any]:
        user_role = current_user.get("role", "USER")
        if not check_role_permission(user_role, allowed_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required roles: {[str(r) for r in allowed_roles]}, user role: '{user_role}'.",
            )
        return current_user

    return role_checker


def verify_resource_ownership(
    resource_owner_id: Optional[str],
    current_user: Dict[str, Any],
    resource_name: str = "Resource",
) -> None:
    """Enforces that only the resource owner or an ADMIN can access the resource."""
    from fastapi import HTTPException, status

    if current_user.get("role") == UserRole.ADMIN.value:
        return

    if resource_owner_id and resource_owner_id != current_user.get("id"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Forbidden: You do not have ownership access to this {resource_name}.",
        )

