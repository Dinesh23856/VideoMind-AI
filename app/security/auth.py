"""JWT Authentication, secure password hashing, and Role-Based Access Control (RBAC)."""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from typing import Any, Dict, List, Optional
import jwt

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


def check_role_permission(user_role: str, allowed_roles: List[UserRole]) -> bool:
    allowed_values = [r.value for r in allowed_roles]
    # ADMIN role has superuser access
    if user_role == UserRole.ADMIN.value:
        return True
    return user_role in allowed_values
