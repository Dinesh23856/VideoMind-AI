"""Tests for JWT security guardrails, production secret validation, and token verification."""

from __future__ import annotations

import time
from unittest.mock import patch

import jwt
import pytest

from config.settings import Settings
from app.core.errors import AuthenticationError
from app.security.auth import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


def test_production_rejects_insecure_jwt_secret():
    # Production with default secret must fail fast
    with pytest.raises(ValueError) as exc:
        Settings(
            app_env="production",
            secret_key="valid_production_secret_key_16_chars",
            jwt_secret_key="videomind-secret-key-change-in-production",
        )
    assert "jwt_secret_key" in str(exc.value).lower() or "secret" in str(exc.value).lower()

    # Production with weak/short secret must fail fast
    with pytest.raises(ValueError) as exc:
        Settings(
            app_env="production",
            secret_key="valid_production_secret_key_16_chars",
            jwt_secret_key="short-secret-1234",
        )
    assert "32 characters" in str(exc.value).lower()


def test_production_accepts_strong_secret():
    strong_secret = "a_super_strong_cryptographically_secure_random_production_secret_key_9876543210"
    s = Settings(
        app_env="production",
        secret_key="valid_production_secret_key_16_chars",
        jwt_secret_key=strong_secret,
    )
    assert s.jwt_secret_key == strong_secret


def test_development_and_testing_permits_default_secret():
    dev_settings = Settings(
        app_env="development",
        jwt_secret_key="videomind-secret-key-change-in-production",
    )
    assert dev_settings.jwt_secret_key is not None

    test_settings = Settings(
        app_env="testing",
        jwt_secret_key="videomind-secret-key-change-in-production",
    )
    assert test_settings.jwt_secret_key is not None


def test_jwt_token_creation_and_decoding():
    user_id = "usr_test123"
    token = create_access_token(user_id=user_id, role="INSTRUCTOR")
    payload = decode_access_token(token)

    assert payload["sub"] == user_id
    assert payload["role"] == "INSTRUCTOR"
    assert "exp" in payload


def test_jwt_tampered_token_rejected():
    token = create_access_token(user_id="usr_normal", role="STUDENT")
    # Tamper with the signature portion
    tampered = token[:-5] + "XXXXX"
    with pytest.raises(AuthenticationError):
        decode_access_token(tampered)


def test_jwt_expired_token_rejected():
    with patch("time.time", return_value=1000000.0):
        # Create token that expired in the past
        token = create_access_token(user_id="usr_expired", role="STUDENT", expires_delta_seconds=-100)

    with pytest.raises(AuthenticationError) as exc_info:
        decode_access_token(token)
    assert "expired" in str(exc_info.value).lower()


def test_password_hashing_and_verification():
    raw = "SuperSecurePassword#2026!"
    hashed = hash_password(raw)
    assert hashed != raw
    assert verify_password(raw, hashed) is True
    assert verify_password("WrongPassword123", hashed) is False
