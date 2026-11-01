"""Tests for hardened SSRF and DNS rebinding protections."""

from __future__ import annotations

import socket
from unittest.mock import patch

import pytest

from app.core.errors import SSRFSecurityError
from app.ingestion.youtube import harvest_youtube_captions
from app.security.sanitizer import (
    is_ip_blocked,
    validate_external_url,
    validate_redirect_url,
)


@pytest.mark.parametrize(
    "ip_str",
    [
        "127.0.0.1",
        "10.0.0.1",
        "172.16.0.1",
        "192.168.1.1",
        "169.254.169.254",  # AWS/GCP/Azure link-local metadata
        "100.64.0.1",       # Carrier-grade NAT
        "0.0.0.0",
        "::1",              # IPv6 loopback
        "fe80::1",          # IPv6 link-local
        "fc00::1",          # IPv6 unique local
    ],
)
def test_blocked_ip_ranges(ip_str):
    assert is_ip_blocked(ip_str) is True


def test_allowed_public_ips():
    assert is_ip_blocked("8.8.8.8") is False
    assert is_ip_blocked("1.1.1.1") is False


@pytest.mark.parametrize(
    "bad_url",
    [
        "http://127.0.0.1/video.mp4",
        "http://10.1.2.3:8080/lecture.mp4",
        "http://192.168.0.1/stream.m3u8",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/video.mp4",
        "http://metadata.google.internal/computeMetadata/v1/",
        "http://instance-data/latest/meta-data/",
    ],
)
def test_direct_ssrf_urls_rejected(bad_url):
    with pytest.raises(SSRFSecurityError):
        validate_external_url(bad_url, resolve_dns=False)


def test_dns_rebinding_resolution_protection():
    # Simulate a domain that resolves to an internal/private IP
    with patch("socket.getaddrinfo", return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))]):
        with pytest.raises(SSRFSecurityError) as exc_info:
            validate_external_url("http://attacker-controlled-domain.com/video.mp4", resolve_dns=True)
        assert "restricted network" in str(exc_info.value).lower()


def test_unresolvable_hostname_rejected():
    with patch("socket.getaddrinfo", side_effect=socket.gaierror(11001, "getaddrinfo failed")):
        with pytest.raises(SSRFSecurityError) as exc_info:
            validate_external_url("http://nonexistent-fake-domain-12345.org/video.mp4", resolve_dns=True)
        assert "resolve hostname" in str(exc_info.value).lower()


def test_redirect_url_validation():
    source_url = "https://public-service.com/video"
    with patch("socket.getaddrinfo", return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]):
        # Safe redirect
        safe_redirect = validate_redirect_url("https://cdn.public-service.com/media/1.mp4", original_url=source_url)
        assert safe_redirect.startswith("https://cdn.public-service.com")

        # Unsafe redirect to internal cloud metadata
        with pytest.raises(SSRFSecurityError):
            validate_redirect_url("http://169.254.169.254/latest/meta-data/", original_url=source_url)


def test_youtube_ingestion_blocks_ssrf_before_ytdlp(tmp_path):
    with pytest.raises(SSRFSecurityError):
        harvest_youtube_captions("http://127.0.0.1:8000/internal_video", str(tmp_path))
