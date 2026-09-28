"""URL validation against SSRF and secure filename sanitization."""

from __future__ import annotations

import ipaddress
import os
import re
import socket
from urllib.parse import urlparse

from app.core.errors import SSRFSecurityError, ValidationError


BLOCKED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),       # Loopback
    ipaddress.ip_network("10.0.0.0/8"),        # RFC 1918 Private
    ipaddress.ip_network("172.16.0.0/12"),     # RFC 1918 Private
    ipaddress.ip_network("192.168.0.0/16"),    # RFC 1918 Private
    ipaddress.ip_network("169.254.0.0/16"),    # Link-local / Cloud metadata (AWS/GCP/Azure)
    ipaddress.ip_network("0.0.0.0/8"),         # Broadcast/Current network
    ipaddress.ip_network("::1/128"),           # IPv6 loopback
    ipaddress.ip_network("fc00::/7"),          # IPv6 Unique local
    ipaddress.ip_network("fe80::/10"),         # IPv6 Link-local
]

ALLOWED_MEDIA_EXTENSIONS = {
    ".mp4", ".mkv", ".mov", ".webm", ".avi", ".flv",
    ".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac",
}


def validate_external_url(url: str) -> str:
    """
    Validates URL to protect against SSRF and malicious protocol schemes.
    Blocks private IP addresses, loopback, cloud metadata endpoints, and non-HTTP schemes.
    """
    if not url or not isinstance(url, str):
        raise ValidationError("URL must be a non-empty string.")

    url = url.strip()
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        raise SSRFSecurityError(f"Unsupported URL scheme '{parsed.scheme}'. Only HTTP and HTTPS are permitted.")

    hostname = parsed.hostname
    if not hostname:
        raise SSRFSecurityError("Invalid URL: hostname is missing.")

    # Check for direct localhost naming
    if hostname.lower() in ("localhost", "127.0.0.1", "::1", "metadata.google.internal"):
        raise SSRFSecurityError(f"Targeting host '{hostname}' is forbidden (SSRF protection).")

    # Resolve IP address to detect internal / cloud metadata redirection
    try:
        addr_info = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        # If DNS cannot resolve, permit if it looks like a valid external domain
        if "." not in hostname:
            raise SSRFSecurityError(f"Cannot resolve hostname '{hostname}'.")
        return url

    for family, _, _, _, sockaddr in addr_info:
        ip_str = sockaddr[0]
        try:
            ip_obj = ipaddress.ip_address(ip_str)
            for blocked in BLOCKED_NETWORKS:
                if ip_obj in blocked:
                    raise SSRFSecurityError(
                        f"Resolved address '{ip_str}' for host '{hostname}' is inside restricted range {blocked}."
                    )
        except ValueError:
            continue

    return url


def sanitize_filename(filename: str, default: str = "uploaded_media.mp4") -> str:
    """
    Removes path traversal sequences, control characters, and enforces allowed media extensions.
    """
    if not filename:
        return default

    # Remove path directory separators
    clean = os.path.basename(filename).strip()
    # Strip null bytes and non-printable characters
    clean = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", clean)
    # Replace unsafe shell characters
    clean = re.sub(r"[^\w\.\- ]", "_", clean)
    clean = clean.strip(" .")

    if not clean:
        return default

    ext = os.path.splitext(clean)[1].lower()
    if ext not in ALLOWED_MEDIA_EXTENSIONS:
        clean = f"{clean}.mp4"

    return clean
