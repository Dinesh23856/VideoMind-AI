"""URL validation against SSRF and secure filename sanitization."""

from __future__ import annotations

import ipaddress
import os
import re
import socket
from urllib.parse import urlparse

from app.core.errors import SSRFSecurityError, ValidationError


BLOCKED_NETWORKS = [
    # IPv4 Private, Local, Reserved, CGNAT, Metadata
    ipaddress.ip_network("0.0.0.0/8"),         # Broadcast / current network
    ipaddress.ip_network("10.0.0.0/8"),        # RFC 1918 Private
    ipaddress.ip_network("100.64.0.0/10"),     # RFC 6598 Shared Address Space / CGNAT
    ipaddress.ip_network("127.0.0.0/8"),       # Loopback
    ipaddress.ip_network("169.254.0.0/16"),    # Link-local / Cloud metadata (AWS/GCP/Azure)
    ipaddress.ip_network("172.16.0.0/12"),     # RFC 1918 Private
    ipaddress.ip_network("192.0.0.0/24"),      # IETF Protocol Assignments
    ipaddress.ip_network("192.0.2.0/24"),      # TEST-NET-1 documentation
    ipaddress.ip_network("192.168.0.0/16"),    # RFC 1918 Private
    ipaddress.ip_network("198.18.0.0/15"),     # Network benchmark tests
    ipaddress.ip_network("198.51.100.0/24"),   # TEST-NET-2 documentation
    ipaddress.ip_network("203.0.113.0/24"),    # TEST-NET-3 documentation
    ipaddress.ip_network("224.0.0.0/4"),       # Multicast
    ipaddress.ip_network("240.0.0.0/4"),       # Reserved / Future use
    ipaddress.ip_network("255.255.255.255/32"),# Limited broadcast
    # IPv6 Special, Local, Documentation, Multicast
    ipaddress.ip_network("::/128"),            # Unspecified address
    ipaddress.ip_network("::1/128"),           # IPv6 loopback
    ipaddress.ip_network("::ffff:0:0/96"),     # IPv4-mapped IPv6
    ipaddress.ip_network("100::/64"),          # Discard prefix
    ipaddress.ip_network("2001:db8::/32"),     # Documentation
    ipaddress.ip_network("fc00::/7"),          # Unique local (ULA)
    ipaddress.ip_network("fe80::/10"),         # Link-local unicast
    ipaddress.ip_network("ff00::/8"),          # Multicast
]

BLOCKED_HOSTNAMES = {
    "localhost",
    "127.0.0.1",
    "::1",
    "0.0.0.0",
    "metadata.google.internal",
    "metadata.aws.internal",
    "instance-data",
    "169.254.169.254",
}

ALLOWED_MEDIA_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".webm",
    ".flv",
    ".mp3",
    ".wav",
    ".m4a",
    ".aac",
}


def is_ip_blocked(ip: str | ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Checks if an IP address belongs to any restricted or private network."""
    if isinstance(ip, str):
        try:
            ip = ipaddress.ip_address(ip)
        except ValueError:
            return True

    # Check IPv4-mapped IPv6 addresses (e.g. ::ffff:127.0.0.1)
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped

    # Check NAT64 / DNS64 prefix (64:ff9b::/96) - validate embedded IPv4 address
    if isinstance(ip, ipaddress.IPv6Address) and ip in ipaddress.ip_network("64:ff9b::/96"):
        embedded_ipv4 = ipaddress.IPv4Address(ip.packed[-4:])
        return is_ip_blocked(embedded_ipv4)

    for blocked in BLOCKED_NETWORKS:
        if ip in blocked:
            return True
    return False


def validate_external_url(url: str, resolve_dns: bool = True) -> str:
    """
    Validates URL to protect against SSRF, DNS rebinding, and malicious schemes.
    Blocks private IP addresses, loopback, cloud metadata endpoints, CGNAT, and non-HTTP schemes.
    Resolves DNS and verifies ALL resolved socket addresses.
    """
    if not url or not isinstance(url, str):
        raise ValidationError("URL must be a non-empty string.")

    url = url.strip()
    parsed = urlparse(url)

    if parsed.scheme.lower() not in ("http", "https"):
        raise SSRFSecurityError(f"Unsupported URL scheme '{parsed.scheme}'. Only HTTP and HTTPS are permitted.")

    if parsed.username or parsed.password:
        raise SSRFSecurityError("URLs containing user credentials/userinfo are forbidden (SSRF protection).")

    hostname = parsed.hostname
    if not hostname:
        raise SSRFSecurityError("Invalid URL: hostname is missing.")

    hostname_lower = hostname.lower()

    # Block direct prohibited hostname literals
    if hostname_lower in BLOCKED_HOSTNAMES or any(
        hostname_lower.endswith(suffix) for suffix in (".local", ".internal", ".localhost", ".lan", ".home")
    ):
        raise SSRFSecurityError(f"Targeting host '{hostname}' is forbidden (SSRF protection).")

    # If hostname is a raw IP literal or decimal/octal/hex encoded IP trick, validate immediately
    try:
        ip_obj = ipaddress.ip_address(hostname)
        if is_ip_blocked(ip_obj):
            raise SSRFSecurityError(f"Target IP address '{hostname}' is inside a restricted network.")
        return url
    except ValueError:
        pass

    # Check for decimal, octal, hex, or packed numeric IP tricks
    try:
        packed = socket.inet_aton(hostname)
        ip_obj = ipaddress.IPv4Address(packed)
        if is_ip_blocked(ip_obj):
            raise SSRFSecurityError(f"Target IP address '{hostname}' ({ip_obj}) is inside a restricted network.")
        return url
    except (socket.error, OSError, ValueError):
        pass

    if hostname.isdigit() or hostname.startswith("0x") or hostname.startswith("0X"):
        try:
            val = int(hostname, 0)
            ip_obj = ipaddress.IPv4Address(val)
            if is_ip_blocked(ip_obj):
                raise SSRFSecurityError(f"Target IP address '{hostname}' ({ip_obj}) is inside a restricted network.")
            return url
        except (ValueError, OverflowError):
            pass

    if resolve_dns:
        try:
            addr_info = socket.getaddrinfo(hostname, None)
        except socket.gaierror as exc:
            raise SSRFSecurityError(f"Cannot resolve hostname '{hostname}' for SSRF verification: {exc}")

        if not addr_info:
            raise SSRFSecurityError(f"No IP addresses resolved for hostname '{hostname}'.")

        for _, _, _, _, sockaddr in addr_info:
            ip_str = sockaddr[0]
            try:
                resolved_ip = ipaddress.ip_address(ip_str)
                if is_ip_blocked(resolved_ip):
                    raise SSRFSecurityError(
                        f"Resolved address '{ip_str}' for host '{hostname}' is inside a restricted network."
                    )
            except ValueError:
                raise SSRFSecurityError(f"Malformed resolved IP address '{ip_str}' for host '{hostname}'.")

    return url


def validate_redirect_url(target_url: str, original_url: Optional[str] = None) -> str:
    """
    Validates HTTP redirect targets against SSRF.
    Prevents a legitimate public URL from redirecting into internal services or cloud metadata.
    """
    return validate_external_url(target_url, resolve_dns=True)


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
