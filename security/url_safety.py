"""
URL safety checks for external fetch tools.
"""
from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True)
class UrlSafetyDecision:
    allowed: bool
    reason: str


@dataclass(frozen=True)
class FetchSafetyLimits:
    max_redirects: int = 3
    max_response_bytes: int = 2_000_000
    allowed_content_types: tuple[str, ...] = (
        "text/html",
        "text/plain",
        "application/xhtml+xml",
    )


@dataclass(frozen=True)
class ResponseSafetyDecision:
    allowed: bool
    reason: str


def _is_private_address(hostname: str) -> bool:
    try:
        addresses = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return False

    for item in addresses:
        raw_ip = item[4][0]
        try:
            ip = ipaddress.ip_address(raw_ip)
        except ValueError:
            continue
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return True
    return False


def check_url_safety(url: str) -> UrlSafetyDecision:
    if not url:
        return UrlSafetyDecision(False, "URL is empty")
    if len(url) > 2048:
        return UrlSafetyDecision(False, "URL is too long")

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return UrlSafetyDecision(False, f"Unsupported URL scheme: {parsed.scheme or '(missing)'}")
    if not parsed.hostname:
        return UrlSafetyDecision(False, "URL hostname is missing")
    if parsed.username or parsed.password:
        return UrlSafetyDecision(False, "URL credentials are not allowed")

    hostname = parsed.hostname.lower()
    if hostname in {"localhost", "localhost.localdomain"} or hostname.endswith(".localhost"):
        return UrlSafetyDecision(False, "Localhost URLs are not allowed")

    try:
        ip = ipaddress.ip_address(hostname)
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return UrlSafetyDecision(False, "Private or local network addresses are not allowed")
    except ValueError:
        if _is_private_address(hostname):
            return UrlSafetyDecision(False, "Hostname resolves to a private or local network address")

    return UrlSafetyDecision(True, "URL passed safety checks")


def check_response_safety(
    content_type: str = "",
    content_length: str | int | None = None,
    limits: FetchSafetyLimits | None = None,
) -> ResponseSafetyDecision:
    limits = limits or FetchSafetyLimits()
    normalized_type = (content_type or "").split(";", 1)[0].strip().lower()
    if normalized_type and normalized_type not in limits.allowed_content_types:
        return ResponseSafetyDecision(False, f"Unsupported response content type: {normalized_type}")

    if content_length not in {None, ""}:
        try:
            length = int(content_length)
        except (TypeError, ValueError):
            return ResponseSafetyDecision(False, "Invalid Content-Length header")
        if length > limits.max_response_bytes:
            return ResponseSafetyDecision(False, "Response body is too large")

    return ResponseSafetyDecision(True, "Response headers passed safety checks")
