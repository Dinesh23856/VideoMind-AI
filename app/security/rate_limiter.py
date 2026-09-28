"""Sliding window rate limiter for API denial-of-service and abuse protection."""

from __future__ import annotations

import time
from typing import Dict, List
from app.core.errors import RateLimitExceededError


class InMemoryRateLimiter:
    """
    Sliding window in-memory rate limiter per client IP or user token.
    Tracks timestamped requests and purges expired entries outside the window.
    """

    def __init__(self, requests_per_minute: int = 60):
        self.limit = requests_per_minute
        self.history: Dict[str, List[float]] = {}

    def check(self, client_identifier: str) -> None:
        now = time.time()
        window_start = now - 60.0

        if client_identifier not in self.history:
            self.history[client_identifier] = []

        # Retain only timestamps within the last 60 seconds
        self.history[client_identifier] = [
            t for t in self.history[client_identifier] if t > window_start
        ]

        if len(self.history[client_identifier]) >= self.limit:
            raise RateLimitExceededError(
                f"Rate limit exceeded: {self.limit} requests per minute allowed."
            )

        self.history[client_identifier].append(now)


_global_limiter = InMemoryRateLimiter(requests_per_minute=120)


def get_rate_limiter() -> InMemoryRateLimiter:
    return _global_limiter
