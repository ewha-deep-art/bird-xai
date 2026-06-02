"""Per-IP rate limiting (limits library, in-memory)."""

from __future__ import annotations

from fastapi import HTTPException, Request
from limits import parse
from limits.storage import MemoryStorage
from limits.strategies import MovingWindowRateLimiter


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client is None:
        return "unknown"
    return request.client.host


class RateLimiter:
    def __init__(self, *, interval_sec: float) -> None:
        self._enabled = interval_sec > 0
        if self._enabled:
            whole = int(interval_sec)
            if whole != interval_sec:
                raise ValueError(
                    "BIRD_XAI_WISH_RATE_LIMIT_SEC must be a whole number of seconds, "
                    f"got {interval_sec!r}"
                )
            self._limiter = MovingWindowRateLimiter(MemoryStorage())
            self._limit = parse(f"1/{whole} second")

    def __call__(self, request: Request) -> None:
        if not self._enabled:
            return
        if not self._limiter.hit(self._limit, client_ip(request)):
            raise HTTPException(status_code=429, detail="rate limit exceeded")
