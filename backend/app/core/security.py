import time
from collections import defaultdict, deque
from typing import Deque, Dict, Optional

from fastapi import Depends, HTTPException, Request, status

from app.core.config import API_KEY


_RATE_LIMIT_WINDOW_SECONDS = 60
_RATE_LIMIT_MAX_REQUESTS = 60


_request_timestamps: Dict[str, Deque[float]] = defaultdict(deque)


async def enforce_api_key(request: Request) -> None:
    """Simple API-key check via `X-API-Key` header.

    If `API_KEY` is not configured, this is a no-op.
    """

    if not API_KEY:
        return

    provided: Optional[str] = request.headers.get("X-API-Key")
    if not provided or provided != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": {"type": "AuthError", "message": "Invalid or missing API key"}},
        )


async def rate_limiter(request: Request) -> None:
    """Very simple in-memory per-IP rate limiter.

    Not suitable for multi-process or distributed deployments, but
    sufficient for small projects and demos.
    """

    identifier = request.client.host if request.client else "anonymous"
    now = time.time()

    timestamps = _request_timestamps[identifier]

    # Drop timestamps outside the window
    while timestamps and now - timestamps[0] > _RATE_LIMIT_WINDOW_SECONDS:
        timestamps.popleft()

    if len(timestamps) >= _RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={"error": {"type": "RateLimitError", "message": "Too many requests"}},
        )

    timestamps.append(now)
