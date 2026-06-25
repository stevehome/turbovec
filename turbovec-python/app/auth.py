"""Clerk JWT authentication. Disabled when CLERK_SECRET_KEY is unset (local dev)."""
from __future__ import annotations

import base64
import os
import time

import httpx
import jwt
from fastapi import HTTPException, Request

ENABLED = bool(os.environ.get("CLERK_SECRET_KEY"))

_jwks_cache: dict = {}
_jwks_fetched_at: float = 0.0
_JWKS_TTL = 3600.0


def frontend_api() -> str:
    pk = os.environ.get("CLERK_PUBLISHABLE_KEY", "")
    b64 = pk.split("_", 2)[2].rstrip("$")
    b64 += "=" * (-len(b64) % 4)
    return base64.b64decode(b64).decode().rstrip("$")


def script_url() -> str:
    host = frontend_api()
    return f"https://{host}/npm/@clerk/clerk-js@5/dist/clerk.browser.js" if host else ""


def _public_keys() -> dict:
    global _jwks_cache, _jwks_fetched_at
    if time.monotonic() - _jwks_fetched_at < _JWKS_TTL and _jwks_cache:
        return _jwks_cache
    url = f"https://{frontend_api()}/.well-known/jwks.json"
    keys = {
        k["kid"]: jwt.algorithms.RSAAlgorithm.from_jwk(k)
        for k in httpx.get(url, timeout=5).raise_for_status().json()["keys"]
    }
    _jwks_cache, _jwks_fetched_at = keys, time.monotonic()
    return keys


def _verify(token: str) -> dict:
    kid = jwt.get_unverified_header(token).get("kid", "")
    key = _public_keys().get(kid)
    if not key:
        raise ValueError("Unknown signing key")
    return jwt.decode(token, key, algorithms=["RS256"])


async def require_auth(request: Request) -> dict:
    if not ENABLED:
        return {}
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(401, "Not authenticated")
    try:
        return _verify(auth.removeprefix("Bearer "))
    except Exception:
        raise HTTPException(401, "Invalid or expired token")
