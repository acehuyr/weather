"""Disk cache for HTTP responses.

Weather APIs are rate-limited and your demo will hit the same city many
times. Caching keeps the app fast and keeps you inside free-tier limits.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import requests

from config.settings import settings
from src.core.errors import ProviderError
from src.core.logging import get_logger

log = get_logger("ingestion.cache")


def _key(url: str, params: dict[str, Any]) -> str:
    # sort_keys matters: {"a":1,"b":2} and {"b":2,"a":1} must hit the same file.
    blob = url + json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def cached_get(url: str, params: dict[str, Any], ttl: int | None = None) -> dict[str, Any]:
    """GET with a TTL'd disk cache. Raises ProviderError on failure."""
    ttl = settings.cache_ttl_seconds if ttl is None else ttl
    path = settings.cache_dir / f"{_key(url, params)}.json"

    if path.exists() and (time.time() - path.stat().st_mtime) < ttl:
        log.debug("cache hit %s", path.name)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log.warning("corrupt cache file %s - refetching", path.name)

    log.info("GET %s", url.rsplit("/", 1)[-1])
    try:
        resp = requests.get(url, params=params, timeout=settings.request_timeout)
        resp.raise_for_status()
        payload = resp.json()
    except requests.exceptions.Timeout as exc:
        raise ProviderError(f"Weather provider timed out after {settings.request_timeout}s") from exc
    except requests.exceptions.HTTPError as exc:
        raise ProviderError(f"Weather provider returned {exc.response.status_code}") from exc
    except requests.exceptions.RequestException as exc:
        raise ProviderError(f"Network error reaching weather provider: {exc}") from exc
    except ValueError as exc:
        raise ProviderError("Weather provider returned malformed JSON") from exc

    # Open-Meteo reports errors in a 200 body.
    if isinstance(payload, dict) and payload.get("error"):
        raise ProviderError(str(payload.get("reason", "unknown provider error")))

    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload
