"""Live Anthropic model catalogue for the Claude API plugin."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

import httpx

from ccproxy.core.logging import get_plugin_logger


logger = get_plugin_logger()

MODELS_URL = "https://api.anthropic.com/v1/models"
CACHE_TTL_SECONDS = 3600.0
REQUEST_TIMEOUT_SECONDS = 15.0


def _parse_created(created_at: Any) -> int:
    """Convert an Anthropic RFC3339 timestamp into a Unix timestamp."""
    if not isinstance(created_at, str) or not created_at:
        return 0
    try:
        parsed = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError:
        return 0
    return int(parsed.timestamp())


class LiveModelCatalogue:
    """Fetch Anthropic's current model list and cache it for a short while."""

    def __init__(self, ttl_seconds: float = CACHE_TTL_SECONDS) -> None:
        self._ttl = ttl_seconds
        self._models: list[dict[str, Any]] = []
        self._fetched_at = 0.0

    def _is_fresh(self) -> bool:
        return bool(self._models) and (time.monotonic() - self._fetched_at) < self._ttl

    async def fetch(self, token: str) -> list[dict[str, Any]]:
        """Return Anthropic's live models, keeping the last good list on failure."""
        if self._is_fresh():
            return self._models

        headers = {
            "authorization": f"Bearer {token}",
            "anthropic-version": "2023-06-01",
        }
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
                response = await client.get(MODELS_URL, headers=headers)
                response.raise_for_status()
                payload = response.json()
        except Exception as exc:
            logger.warning("live_model_catalogue_fetch_failed", error=str(exc))
            return self._models

        entries = payload.get("data") or []
        models = [self._to_model(entry) for entry in entries if entry.get("id")]
        if not models:
            return self._models

        self._models = models
        self._fetched_at = time.monotonic()
        return self._models

    @staticmethod
    def _to_model(entry: dict[str, Any]) -> dict[str, Any]:
        """Convert an Anthropic model entry into an OpenAI-compatible model card."""
        model: dict[str, Any] = {
            "id": entry["id"],
            "created": _parse_created(entry.get("created_at")),
            "object": "model",
            "owned_by": "anthropic",
        }
        for key in ("display_name", "line"):
            value = entry.get(key)
            if value:
                model[key] = value
        return model


def merge_models(
    live_models: list[dict[str, Any]], static_models: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Merge live models with the static ones, live entries win."""
    known = {model["id"] for model in live_models}
    return live_models + [m for m in static_models if m.get("id") not in known]


catalogue = LiveModelCatalogue()


async def fetch_live_models(adapter: Any) -> list[dict[str, Any]]:
    """Fetch the live catalogue with the adapter's OAuth token."""
    getter = getattr(adapter, "get_access_token", None)
    if getter is None:
        return []

    try:
        token = await getter()
    except Exception as exc:
        logger.warning("live_model_token_unavailable", error=str(exc))
        return []

    return await catalogue.fetch(token)
