"""Unit tests for the live Anthropic model catalogue."""

from unittest.mock import AsyncMock, Mock

import pytest
from pytest_httpx import HTTPXMock

from ccproxy.plugins.claude_api.live_models import (
    LiveModelCatalogue,
    fetch_live_models,
    merge_models,
)


MODELS_URL = "https://api.anthropic.com/v1/models"


class TestLiveModelCatalogue:
    """Test the live model catalogue."""

    async def test_fetch_converts_anthropic_entries(
        self, httpx_mock: HTTPXMock
    ) -> None:
        """Anthropic entries become OpenAI-compatible model cards."""
        httpx_mock.add_response(
            url=MODELS_URL,
            json={
                "data": [
                    {
                        "id": "claude-opus-5-5",
                        "created_at": "2026-05-01T12:00:00Z",
                        "display_name": "Claude Opus 5.5",
                        "line": "opus",
                    }
                ]
            },
        )

        catalogue = LiveModelCatalogue()
        models = await catalogue.fetch("test-token")

        assert len(models) == 1
        assert models[0]["id"] == "claude-opus-5-5"
        assert models[0]["object"] == "model"
        assert models[0]["owned_by"] == "anthropic"
        assert models[0]["created"] == 1777636800
        assert models[0]["display_name"] == "Claude Opus 5.5"
        assert models[0]["line"] == "opus"

        request = httpx_mock.get_requests()[0]
        assert request.headers["authorization"] == "Bearer test-token"
        assert request.headers["anthropic-version"] == "2023-06-01"

    async def test_fetch_uses_cache_within_ttl(self, httpx_mock: HTTPXMock) -> None:
        """A second fetch inside the TTL reuses the cached catalogue."""
        httpx_mock.add_response(
            url=MODELS_URL, json={"data": [{"id": "claude-opus-5-5"}]}
        )

        catalogue = LiveModelCatalogue()
        await catalogue.fetch("test-token")
        models = await catalogue.fetch("test-token")

        assert [model["id"] for model in models] == ["claude-opus-5-5"]
        assert len(httpx_mock.get_requests()) == 1

    async def test_fetch_keeps_last_models_on_failure(
        self, httpx_mock: HTTPXMock
    ) -> None:
        """An upstream failure keeps the last successful catalogue."""
        httpx_mock.add_response(
            url=MODELS_URL, json={"data": [{"id": "claude-opus-5-5"}]}
        )
        catalogue = LiveModelCatalogue(ttl_seconds=0.0)
        await catalogue.fetch("test-token")

        httpx_mock.reset()
        httpx_mock.add_response(url=MODELS_URL, status_code=500)
        models = await catalogue.fetch("test-token")

        assert [model["id"] for model in models] == ["claude-opus-5-5"]

    async def test_fetch_without_token_returns_nothing(self) -> None:
        """Without a usable token the live catalogue stays empty."""
        adapter = Mock()
        adapter.get_access_token = AsyncMock(side_effect=RuntimeError("no credentials"))

        assert await fetch_live_models(adapter) == []


class TestMergeModels:
    """Test merging live and static model cards."""

    def test_live_models_win(self) -> None:
        """Live entries replace static ones with the same id."""
        live = [{"id": "claude-sonnet-4-6", "created": 1}]
        static = [
            {"id": "claude-sonnet-4-6", "created": 0},
            {"id": "claude-3-5-sonnet-20241022", "created": 0},
        ]

        merged = merge_models(live, static)

        assert [model["id"] for model in merged] == [
            "claude-sonnet-4-6",
            "claude-3-5-sonnet-20241022",
        ]
        assert merged[0]["created"] == 1
