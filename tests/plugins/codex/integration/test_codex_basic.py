from typing import Any

import pytest
from tests.helpers.assertions import (
    assert_codex_response_format,
    assert_openai_response_format,
)
from tests.helpers.test_data import (
    STANDARD_CODEX_REQUEST,
    STANDARD_OPENAI_REQUEST,
)


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.codex
async def test_models_endpoint_available_when_enabled(
    integration_client_factory,  # type: ignore[no-untyped-def]
) -> None:
    """GET /api/codex/v1/models returns a model list when enabled."""
    client = await integration_client_factory({"codex": {"enabled": True}})
    async with client:
        resp = await client.get("/api/codex/v1/models")
        assert resp.status_code == 200
        data: dict[str, Any] = resp.json()
        assert data.get("object") == "list"
        models = data.get("data")
        assert isinstance(models, list)
        assert len(models) > 0
        assert {"id", "object", "created", "owned_by"}.issubset(models[0].keys())


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.codex
async def test_codex_responses_passthrough(
    integration_client_factory,  # type: ignore[no-untyped-def]
    mock_external_openai_codex_api,  # type: ignore[no-untyped-def]
) -> None:
    """POST /api/codex/responses proxies to Codex and returns Codex format."""
    client = await integration_client_factory({"codex": {"enabled": True}})
    async with client:
        resp = await client.post("/api/codex/responses", json=STANDARD_CODEX_REQUEST)
        assert resp.status_code == 200
        data: dict[str, Any] = resp.json()
        assert_codex_response_format(data)


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.codex
async def test_openai_chat_completions_conversion(
    integration_client_factory,  # type: ignore[no-untyped-def]
    mock_external_openai_codex_api,  # type: ignore[no-untyped-def]
) -> None:
    """OpenAI /v1/chat/completions converts through Codex and returns OpenAI format."""
    client = await integration_client_factory({"codex": {"enabled": True}})
    async with client:
        resp = await client.post(
            "/api/codex/v1/chat/completions", json=STANDARD_OPENAI_REQUEST
        )
        assert resp.status_code == 200
        data: dict[str, Any] = resp.json()
        assert_openai_response_format(data)


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.codex
async def test_openai_chat_completions_streaming(
    integration_client_factory,  # type: ignore[no-untyped-def]
    mock_external_openai_codex_api_streaming,  # type: ignore[no-untyped-def]
) -> None:
    """Streaming OpenAI /v1/chat/completions returns SSE with valid chunks."""
    # Enable plugin
    client = await integration_client_factory({"codex": {"enabled": True}})
    async with client:
        request = {**STANDARD_OPENAI_REQUEST, "stream": True}
        resp = await client.post("/api/codex/v1/chat/completions", json=request)

        # Validate SSE headers (note: proxy strips 'connection')
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert resp.headers.get("cache-control") == "no-cache"

        # Read entire body and split by double newlines to get SSE chunks
        body = (await resp.aread()).decode()
        chunks = [c for c in body.split("\n\n") if c.strip()]
        # Should have multiple data: chunks and a final [DONE]
        assert any(line.startswith("data: ") for line in chunks[0].splitlines())
        # Verify the stream yields at least 3 codex chunks then [DONE]
        assert len(chunks) >= 3
        assert chunks[-1].strip() == "data: [DONE]"
