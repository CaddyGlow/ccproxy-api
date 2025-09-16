from typing import Any

import pytest
import pytest_asyncio
from tests.helpers.assertions import (
    assert_codex_response_format,
    assert_openai_responses_format,
)
from tests.helpers.test_data import (
    STANDARD_CODEX_REQUEST,
    STANDARD_OPENAI_REQUEST,
)


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.codex
async def test_models_endpoint_available_when_enabled(
    codex_client,  # type: ignore[no-untyped-def]
) -> None:
    """GET /api/codex/v1/models returns a model list when enabled."""
    resp = await codex_client.get("/api/codex/v1/models")
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
    codex_client,  # type: ignore[no-untyped-def]
    mock_external_openai_codex_api,  # type: ignore[no-untyped-def]
) -> None:
    """POST /api/codex/responses proxies to Codex and returns Codex format."""
    resp = await codex_client.post("/api/codex/responses", json=STANDARD_CODEX_REQUEST)
    assert resp.status_code == 200
    data: dict[str, Any] = resp.json()
    assert_codex_response_format(data)


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.codex
async def test_openai_chat_completions_conversion(
    codex_client,  # type: ignore[no-untyped-def]
    mock_external_openai_codex_api,  # type: ignore[no-untyped-def]
) -> None:
    """OpenAI /v1/chat/completions converts through Codex and returns OpenAI format."""
    resp = await codex_client.post(
        "/api/codex/v1/chat/completions", json=STANDARD_OPENAI_REQUEST
    )
    assert resp.status_code == 200
    data: dict[str, Any] = resp.json()
    assert_openai_responses_format(data)


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.codex
async def test_openai_chat_completions_streaming(
    codex_client,  # type: ignore[no-untyped-def]
    mock_external_openai_codex_api_streaming,  # type: ignore[no-untyped-def]
) -> None:
    """Streaming OpenAI /v1/chat/completions returns SSE with valid chunks."""
    # Enable plugin
    request = {**STANDARD_OPENAI_REQUEST, "stream": True}
    resp = await codex_client.post("/api/codex/v1/chat/completions", json=request)

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


# Module-scoped client to avoid per-test startup cost
# Use module-level async loop for all tests here
pytestmark = pytest.mark.asyncio(loop_scope="module")


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def codex_client():  # type: ignore[no-untyped-def]
    # Build app and client once to avoid factory scope conflicts
    from httpx import ASGITransport, AsyncClient

    from ccproxy.api.app import create_app, initialize_plugins_startup
    from ccproxy.api.bootstrap import create_service_container
    from ccproxy.config.settings import Settings
    from ccproxy.core.logging import setup_logging

    setup_logging(json_logs=False, log_level_name="ERROR")
    settings = Settings(
        enable_plugins=True,
        plugins={"codex": {"enabled": True}},
        plugins_disable_local_discovery=False,  # Enable local plugin discovery
    )
    service_container = create_service_container(settings)
    app = create_app(service_container)
    await initialize_plugins_startup(app, settings)

    transport = ASGITransport(app=app)
    client = AsyncClient(transport=transport, base_url="http://test")
    try:
        yield client
    finally:
        await client.aclose()
