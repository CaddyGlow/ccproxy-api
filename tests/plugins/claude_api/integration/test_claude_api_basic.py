from typing import Any

import pytest
import pytest_asyncio
from tests.helpers.assertions import (
    assert_anthropic_response_format,
)
from tests.helpers.test_data import (
    STANDARD_ANTHROPIC_REQUEST,
)


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.claude_api
async def test_models_endpoint_available_when_enabled(
    claude_api_client,  # type: ignore[no-untyped-def]
) -> None:
    """GET /api/v1/models returns a model list when enabled."""
    resp = await claude_api_client.get("/api/v1/models")
    assert resp.status_code == 200
    data: dict[str, Any] = resp.json()
    assert data.get("object") == "list"
    models = data.get("data")
    assert isinstance(models, list)
    assert len(models) > 0
    assert {"id", "object", "created", "owned_by"}.issubset(models[0].keys())
    # Verify Claude models are present
    model_ids = {model["id"] for model in models}
    assert "claude-3-5-sonnet-20241022" in model_ids


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.claude_api
async def test_anthropic_messages_passthrough(
    claude_api_client,  # type: ignore[no-untyped-def]
    mock_external_anthropic_api,  # type: ignore[no-untyped-def]
) -> None:
    """POST /api/v1/messages proxies to Claude API and returns Anthropic format."""
    resp = await claude_api_client.post(
        "/api/v1/messages", json=STANDARD_ANTHROPIC_REQUEST
    )
    assert resp.status_code == 200
    data: dict[str, Any] = resp.json()
    assert_anthropic_response_format(data)


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.claude_api
async def test_openai_chat_completions_conversion(
    integration_client_factory,  # type: ignore[no-untyped-def]
) -> None:
    """OpenAI /v1/chat/completions converts through Claude API and returns OpenAI format."""
    # Skip this test until format adapter is properly configured
    pytest.skip("Format adapter anthropic->openai not configured in test environment")


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.claude_api
async def test_claude_response_api_endpoint(
    integration_client_factory,  # type: ignore[no-untyped-def]
) -> None:
    """POST /api/v1/responses handles Response API format."""
    # Skip this test until response API format handling is clarified
    pytest.skip("Response API format handling needs clarification")


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.claude_api
async def test_claude_response_api_with_session(
    integration_client_factory,  # type: ignore[no-untyped-def]
) -> None:
    """POST /api/{session_id}/v1/responses handles session-based requests."""
    # Skip this test until response API format handling is clarified
    pytest.skip("Response API format handling needs clarification")


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.claude_api
async def test_openai_chat_completions_streaming(
    integration_client_factory,  # type: ignore[no-untyped-def]
) -> None:
    """Streaming OpenAI /v1/chat/completions returns SSE with valid chunks."""
    # Skip this test until format adapter is properly configured
    pytest.skip("Format adapter anthropic->openai not configured in test environment")


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.claude_api
async def test_anthropic_messages_streaming(
    claude_api_client,  # type: ignore[no-untyped-def]
    mock_external_anthropic_api_streaming,  # type: ignore[no-untyped-def]
) -> None:
    """Streaming Anthropic /v1/messages returns SSE with valid chunks."""
    request = {**STANDARD_ANTHROPIC_REQUEST, "stream": True}
    resp = await claude_api_client.post("/api/v1/messages", json=request)

    # Validate SSE headers
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.headers.get("cache-control") == "no-cache"

    # Read entire body and validate streaming format
    body = (await resp.aread()).decode()
    chunks = [c for c in body.split("\n\n") if c.strip()]

    # Should have multiple event chunks and message_stop
    assert any(line.startswith("data: ") for line in chunks[0].splitlines())
    assert len(chunks) >= 3
    # Anthropic streams end with message_stop event
    assert any("message_stop" in chunk for chunk in chunks[-3:])


# Module-scoped client to avoid per-test startup cost
# Use module-level async loop for all tests here
pytestmark = pytest.mark.asyncio(loop_scope="module")


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def claude_api_client():  # type: ignore[no-untyped-def]
    # Build app and client once to avoid factory scope conflicts
    from httpx import ASGITransport, AsyncClient

    from ccproxy.api.app import create_app, initialize_plugins_startup
    from ccproxy.api.bootstrap import create_service_container
    from ccproxy.config.settings import Settings
    from ccproxy.core.logging import setup_logging

    setup_logging(json_logs=False, log_level_name="ERROR")
    settings = Settings(
        enable_plugins=True,
        plugins={"claude_api": {"enabled": True}},
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
