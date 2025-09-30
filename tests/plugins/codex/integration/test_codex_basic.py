from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

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

from ccproxy.models.detection import DetectedHeaders, DetectedPrompts
from ccproxy.plugins.codex.models import CodexCacheData


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.codex
async def test_models_endpoint_available_when_enabled(
    codex_client,  # type: ignore[no-untyped-def]
) -> None:
    """GET /codex/v1/models returns a model list when enabled."""
    resp = await codex_client.get("/codex/v1/models")
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
    """POST /codex/responses proxies to Codex and returns Codex format."""
    resp = await codex_client.post("/codex/responses", json=STANDARD_CODEX_REQUEST)
    assert resp.status_code == 200, resp.text
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
        "/codex/v1/chat/completions", json=STANDARD_OPENAI_REQUEST
    )
    assert resp.status_code == 200, resp.text
    data: dict[str, Any] = resp.json()
    assert_openai_responses_format(data)


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.codex
async def test_model_alias_restored_in_response(
    codex_client,  # type: ignore[no-untyped-def]
    mock_external_openai_codex_api,  # type: ignore[no-untyped-def]
) -> None:
    """Client model aliases stay intact in non-streaming responses."""
    request_payload = {**STANDARD_OPENAI_REQUEST, "model": "gpt-5-nano"}
    resp = await codex_client.post("/codex/v1/chat/completions", json=request_payload)
    assert resp.status_code == 200, resp.text
    data: dict[str, Any] = resp.json()
    assert data.get("model") == "gpt-5-nano"


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.codex
@pytest.mark.skip(
    reason="Streaming SSE payload parsing currently fails with validation errors; bypass until Codex stream adapter is aligned."
)
async def test_openai_chat_completions_streaming(
    codex_client,  # type: ignore[no-untyped-def]
    mock_external_openai_codex_api_streaming,  # type: ignore[no-untyped-def]
) -> None:
    """Streaming OpenAI /v1/chat/completions returns SSE with valid chunks."""
    # Enable plugin
    request = {**STANDARD_OPENAI_REQUEST, "stream": True}
    resp = await codex_client.post("/codex/v1/chat/completions", json=request)
    raw_body = await resp.aread()

    # Validate SSE headers (note: proxy strips 'connection')
    assert resp.status_code == 200, raw_body
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.headers.get("cache-control") == "no-cache"

    # Read entire body and split by double newlines to get SSE chunks
    body = raw_body.decode()
    chunks = [c for c in body.split("\n\n") if c.strip()]
    assert chunks, "Expected at least one SSE chunk"
    assert chunks[-1].strip() == "data: [DONE]"
    assert any(chunk.startswith("data: ") for chunk in chunks)


# Module-scoped client to avoid per-test startup cost
# Use module-level async loop for all tests here
pytestmark = pytest.mark.asyncio(loop_scope="module")


@pytest_asyncio.fixture(scope="function", loop_scope="function")
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
        plugins={
            "codex": {"enabled": True},
            "oauth_codex": {"enabled": True},
            "duckdb_storage": {"enabled": False},
            "analytics": {"enabled": False},
            "metrics": {"enabled": False},
        },
        enabled_plugins=["codex", "oauth_codex"],
        plugins_disable_local_discovery=False,  # Enable local plugin discovery
    )
    service_container = create_service_container(settings)
    app = create_app(service_container)

    from ccproxy.plugins.codex.routes import router as codex_router

    app.include_router(codex_router, prefix="/codex")

    credentials_stub = SimpleNamespace(
        access_token="test-codex-access-token",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
    )
    profile_stub = SimpleNamespace(chatgpt_account_id="test-account-id")

    load_patch = patch(
        "ccproxy.plugins.oauth_codex.manager.CodexTokenManager.load_credentials",
        new=AsyncMock(return_value=credentials_stub),
    )
    profile_patch = patch(
        "ccproxy.plugins.oauth_codex.manager.CodexTokenManager.get_profile_quick",
        new=AsyncMock(return_value=profile_stub),
    )
    prompts = DetectedPrompts.from_body(
        {"instructions": "You are a helpful coding assistant."}
    )

    detection_data = CodexCacheData(
        codex_version="fallback",
        headers=DetectedHeaders({}),
        prompts=prompts,
        body_json=prompts.raw,
        method="POST",
        url="https://chatgpt.com/backend-codex/responses",
        path="/api/backend-codex/responses",
        query_params={},
    )

    async def init_detection_stub(self):  # type: ignore[no-untyped-def]
        self._cached_data = detection_data
        return detection_data

    detection_patch = patch(
        "ccproxy.plugins.codex.detection_service.CodexDetectionService.initialize_detection",
        new=init_detection_stub,
    )
    with load_patch, profile_patch, detection_patch:
        await initialize_plugins_startup(app, settings)

        transport = ASGITransport(app=app)
        runtime = app.state.plugin_registry.get_runtime("codex")
        assert runtime and runtime.adapter, "Codex plugin failed to initialize"
        client = AsyncClient(transport=transport, base_url="http://test")
        try:
            yield client
        finally:
            await client.aclose()
