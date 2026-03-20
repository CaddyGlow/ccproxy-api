"""End-to-end integration tests for CCProxy WebSocket endpoints.

Follows the same parameterized pattern as test_endpoint_e2e.py, covering
WebSocket transport for Codex responses (v1 and legacy paths).

Tests validate:
- WebSocket configuration structure
- Request builder correctness
- Event sequence validation helpers
- Warmup, streaming, error, and multi-message flows
- Live server WebSocket flows (when CCPROXY_BASE_URL is set)
"""

import asyncio
import json
import os
from collections.abc import Generator
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from ccproxy.api.app import create_app, initialize_plugins_startup
from ccproxy.api.bootstrap import create_service_container
from ccproxy.config.settings import Settings
from ccproxy.core.logging import setup_logging
from ccproxy.models.detection import DetectedHeaders, DetectedPrompts
from ccproxy.plugins.codex.models import CodexCacheData
from tests.helpers.e2e_validation import (
    validate_ws_codex_error_response,
    validate_ws_codex_event_sequence,
    validate_ws_codex_streaming_content,
    validate_ws_codex_warmup_response,
)
from tests.helpers.test_data import (
    CODEX_WS_TERMINAL_EVENT_TYPES,
    WS_ENDPOINT_CONFIGURATIONS,
    create_ws_codex_request,
    create_ws_codex_warmup_request,
)


pytestmark = [pytest.mark.integration, pytest.mark.e2e]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_detection_data() -> CodexCacheData:
    prompts = DetectedPrompts.from_body(
        {"instructions": "You are a helpful coding assistant."}
    )
    return CodexCacheData(
        codex_version="fallback",
        headers=DetectedHeaders({}),
        prompts=prompts,
        body_json=prompts.raw,
        method="POST",
        url="https://chatgpt.com/backend-codex/responses",
        path="/api/backend-codex/responses",
        query_params={},
    )


@pytest.fixture
def codex_ws_app() -> Generator[TestClient, None, None]:
    """Create a fully-initialised Codex app wrapped in a sync TestClient.

    Patches OAuth credentials and detection so no real providers are needed.
    """
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
        plugins_disable_local_discovery=False,
    )
    service_container = create_service_container(settings)
    app = create_app(service_container)

    credentials_stub = SimpleNamespace(
        access_token="test-codex-access-token",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
    )
    profile_stub = SimpleNamespace(chatgpt_account_id="test-account-id")
    detection_data = _build_detection_data()

    async def init_detection_stub(self: Any) -> CodexCacheData:
        self._cached_data = detection_data
        return detection_data

    with (
        patch(
            "ccproxy.plugins.oauth_codex.manager.CodexTokenManager.load_credentials",
            new=AsyncMock(return_value=credentials_stub),
        ),
        patch(
            "ccproxy.plugins.oauth_codex.manager.CodexTokenManager.get_profile_quick",
            new=AsyncMock(return_value=profile_stub),
        ),
        patch(
            "ccproxy.plugins.codex.detection_service.CodexDetectionService.initialize_detection",
            new=init_detection_stub,
        ),
    ):
        asyncio.run(initialize_plugins_startup(app, settings))
        with TestClient(app) as client:
            yield client


# ---------------------------------------------------------------------------
# Configuration structure tests (no app needed, mirrors test_endpoint_e2e.py)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ws_endpoint_configurations_structure() -> None:
    """Verify all WebSocket endpoint configs have required fields."""
    assert len(WS_ENDPOINT_CONFIGURATIONS) > 0

    for config in WS_ENDPOINT_CONFIGURATIONS:
        required_fields = ["name", "endpoint", "model", "description"]
        assert all(field in config for field in required_fields), (
            f"Config {config.get('name')} missing fields"
        )

        endpoint = config["endpoint"]
        assert isinstance(endpoint, str)
        assert endpoint.startswith("/")
        assert isinstance(config["model"], str)
        assert len(config["model"]) > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("config", WS_ENDPOINT_CONFIGURATIONS)
async def test_ws_request_creation_for_each_endpoint(
    config: dict[str, Any],
) -> None:
    """Verify request builders produce valid payloads for each config."""
    model = config["model"]

    request_data = create_ws_codex_request(
        content="Test WebSocket message", model=model
    )
    assert request_data["type"] == "response.create"
    assert request_data["model"] == model
    assert isinstance(request_data["input"], list)
    assert len(request_data["input"]) > 0

    warmup = create_ws_codex_warmup_request(model=model)
    assert warmup["type"] == "response.create"
    assert warmup["input"] == []


@pytest.mark.asyncio
async def test_ws_validation_helpers_work() -> None:
    """Verify validation helpers detect good and bad event sequences."""
    good_events: list[dict[str, Any]] = [
        {"type": "response.created", "response": {"id": "r1", "object": "response"}},
        {"type": "response.output_text.delta", "delta": "Hello"},
        {
            "type": "response.completed",
            "response": {
                "id": "r1",
                "object": "response",
                "status": "completed",
                "output": [],
            },
        },
    ]
    is_valid, errors = validate_ws_codex_event_sequence(good_events)
    assert is_valid, errors

    text, text_errors = validate_ws_codex_streaming_content(good_events)
    assert text == "Hello"
    assert not text_errors

    # Empty events should fail
    is_valid, errors = validate_ws_codex_event_sequence([])
    assert not is_valid

    # Missing terminal event should fail
    is_valid, errors = validate_ws_codex_event_sequence([{"type": "response.created"}])
    assert not is_valid

    # Warmup validation
    warmup_event = {
        "type": "response.completed",
        "response": {
            "id": "w1",
            "object": "response",
            "status": "completed",
            "output": [],
        },
    }
    is_valid, errors = validate_ws_codex_warmup_response(warmup_event)
    assert is_valid, errors

    # Error validation
    error_event = {
        "type": "response.failed",
        "response": {
            "id": "e1",
            "object": "response",
            "status": "failed",
            "error": {"type": "invalid_request_error", "message": "bad"},
        },
    }
    is_valid, errors = validate_ws_codex_error_response(error_event)
    assert is_valid, errors


# ---------------------------------------------------------------------------
# Live WebSocket tests (require codex_ws_app + external API mocks)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config", WS_ENDPOINT_CONFIGURATIONS, ids=lambda c: c["name"])
def test_ws_warmup_request(
    codex_ws_app: TestClient,
    config: dict[str, Any],
) -> None:
    """Empty-input warmup should return a completed terminal event immediately."""
    warmup = create_ws_codex_warmup_request(model=config["model"])

    with codex_ws_app.websocket_connect(config["endpoint"]) as ws:
        ws.send_json(warmup)
        event = ws.receive_json()
        ws.close()

    is_valid, errors = validate_ws_codex_warmup_response(event)
    assert is_valid, errors


@pytest.mark.parametrize("config", WS_ENDPOINT_CONFIGURATIONS, ids=lambda c: c["name"])
def test_ws_streaming_response(
    codex_ws_app: TestClient,
    mock_external_openai_codex_api_streaming: Any,
    config: dict[str, Any],
) -> None:
    """A real request should stream events ending with a terminal event."""
    request = create_ws_codex_request(
        content="Reply with exactly OK", model=config["model"]
    )

    events: list[dict[str, Any]] = []
    with codex_ws_app.websocket_connect(config["endpoint"]) as ws:
        ws.send_json(request)
        while True:
            event = ws.receive_json()
            events.append(event)
            if event.get("type") in CODEX_WS_TERMINAL_EVENT_TYPES:
                ws.close()
                break

    is_valid, errors = validate_ws_codex_event_sequence(events)
    assert is_valid, errors

    text, text_errors = validate_ws_codex_streaming_content(events)
    assert not text_errors, text_errors
    assert len(text) > 0, "Expected non-empty streamed text"


@pytest.mark.parametrize("config", WS_ENDPOINT_CONFIGURATIONS, ids=lambda c: c["name"])
def test_ws_upstream_error(
    codex_ws_app: TestClient,
    mock_external_openai_codex_api_error: Any,
    config: dict[str, Any],
) -> None:
    """Upstream errors should produce a failed terminal event."""
    request = create_ws_codex_request(
        content="Reply with exactly OK", model=config["model"]
    )

    with codex_ws_app.websocket_connect(config["endpoint"]) as ws:
        ws.send_json(request)
        event = ws.receive_json()
        ws.close()

    is_valid, errors = validate_ws_codex_error_response(event)
    assert is_valid, errors


@pytest.mark.parametrize("config", WS_ENDPOINT_CONFIGURATIONS, ids=lambda c: c["name"])
def test_ws_warmup_then_real_request(
    codex_ws_app: TestClient,
    mock_external_openai_codex_api_streaming: Any,
    config: dict[str, Any],
) -> None:
    """Warmup followed by real request on same connection should both succeed."""
    warmup = create_ws_codex_warmup_request(model=config["model"])
    request = create_ws_codex_request(
        content="Reply with exactly OK", model=config["model"]
    )

    with codex_ws_app.websocket_connect(config["endpoint"]) as ws:
        # Warmup
        ws.send_json(warmup)
        warmup_event = ws.receive_json()

        # Strip synthetic previous_response_id
        warmup_id = warmup_event.get("response", {}).get("id")
        request["previous_response_id"] = warmup_id

        # Real request
        ws.send_json(request)
        events: list[dict[str, Any]] = []
        while True:
            event = ws.receive_json()
            events.append(event)
            if event.get("type") in CODEX_WS_TERMINAL_EVENT_TYPES:
                ws.close()
                break

    # Validate warmup
    is_valid, errors = validate_ws_codex_warmup_response(warmup_event)
    assert is_valid, errors

    # Validate streaming
    is_valid, errors = validate_ws_codex_event_sequence(events)
    assert is_valid, errors

    text, text_errors = validate_ws_codex_streaming_content(events)
    assert not text_errors, text_errors
    assert len(text) > 0


# ---------------------------------------------------------------------------
# Live server tests (require `make dev` + real credentials)
#
# Run with: CCPROXY_BASE_URL=http://127.0.0.1:8000 pytest -m real_api -k websocket
# ---------------------------------------------------------------------------

_LIVE_BASE_URL = os.environ.get("CCPROXY_BASE_URL", "").rstrip("/")
_skip_no_live = pytest.mark.skipif(
    not _LIVE_BASE_URL,
    reason="CCPROXY_BASE_URL not set; skipping live WebSocket tests",
)


def _ws_url(http_base: str, path: str) -> str:
    """Convert http(s) base URL + path to a ws(s) URL."""
    return http_base.replace("https://", "wss://").replace("http://", "ws://") + path


@_skip_no_live
@pytest.mark.real_api
@pytest.mark.parametrize("config", WS_ENDPOINT_CONFIGURATIONS, ids=lambda c: c["name"])
@pytest.mark.asyncio
async def test_live_ws_warmup(config: dict[str, Any]) -> None:
    """Send a warmup to the live server and validate the response."""
    import websockets

    warmup = create_ws_codex_warmup_request(model=config["model"])
    url = _ws_url(_LIVE_BASE_URL, config["endpoint"])

    async with websockets.connect(url) as ws:
        await ws.send(json.dumps(warmup))
        raw = await asyncio.wait_for(ws.recv(), timeout=10)
        event = json.loads(raw)

    is_valid, errors = validate_ws_codex_warmup_response(event)
    assert is_valid, errors


@_skip_no_live
@pytest.mark.real_api
@pytest.mark.parametrize("config", WS_ENDPOINT_CONFIGURATIONS, ids=lambda c: c["name"])
@pytest.mark.asyncio
async def test_live_ws_streaming(config: dict[str, Any]) -> None:
    """Send a real request to the live server and collect streaming events."""
    import websockets

    request = create_ws_codex_request(
        content="Reply with exactly OK", model=config["model"]
    )
    url = _ws_url(_LIVE_BASE_URL, config["endpoint"])

    events: list[dict[str, Any]] = []
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps(request))
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=60)
            event = json.loads(raw)
            events.append(event)
            if event.get("type") in CODEX_WS_TERMINAL_EVENT_TYPES:
                break

    is_valid, errors = validate_ws_codex_event_sequence(events)
    assert is_valid, errors

    text, text_errors = validate_ws_codex_streaming_content(events)
    assert not text_errors, text_errors
    assert len(text) > 0, "Expected non-empty response from live server"


@_skip_no_live
@pytest.mark.real_api
@pytest.mark.parametrize("config", WS_ENDPOINT_CONFIGURATIONS, ids=lambda c: c["name"])
@pytest.mark.asyncio
async def test_live_ws_warmup_then_request(config: dict[str, Any]) -> None:
    """Warmup followed by real request on a single live WebSocket connection."""
    import websockets

    warmup = create_ws_codex_warmup_request(model=config["model"])
    request = create_ws_codex_request(
        content="Reply with exactly OK", model=config["model"]
    )
    url = _ws_url(_LIVE_BASE_URL, config["endpoint"])

    async with websockets.connect(url) as ws:
        # Warmup
        await ws.send(json.dumps(warmup))
        raw = await asyncio.wait_for(ws.recv(), timeout=10)
        warmup_event = json.loads(raw)

        is_valid, errors = validate_ws_codex_warmup_response(warmup_event)
        assert is_valid, errors

        # Attach previous_response_id from warmup
        warmup_id = warmup_event.get("response", {}).get("id")
        request["previous_response_id"] = warmup_id

        # Real request
        await ws.send(json.dumps(request))
        events: list[dict[str, Any]] = []
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=60)
            event = json.loads(raw)
            events.append(event)
            if event.get("type") in CODEX_WS_TERMINAL_EVENT_TYPES:
                break

    is_valid, errors = validate_ws_codex_event_sequence(events)
    assert is_valid, errors

    text, text_errors = validate_ws_codex_streaming_content(events)
    assert not text_errors, text_errors
    assert len(text) > 0
