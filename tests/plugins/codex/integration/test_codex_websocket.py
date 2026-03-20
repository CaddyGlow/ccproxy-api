import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from starlette.testclient import WebSocketDenialResponse

from ccproxy.api.app import create_app, initialize_plugins_startup
from ccproxy.api.bootstrap import create_service_container
from ccproxy.config.core import ServerSettings
from ccproxy.config.security import SecuritySettings
from ccproxy.config.settings import Settings
from ccproxy.core.logging import setup_logging
from ccproxy.models.detection import DetectedHeaders, DetectedPrompts
from ccproxy.plugins.codex.models import CodexCacheData


@pytest.fixture
def codex_ws_client() -> Any:
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

    load_patch = patch(
        "ccproxy.plugins.oauth_codex.manager.CodexTokenManager.load_credentials",
        new=AsyncMock(return_value=credentials_stub),
    )
    profile_patch = patch(
        "ccproxy.plugins.oauth_codex.manager.CodexTokenManager.get_profile_quick",
        new=AsyncMock(return_value=profile_stub),
    )
    detection_patch = patch(
        "ccproxy.plugins.codex.detection_service.CodexDetectionService.initialize_detection",
        new=init_detection_stub,
    )

    with load_patch, profile_patch, detection_patch:
        asyncio.run(initialize_plugins_startup(app, settings))
        with TestClient(app) as client:
            yield client


@pytest.fixture
def codex_ws_bypass_client() -> Any:
    setup_logging(json_logs=False, log_level_name="ERROR")
    settings = Settings(
        enable_plugins=True,
        server=ServerSettings(bypass_mode=True),
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

    with detection_patch:
        asyncio.run(initialize_plugins_startup(app, settings))
        with TestClient(app) as client:
            yield client


@pytest.fixture
def codex_ws_auth_client() -> Any:
    setup_logging(json_logs=False, log_level_name="ERROR")
    settings = Settings(
        enable_plugins=True,
        security=SecuritySettings(auth_token=SecretStr("test-auth-token")),
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


@pytest.mark.integration
@pytest.mark.codex
def test_codex_websocket_responses_streaming(
    codex_ws_client: TestClient,
    mock_external_openai_codex_api_streaming: Any,
) -> None:
    request_payload = {
        "type": "response.create",
        "model": "gpt-5",
        "stream": True,
        "instructions": "Reply with exactly OK",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Reply with exactly OK"}],
            }
        ],
    }

    with codex_ws_client.websocket_connect(
        "/codex/v1/responses",
        headers={
            "authorization": "Bearer ignored-client-token",
            "chatgpt-account-id": "test-account-id",
            "openai-beta": "responses_websockets=2026-02-06",
            "originator": "Codex Desktop",
            "session_id": "test-session",
            "version": "0.114.0",
            "x-codex-beta-features": "multi_agent",
            "x-codex-turn-metadata": '{"turn_id":"","sandbox":"seatbelt"}',
        },
    ) as websocket:
        websocket.send_json(request_payload)

        events: list[dict[str, Any]] = []
        while True:
            try:
                events.append(websocket.receive_json())
                if events[-1].get("type") == "response.completed":
                    websocket.close()
                    break
            except Exception:
                break

    event_types = [event.get("type") for event in events]
    assert event_types == [
        "response.created",
        "response.output_text.delta",
        "response.output_text.delta",
        "response.output_text.done",
        "response.completed",
    ]
    assert events[-1]["response"]["output"][0]["content"][0]["text"] == "Hello Codex!"


@pytest.mark.integration
@pytest.mark.codex
def test_codex_websocket_returns_terminal_event_on_upstream_error(
    codex_ws_client: TestClient,
    mock_external_openai_codex_api_error: Any,
) -> None:
    request_payload = {
        "type": "response.create",
        "model": "gpt-5",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Reply with exactly OK"}],
            }
        ],
    }

    with codex_ws_client.websocket_connect("/codex/v1/responses") as websocket:
        websocket.send_json(request_payload)
        event = websocket.receive_json()
        websocket.close()

    assert event["type"] == "response.failed"
    assert event["response"]["status"] == "failed"
    assert event["response"]["error"]["type"] == "invalid_request_error"


@pytest.mark.integration
@pytest.mark.codex
def test_codex_websocket_short_circuits_empty_warmup_request(
    codex_ws_client: TestClient,
) -> None:
    request_payload = {
        "type": "response.create",
        "model": "gpt-5",
        "input": [],
    }

    with codex_ws_client.websocket_connect("/codex/v1/responses") as websocket:
        websocket.send_json(request_payload)
        event = websocket.receive_json()
        websocket.close()

    assert event["type"] == "response.completed"
    assert event["response"]["status"] == "completed"
    assert event["response"]["output"] == []


@pytest.mark.integration
@pytest.mark.codex
def test_codex_websocket_warmup_then_real_request_same_connection(
    codex_ws_client: TestClient,
    mock_external_openai_codex_api_streaming: Any,
) -> None:
    warmup_payload = {
        "type": "response.create",
        "model": "gpt-5",
        "input": [],
    }
    request_payload = {
        "type": "response.create",
        "model": "gpt-5",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Reply with exactly OK"}],
            }
        ],
    }

    with codex_ws_client.websocket_connect("/codex/v1/responses") as websocket:
        websocket.send_json(warmup_payload)
        warmup_event = websocket.receive_json()
        request_payload["previous_response_id"] = warmup_event["response"]["id"]

        websocket.send_json(request_payload)

        events: list[dict[str, Any]] = []
        while True:
            event = websocket.receive_json()
            events.append(event)
            if event.get("type") == "response.completed":
                websocket.close()
                break

    assert warmup_event["type"] == "response.completed"
    assert events[-1]["type"] == "response.completed"
    assert events[-1]["response"]["output"][0]["content"][0]["text"] == "Hello Codex!"


@pytest.mark.integration
@pytest.mark.codex
def test_codex_websocket_bypass_mode_streams_mock_events(
    codex_ws_bypass_client: TestClient,
) -> None:
    request_payload = {
        "type": "response.create",
        "model": "gpt-5",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Reply with exactly OK"}],
            }
        ],
    }

    with codex_ws_bypass_client.websocket_connect("/codex/v1/responses") as websocket:
        websocket.send_json(request_payload)

        events: list[dict[str, Any]] = []
        while True:
            event = websocket.receive_json()
            events.append(event)
            if event.get("type") == "response.completed":
                websocket.close()
                break

    assert events[0]["type"] == "response.created"
    assert events[-1]["type"] == "response.completed"
    assert events[-1]["response"]["status"] == "completed"


@pytest.mark.integration
@pytest.mark.codex
def test_codex_websocket_denies_unauthorized_handshake(
    codex_ws_auth_client: TestClient,
) -> None:
    with (
        pytest.raises(WebSocketDenialResponse) as exc_info,
        codex_ws_auth_client.websocket_connect("/codex/v1/responses"),
    ):
        pass

    assert exc_info.value.status_code == 401
    assert exc_info.value.text == "Authentication required"
