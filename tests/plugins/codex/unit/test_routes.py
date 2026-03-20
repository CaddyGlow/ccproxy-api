"""Unit tests for Codex websocket route helpers."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from ccproxy.config.settings import Settings
from ccproxy.core.request_context import RequestContext
from ccproxy.models.detection import DetectedHeaders, DetectedPrompts
from ccproxy.models.provider import ModelMappingRule
from ccproxy.plugins.codex.adapter import CodexAdapter
from ccproxy.plugins.codex.detection_service import CodexDetectionService
from ccproxy.plugins.codex.routes import (
    _get_websocket_settings,
    _restore_websocket_event_models,
    _sanitize_websocket_payload,
)


@pytest.fixture
def codex_adapter() -> CodexAdapter:
    detection_service = Mock(spec=CodexDetectionService)
    prompts = DetectedPrompts.from_body({"instructions": "Detected instructions"})
    detection_service.get_detected_prompts = Mock(return_value=prompts)
    detection_service.get_system_prompt = Mock(
        return_value=prompts.instructions_payload()
    )
    detection_service.get_detected_headers = Mock(return_value=DetectedHeaders({}))
    detection_service.get_ignored_headers = Mock(return_value=[])
    detection_service.get_redacted_headers = Mock(return_value=[])

    auth_manager = Mock()
    auth_manager.get_access_token = AsyncMock(return_value="test-token")
    auth_manager.get_access_token_with_refresh = AsyncMock(return_value="test-token")
    auth_manager.load_credentials = AsyncMock(return_value=Mock(access_token="token"))
    auth_manager.should_refresh = Mock(return_value=False)
    auth_manager.get_token_snapshot = AsyncMock(return_value=None)
    auth_manager.get_profile_quick = AsyncMock(return_value=None)

    config = Mock()
    config.base_url = "https://chatgpt.com/backend-codex"
    config.model_mappings = [
        ModelMappingRule(
            match="alias-model",
            target="gpt-5.3-codex",
            kind="exact",
        )
    ]

    return CodexAdapter(
        detection_service=detection_service,
        config=config,
        auth_manager=auth_manager,
        http_pool_manager=Mock(),
    )


@pytest.mark.asyncio
async def test_sanitize_websocket_payload_applies_model_mapping(
    codex_adapter: CodexAdapter,
) -> None:
    request_context = RequestContext(
        request_id="ws-test",
        start_time=0.0,
        logger=Mock(),
        metadata={},
        format_chain=["openai.responses"],
    )

    payload, headers = await _sanitize_websocket_payload(
        codex_adapter,
        {"model": "alias-model", "input": []},
        {"content-type": "application/json"},
        request_context,
    )

    assert payload["model"] == "gpt-5.3-codex"
    assert headers["authorization"] == "Bearer test-token"
    assert request_context.metadata["_model_alias_map"] == {
        "gpt-5.3-codex": "alias-model"
    }


def test_restore_websocket_event_models_uses_client_alias() -> None:
    request_context = RequestContext(
        request_id="ws-test",
        start_time=0.0,
        logger=Mock(),
        metadata={"_model_alias_map": {"gpt-5.3-codex": "alias-model"}},
    )
    event = {
        "type": "response.completed",
        "response": {
            "model": "gpt-5.3-codex",
            "output": [{"type": "message", "model": "gpt-5.3-codex"}],
        },
    }

    restored = _restore_websocket_event_models(event, request_context)

    assert restored["response"]["model"] == "alias-model"
    assert restored["response"]["output"][0]["model"] == "alias-model"


def test_get_websocket_settings_prefers_app_state_settings() -> None:
    settings = Settings()
    websocket = Mock()
    websocket.app.state = SimpleNamespace(settings=settings)

    resolved = _get_websocket_settings(websocket)

    assert resolved is settings


def test_get_websocket_settings_raises_when_container_cannot_provide_settings() -> None:
    container = Mock()
    container.get_service.side_effect = ValueError("missing settings")
    websocket = Mock()
    websocket.app.state = SimpleNamespace(service_container=container)

    with pytest.raises(RuntimeError, match="Settings service unavailable"):
        _get_websocket_settings(websocket)
