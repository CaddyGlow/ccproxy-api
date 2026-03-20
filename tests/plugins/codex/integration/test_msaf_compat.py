from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from pytest_httpx import HTTPXMock
from tests.helpers.assertions import assert_openai_responses_format

from ccproxy.api.app import create_app, initialize_plugins_startup, shutdown_plugins
from ccproxy.api.bootstrap import create_service_container
from ccproxy.config.settings import Settings
from ccproxy.core.logging import setup_logging
from ccproxy.models.detection import DetectedHeaders, DetectedPrompts
from ccproxy.plugins.codex.models import CodexCacheData


pytestmark = pytest.mark.asyncio(loop_scope="module")

DETECTED_CLI_INSTRUCTIONS = "Detected Codex CLI instructions"
MSAF_CHAT_COMPLETIONS_REQUEST: dict[str, Any] = {
    "model": "gpt-5.4",
    "messages": [
        {
            "role": "system",
            "content": "You are part of a requirements workshop for a login form.",
        },
        {"role": "user", "content": "Составьте требования для формы логина."},
    ],
    "reasoning_effort": "medium",
    "max_completion_tokens": 256,
    "temperature": 0.1,
}


def _build_detection_data() -> CodexCacheData:
    prompts = DetectedPrompts.from_body({"instructions": DETECTED_CLI_INSTRUCTIONS})
    return CodexCacheData(
        codex_version="fallback",
        headers=DetectedHeaders({}),
        prompts=prompts,
        body_json=prompts.raw,
        method="POST",
        url="https://chatgpt.com/backend-api/codex/responses",
        path="/backend-api/codex/responses",
        query_params={},
    )


@pytest_asyncio.fixture
async def codex_msaf_client() -> AsyncGenerator[AsyncClient, None]:
    setup_logging(json_logs=False, log_level_name="ERROR")

    settings = Settings(
        enable_plugins=True,
        plugins_disable_local_discovery=False,
        enabled_plugins=["codex", "oauth_codex"],
        plugins={
            "codex": {
                "enabled": True,
                "inject_detection_payload": False,
            },
            "oauth_codex": {"enabled": True},
            "duckdb_storage": {"enabled": False},
            "analytics": {"enabled": False},
            "metrics": {"enabled": False},
        },
        llm=Settings.LLMSettings(openai_thinking_xml=False),
    )
    service_container = create_service_container(settings)
    app = create_app(service_container)

    credentials_stub = SimpleNamespace(
        access_token="test-codex-access-token",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
    )
    profile_stub = SimpleNamespace(chatgpt_account_id="test-account-id")
    detection_data = _build_detection_data()

    async def init_detection_stub(self):  # type: ignore[no-untyped-def]
        self._cached_data = detection_data
        return detection_data

    async with AsyncExitStack() as stack:
        stack.enter_context(
            patch(
                "ccproxy.plugins.oauth_codex.manager.CodexTokenManager.load_credentials",
                new=AsyncMock(return_value=credentials_stub),
            )
        )
        stack.enter_context(
            patch(
                "ccproxy.plugins.oauth_codex.manager.CodexTokenManager.get_access_token",
                new=AsyncMock(return_value="test-codex-access-token"),
            )
        )
        stack.enter_context(
            patch(
                "ccproxy.plugins.oauth_codex.manager.CodexTokenManager.get_access_token_with_refresh",
                new=AsyncMock(return_value="test-codex-access-token"),
            )
        )
        stack.enter_context(
            patch(
                "ccproxy.plugins.oauth_codex.manager.CodexTokenManager.get_profile_quick",
                new=AsyncMock(return_value=profile_stub),
            )
        )
        stack.enter_context(
            patch(
                "ccproxy.plugins.codex.detection_service.CodexDetectionService.initialize_detection",
                new=init_detection_stub,
            )
        )

        await initialize_plugins_startup(app, settings)
        transport = ASGITransport(app=app)
        client = AsyncClient(transport=transport, base_url="http://test")
        try:
            yield client
        finally:
            await client.aclose()
            await shutdown_plugins(app)
            await service_container.close()


@pytest.mark.integration
@pytest.mark.codex
async def test_msaf_chat_completions_request_reaches_codex_without_cli_injection(
    codex_msaf_client: AsyncClient,
    mock_external_codex_api: HTTPXMock,
) -> None:
    response = await codex_msaf_client.post(
        "/codex/v1/chat/completions",
        json=MSAF_CHAT_COMPLETIONS_REQUEST,
    )

    assert response.status_code == 200, response.text
    data = response.json()
    assert_openai_responses_format(data)

    requests = mock_external_codex_api.get_requests()
    assert len(requests) == 1

    upstream_payload = json.loads(requests[0].read().decode())
    assert (
        upstream_payload["instructions"]
        == MSAF_CHAT_COMPLETIONS_REQUEST["messages"][0]["content"]
    )
    assert DETECTED_CLI_INSTRUCTIONS not in upstream_payload["instructions"]
    assert upstream_payload["reasoning"] == {"effort": "medium", "summary": "auto"}
    assert upstream_payload["stream"] is True
    assert upstream_payload["store"] is False
    assert "max_tokens" not in upstream_payload
    assert "max_output_tokens" not in upstream_payload
    assert "temperature" not in upstream_payload
    assert upstream_payload["input"][0]["type"] == "message"
    assert (
        upstream_payload["input"][0]["content"][0]["text"]
        == "Составьте требования для формы логина."
    )


@pytest.mark.integration
@pytest.mark.codex
async def test_msaf_chat_completions_hides_thinking_xml_when_disabled(
    codex_msaf_client: AsyncClient,
    mock_external_codex_api: HTTPXMock,
) -> None:
    request_payload = {
        **MSAF_CHAT_COMPLETIONS_REQUEST,
        "reasoning_effort": "high",
    }

    response = await codex_msaf_client.post(
        "/codex/v1/chat/completions",
        json=request_payload,
    )

    assert response.status_code == 200, response.text
    data = response.json()
    assert_openai_responses_format(data)
    assert "<thinking>" not in json.dumps(data, ensure_ascii=False)
