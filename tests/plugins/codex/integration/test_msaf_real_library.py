"""Tests for MSAF-style sequential agent workflows through the Codex proxy.

Validates that multi-step agent patterns (analyst -> editor) work correctly
without requiring the agent_framework library, using plain httpx calls to
simulate the same request flow.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import pytest_asyncio
from pytest_httpx import HTTPXMock
from tests.helpers.assertions import assert_openai_responses_format

from ccproxy.api.app import create_app, initialize_plugins_startup, shutdown_plugins
from ccproxy.api.bootstrap import create_service_container
from ccproxy.config.settings import Settings
from ccproxy.core.logging import setup_logging
from ccproxy.models.detection import DetectedHeaders, DetectedPrompts
from ccproxy.plugins.codex.models import CodexCacheData


pytestmark = [
    pytest.mark.asyncio(loop_scope="module"),
    pytest.mark.integration,
    pytest.mark.codex,
]

DETECTED_CLI_INSTRUCTIONS = "Detected Codex CLI instructions"
COMMON_INSTRUCTIONS = (
    "You are part of a requirements workshop for a login form. "
    "Reply in the same language as the user request. "
    "Be concise and practical."
)


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


def _build_codex_response(
    *,
    response_id: str,
    message_id: str,
    text: str,
    reasoning_text: str,
) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "response",
        "created_at": 1773389433,
        "status": "completed",
        "model": "gpt-5-2025-08-07",
        "output": [
            {
                "type": "reasoning",
                "id": f"rs_{response_id}",
                "status": "completed",
                "summary": [{"type": "summary_text", "text": reasoning_text}],
            },
            {
                "type": "message",
                "id": message_id,
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text}],
            },
        ],
        "parallel_tool_calls": False,
        "usage": {
            "input_tokens": 64,
            "output_tokens": 32,
            "total_tokens": 96,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 12},
        },
    }


def _extract_message_text(data: dict[str, Any]) -> str:
    """Extract assistant message text from an OpenAI chat completions response."""
    choices = data.get("choices", [])
    if choices:
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            return content
    return ""


@pytest_asyncio.fixture
async def msaf_codex_client(
    httpx_mock: HTTPXMock,
) -> AsyncGenerator[tuple[httpx.AsyncClient, list[dict[str, Any]]], None]:
    upstream_payloads: list[dict[str, Any]] = []
    response_bodies = [
        _build_codex_response(
            response_id="resp_analyst",
            message_id="msg_analyst",
            text="- Email\n- Password\n- Remember me\n- Inline errors\n- Redirect after success",
            reasoning_text="Hidden analyst reasoning",
        ),
        _build_codex_response(
            response_id="resp_editor",
            message_id="msg_editor",
            text=(
                "## Goal\n"
                "Определить требования к форме логина.\n\n"
                "## Functional Requirements\n"
                "- Поля email и пароль.\n"
                "- Кнопка входа и remember me.\n\n"
                "## Validation Rules\n"
                "- Оба поля обязательны.\n"
                "- Email валидируется по формату.\n\n"
                "## Acceptance Criteria\n"
                "- Успешный вход ведет к редиректу."
            ),
            reasoning_text="Hidden editor reasoning",
        ),
    ]

    def upstream_callback(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode() or "{}")
        upstream_payloads.append(payload)
        index = min(len(upstream_payloads), len(response_bodies)) - 1
        return httpx.Response(
            status_code=200,
            json=response_bodies[index],
            headers={"content-type": "application/json"},
        )

    httpx_mock.add_callback(
        upstream_callback,
        url="https://chatgpt.com/backend-api/codex/responses",
        is_reusable=True,
    )

    setup_logging(json_logs=False, log_level_name="ERROR")

    settings = Settings(
        enable_plugins=True,
        plugins_disable_local_discovery=False,
        enabled_plugins=["codex", "oauth_codex"],
        plugins={
            "codex": {"enabled": True, "inject_detection_payload": False},
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

    async def init_detection_stub(self: Any) -> CodexCacheData:
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
        transport = httpx.ASGITransport(app=app)
        client = httpx.AsyncClient(transport=transport, base_url="http://test")
        try:
            yield client, upstream_payloads
        finally:
            await client.aclose()
            await shutdown_plugins(app)
            await service_container.close()


async def test_msaf_agent_runs_through_codex_proxy(
    msaf_codex_client: tuple[httpx.AsyncClient, list[dict[str, Any]]],
) -> None:
    """Single agent-style call verifies no CLI injection, proper flags, no thinking XML."""
    client, upstream_payloads = msaf_codex_client
    response = await client.post(
        "/codex/v1/chat/completions",
        json={
            "model": "gpt-5.4",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"{COMMON_INSTRUCTIONS} "
                        "Focus on fields, validations, and success criteria. "
                        "Output at most 5 bullets."
                    ),
                },
                {"role": "user", "content": "Составьте требования для формы логина."},
            ],
            "reasoning_effort": "medium",
            "max_completion_tokens": 256,
        },
    )

    assert response.status_code == 200, response.text
    data = response.json()
    assert_openai_responses_format(data)

    assert len(upstream_payloads) == 1
    payload = upstream_payloads[0]
    assert DETECTED_CLI_INSTRUCTIONS not in payload.get("instructions", "")
    assert payload.get("stream") is True
    assert payload.get("store") is False
    assert "<thinking>" not in json.dumps(data, ensure_ascii=False)

    text = _extract_message_text(data)
    assert "Email" in text
    assert "Password" in text


async def test_msaf_sequential_agents_keep_clean_messages(
    msaf_codex_client: tuple[httpx.AsyncClient, list[dict[str, Any]]],
) -> None:
    """Two sequential agent calls (analyst -> editor) keep reasoning hidden and output clean."""
    client, upstream_payloads = msaf_codex_client

    # Step 1: analyst call
    analyst_response = await client.post(
        "/codex/v1/chat/completions",
        json={
            "model": "gpt-5.4",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"{COMMON_INSTRUCTIONS} "
                        "Focus on fields, validations, and success criteria. "
                        "Output at most 5 bullets."
                    ),
                },
                {"role": "user", "content": "Составьте требования для формы логина."},
            ],
            "reasoning_effort": "medium",
        },
    )
    assert analyst_response.status_code == 200, analyst_response.text
    analyst_data = analyst_response.json()
    analyst_text = _extract_message_text(analyst_data)

    # Step 2: editor call, feeding analyst output as context
    editor_response = await client.post(
        "/codex/v1/chat/completions",
        json={
            "model": "gpt-5.4",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are the final editor for login form requirements. "
                        "Reply in the same language as the user request. "
                        "Produce one clean Markdown document with sections "
                        "Goal, Functional Requirements, Validation Rules, Acceptance Criteria."
                    ),
                },
                {"role": "user", "content": "Составьте требования для формы логина."},
                {
                    "role": "assistant",
                    "content": analyst_text,
                    "name": "ProductAnalyst",
                },
            ],
            "reasoning_effort": "medium",
        },
    )
    assert editor_response.status_code == 200, editor_response.text
    editor_data = editor_response.json()
    editor_text = _extract_message_text(editor_data)

    assert len(upstream_payloads) == 2
    assert "Hidden analyst reasoning" not in analyst_text
    assert "Hidden editor reasoning" not in editor_text
    assert "<thinking>" not in analyst_text
    assert "<thinking>" not in editor_text
    assert "## Goal" in editor_text
    assert "## Functional Requirements" in editor_text
