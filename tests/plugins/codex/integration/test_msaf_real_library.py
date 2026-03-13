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
from agent_framework import Message
from agent_framework.openai import OpenAIChatClient
from openai import AsyncOpenAI
from pytest_httpx import HTTPXMock

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


@pytest_asyncio.fixture
async def msaf_codex_client(
    httpx_mock: HTTPXMock,
) -> AsyncGenerator[tuple[OpenAIChatClient, list[dict[str, Any]]], None]:
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
        llm={"openai_thinking_xml": False},
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

    http_client: httpx.AsyncClient | None = None
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
        try:
            await initialize_plugins_startup(app, settings)
            transport = httpx.ASGITransport(app=app)
            http_client = httpx.AsyncClient(
                transport=transport,
                base_url="http://test",
            )
            async_client = AsyncOpenAI(
                api_key="ccproxy",
                base_url="http://test/codex/v1",
                http_client=http_client,
            )
            client = OpenAIChatClient(
                model_id="gpt-5.4",
                async_client=async_client,
            )
            yield client, upstream_payloads
        finally:
            if http_client is not None:
                await http_client.aclose()
            await shutdown_plugins(app)
            await service_container.close()


async def test_msaf_real_library_agent_runs_through_codex_proxy(
    msaf_codex_client: tuple[OpenAIChatClient, list[dict[str, Any]]],
) -> None:
    client, upstream_payloads = msaf_codex_client
    response = await client.get_response(
        [Message("user", ["Составьте требования для формы логина."])],
        options={
            "instructions": (
                f"{COMMON_INSTRUCTIONS} "
                "Focus on fields, validations, and success criteria. "
                "Output at most 5 bullets."
            )
        },
    )

    assert len(upstream_payloads) == 1
    assert all(
        DETECTED_CLI_INSTRUCTIONS not in payload.get("instructions", "")
        for payload in upstream_payloads
    )
    assert all(payload.get("stream") is True for payload in upstream_payloads)
    assert all(payload.get("store") is False for payload in upstream_payloads)
    assert "Detected Codex CLI instructions" not in upstream_payloads[0].get(
        "instructions", ""
    )
    assert "<thinking>" not in response.text
    assert "Email" in response.text
    assert "Password" in response.text


async def test_msaf_real_library_sequential_agents_keep_clean_messages(
    msaf_codex_client: tuple[OpenAIChatClient, list[dict[str, Any]]],
) -> None:
    client, upstream_payloads = msaf_codex_client
    analyst_response = await client.get_response(
        [Message("user", ["Составьте требования для формы логина."])],
        options={
            "instructions": (
                f"{COMMON_INSTRUCTIONS} "
                "Focus on fields, validations, and success criteria. "
                "Output at most 5 bullets."
            )
        },
    )
    editor_response = await client.get_response(
        [
            Message("user", ["Составьте требования для формы логина."], author_name="user"),
            Message(
                "assistant",
                [analyst_response.text],
                author_name="ProductAnalyst",
            ),
        ],
        options={
            "instructions": (
                "You are the final editor for login form requirements. "
                "Reply in the same language as the user request. "
                "Produce one clean Markdown document with sections "
                "Goal, Functional Requirements, Validation Rules, Acceptance Criteria."
            )
        },
    )

    assert len(upstream_payloads) == 2
    assert "Hidden analyst reasoning" not in analyst_response.text
    assert "Hidden editor reasoning" not in editor_response.text
    assert "<thinking>" not in analyst_response.text
    assert "<thinking>" not in editor_response.text
    assert "## Goal" in editor_response.text
    assert "## Functional Requirements" in editor_response.text
