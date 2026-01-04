from contextlib import AsyncExitStack
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from ccproxy.models.detection import DetectedHeaders, DetectedPrompts
from ccproxy.plugins.claude_api.models import ClaudeCacheData


@pytest.mark.unit
@pytest.mark.asyncio
async def test_claude_api_missing_auth_manager_returns_401(
    integration_client_factory: object,
) -> None:
    plugin_configs = {
        "claude_api": {
            "enabled": True,
            "auth_manager": "missing_claude_manager",
        },
        "oauth_claude": {"enabled": True},
    }

    blocked_hosts = {"api.anthropic.com"}
    original_send = httpx.AsyncClient.send

    async def guard_send(
        self: httpx.AsyncClient, request: httpx.Request, *args: object, **kwargs: object
    ) -> httpx.Response:
        if request.url.host in blocked_hosts:
            raise AssertionError(f"Unexpected upstream call to {request.url!s}")
        return await original_send(self, request, *args, **kwargs)  # type: ignore[arg-type]

    prompts = DetectedPrompts.from_body(
        {"system": [{"type": "text", "text": "Hello from tests."}]}
    )
    detection_data = ClaudeCacheData(
        claude_version="fallback",
        headers=DetectedHeaders({}),
        prompts=prompts,
        body_json=prompts.raw,
        method="POST",
        url=None,
        path=None,
        query_params=None,
    )

    async def init_detection_stub(self):  # type: ignore[no-untyped-def]
        self._cached_data = detection_data
        return detection_data

    async with AsyncExitStack() as stack:
        stack.enter_context(
            patch(
                "ccproxy.plugins.claude_api.detection_service.ClaudeAPIDetectionService.initialize_detection",
                new=init_detection_stub,
            )
        )
        stack.enter_context(
            patch(
                "ccproxy.plugins.oauth_claude.manager.ClaudeApiTokenManager.get_access_token",
                new=AsyncMock(return_value="test-claude-access-token"),
            )
        )
        stack.enter_context(
            patch(
                "ccproxy.plugins.oauth_claude.manager.ClaudeApiTokenManager.load_credentials",
                new=AsyncMock(return_value=None),
            )
        )

        client = await integration_client_factory(plugin_configs)  # type: ignore[operator]
        http = await stack.enter_async_context(client)

        with patch("httpx.AsyncClient.send", guard_send):
            resp = await http.post(
                "/claude/v1/messages",
                json={
                    "model": "claude-3-haiku",
                    "messages": [],
                    "max_tokens": 128,
                },
            )

        assert resp.status_code == 401
        body = resp.json()
        assert "error" in body
        if isinstance(body.get("error"), dict):
            assert body["error"].get("message")
