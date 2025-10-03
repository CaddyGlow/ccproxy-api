from contextlib import AsyncExitStack
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from ccproxy.plugins.copilot.models import CopilotCacheData


@pytest.mark.unit
@pytest.mark.asyncio
async def test_copilot_missing_auth_manager_returns_401(
    integration_client_factory: Any,
) -> None:
    plugin_configs = {
        "copilot": {
            "enabled": True,
            "auth_manager": "missing_copilot_manager",
        }
    }

    blocked_hosts = {"api.githubcopilot.com", "api.github.com"}
    original_send = httpx.AsyncClient.send

    async def guard_send(
        self: httpx.AsyncClient, request: httpx.Request, *args: Any, **kwargs: Any
    ) -> httpx.Response:
        if request.url.host in blocked_hosts:
            raise AssertionError(f"Unexpected upstream call to {request.url!s}")
        return await original_send(self, request, *args, **kwargs)

    detection_data = CopilotCacheData(
        cli_available=False,
        cli_version=None,
        auth_status="not_authenticated",
        username=None,
    )

    async def init_detection_stub(self):  # type: ignore[no-untyped-def]
        self._cache = detection_data
        return detection_data

    async with AsyncExitStack() as stack:
        stack.enter_context(
            patch(
                "ccproxy.plugins.copilot.detection_service.CopilotDetectionService.initialize_detection",
                new=init_detection_stub,
            )
        )
        stack.enter_context(
            patch(
                "ccproxy.plugins.copilot.manager.CopilotTokenManager.ensure_copilot_token",
                new=AsyncMock(return_value="copilot_test_service_token"),
            )
        )
        stack.enter_context(
            patch(
                "ccproxy.plugins.copilot.oauth.provider.CopilotOAuthProvider.ensure_oauth_token",
                new=AsyncMock(return_value="gh_oauth_access_token"),
            )
        )

        client = await integration_client_factory(plugin_configs)
        http = await stack.enter_async_context(client)

        with patch("httpx.AsyncClient.send", guard_send):
            resp = await http.post(
                "/copilot/v1/chat/completions",
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )

        assert resp.status_code == 401
        body = resp.json()
        assert "error" in body
        if isinstance(body.get("error"), dict):
            assert body["error"].get("message")
