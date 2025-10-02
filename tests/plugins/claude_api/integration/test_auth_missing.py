from unittest.mock import patch

import httpx
import pytest


@pytest.mark.asyncio
@pytest.mark.integration
async def test_claude_api_missing_auth_manager_returns_401_integration(
    integration_client_factory,  # type: ignore[no-untyped-def]
) -> None:
    plugin_configs = {
        "claude_api": {
            "enabled": True,
            "auth_manager": "missing_claude_manager",
        },
        "oauth_claude": {"enabled": True},
    }

    client = await integration_client_factory(plugin_configs)

    blocked_hosts = {"api.anthropic.com"}
    original_send = httpx.AsyncClient.send

    async def guard_send(self, request: httpx.Request, *args, **kwargs):
        if request.url.host in blocked_hosts:
            raise AssertionError(f"Unexpected upstream call to {request.url!s}")
        return await original_send(self, request, *args, **kwargs)

    async with client as http:
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
