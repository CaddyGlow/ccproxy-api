from unittest.mock import patch

import httpx
import pytest


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

    client = await integration_client_factory(plugin_configs)  # type: ignore[operator]

    blocked_hosts = {"api.anthropic.com"}
    original_send = httpx.AsyncClient.send

    async def guard_send(
        self: httpx.AsyncClient, request: httpx.Request, *args: object, **kwargs: object
    ) -> httpx.Response:
        if request.url.host in blocked_hosts:
            raise AssertionError(f"Unexpected upstream call to {request.url!s}")
        return await original_send(self, request, *args, **kwargs)  # type: ignore[arg-type]

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
        body = resp.json()
        assert "error" in body
        if isinstance(body.get("error"), dict):
            assert body["error"].get("message")
