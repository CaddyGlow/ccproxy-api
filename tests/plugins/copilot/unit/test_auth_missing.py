from unittest.mock import patch

import httpx
import pytest


@pytest.mark.unit
@pytest.mark.asyncio
async def test_copilot_missing_auth_manager_returns_401(
    integration_client_factory,  # type: ignore[no-untyped-def]
) -> None:
    plugin_configs = {
        "copilot": {
            "enabled": True,
            "auth_manager": "missing_copilot_manager",
        }
    }

    client = await integration_client_factory(plugin_configs)

    blocked_hosts = {"api.githubcopilot.com", "api.github.com"}
    original_send = httpx.AsyncClient.send

    async def guard_send(self, request: httpx.Request, *args, **kwargs):
        if request.url.host in blocked_hosts:
            raise AssertionError(f"Unexpected upstream call to {request.url!s}")
        return await original_send(self, request, *args, **kwargs)

    async with client as http:
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
