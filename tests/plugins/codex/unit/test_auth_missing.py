from typing import Any
from unittest.mock import patch

import httpx
import pytest


@pytest.mark.unit
@pytest.mark.asyncio
async def test_codex_missing_auth_manager_returns_401(
    integration_client_factory: Any,
) -> None:
    plugin_configs = {
        "codex": {
            "enabled": True,
            "auth_manager": "missing_codex_manager",
        },
        "oauth_codex": {"enabled": True},
    }

    client = await integration_client_factory(plugin_configs)

    blocked_hosts = {"chatgpt.com", "api.openai.com"}
    original_send = httpx.AsyncClient.send

    async def guard_send(
        self: Any, request: httpx.Request, *args: Any, **kwargs: Any
    ) -> Any:
        if request.url.host in blocked_hosts:
            raise AssertionError(f"Unexpected upstream call to {request.url!s}")
        return await original_send(self, request, *args, **kwargs)

    async with client as http:
        with patch("httpx.AsyncClient.send", guard_send):
            resp = await http.post(
                "/codex/v1/responses",
                json={"model": "gpt-4o-mini", "input": []},
            )

        assert resp.status_code == 401
        body = resp.json()
        assert "error" in body
        if isinstance(body.get("error"), dict):
            assert "message" in body["error"]
