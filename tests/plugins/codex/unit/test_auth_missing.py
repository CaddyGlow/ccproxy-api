from contextlib import AsyncExitStack
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from ccproxy.models.detection import DetectedHeaders, DetectedPrompts
from ccproxy.plugins.codex.models import CodexCacheData


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

    blocked_hosts = {"chatgpt.com", "api.openai.com"}
    original_send = httpx.AsyncClient.send

    async def guard_send(
        self: Any, request: httpx.Request, *args: Any, **kwargs: Any
    ) -> Any:
        if request.url.host in blocked_hosts:
            raise AssertionError(f"Unexpected upstream call to {request.url!s}")
        return await original_send(self, request, *args, **kwargs)

    prompts = DetectedPrompts.from_body(
        {"instructions": "You are a helpful coding assistant."}
    )
    detection_data = CodexCacheData(
        codex_version="fallback",
        headers=DetectedHeaders({}),
        prompts=prompts,
        body_json=prompts.raw,
        method="POST",
        url="https://chatgpt.com/backend-codex/responses",
        path="/api/backend-codex/responses",
        query_params={},
    )

    async def init_detection_stub(self):  # type: ignore[no-untyped-def]
        self._cached_data = detection_data
        return detection_data

    async with AsyncExitStack() as stack:
        stack.enter_context(
            patch(
                "ccproxy.plugins.codex.detection_service.CodexDetectionService.initialize_detection",
                new=init_detection_stub,
            )
        )
        stack.enter_context(
            patch(
                "ccproxy.plugins.oauth_codex.manager.CodexTokenManager.load_credentials",
                new=AsyncMock(return_value=None),
            )
        )
        stack.enter_context(
            patch(
                "ccproxy.plugins.oauth_codex.manager.CodexTokenManager.get_profile_quick",
                new=AsyncMock(return_value=None),
            )
        )

        client = await integration_client_factory(plugin_configs)
        http = await stack.enter_async_context(client)

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
