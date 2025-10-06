from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, cast

import anyio
import httpx
import pytest

from ccproxy.core.async_task_manager import AsyncTaskManager
from ccproxy.plugins.permissions.handlers.cli import SSEConfirmationHandler
from ccproxy.plugins.permissions.models import PermissionRequest


class _DummyTerminalHandler:
    def __init__(self) -> None:
        self.handled: list[str] = []
        self.cancelled: list[tuple[str, str]] = []

    async def handle_permission(self, request: PermissionRequest) -> bool:
        self.handled.append(request.id)
        await anyio.sleep(0)
        return True

    def cancel_confirmation(self, request_id: str, reason: str = "cancelled") -> None:
        self.cancelled.append((request_id, reason))


class _DummyHttpClient:
    async def post(self, url: str, json: Any) -> httpx.Response:  # noqa: ANN401
        await anyio.sleep(0)
        return httpx.Response(status_code=200)

    async def aclose(self) -> None:
        await anyio.sleep(0)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_sse_confirmation_handler_stress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = AsyncTaskManager(max_tasks=512, shutdown_timeout=5.0)
    await manager.start()

    async def create_task_stub(coro, *args, **kwargs):  # noqa: ANN001
        return await manager.create_task(coro, *args, **kwargs)

    monkeypatch.setattr(
        "ccproxy.plugins.permissions.handlers.cli.create_managed_task",
        create_task_stub,
    )

    async def fast_sleep(delay: float) -> None:
        await anyio.sleep(0)

    monkeypatch.setattr(
        "ccproxy.plugins.permissions.handlers.cli.runtime_sleep",
        fast_sleep,
    )

    handler = SSEConfirmationHandler(
        api_url="https://example.test",
        terminal_handler=_DummyTerminalHandler(),
        ui=True,
        auto_reconnect=False,
    )
    handler.client = cast(httpx.AsyncClient, _DummyHttpClient())

    total_requests = 80
    request_ids = [f"perm-{idx}" for idx in range(total_requests)]

    async def emit_request(request_id: str) -> None:
        expires_at = (datetime.now(UTC) + timedelta(minutes=5)).isoformat()
        data = {
            "request_id": request_id,
            "tool_name": "stress-tool",
            "input": {"command": "echo"},
            "expires_at": expires_at,
        }
        await handler._handle_permission_request(data)

    async with anyio.create_task_group() as tg:
        for request_id in request_ids:
            tg.start_soon(emit_request, request_id)

    await anyio.sleep(0.05)

    async def resolve_request(request_id: str) -> None:
        await handler._handle_permission_resolved(
            {"request_id": request_id, "allowed": True}
        )

    async with anyio.create_task_group() as tg:
        for request_id in request_ids:
            tg.start_soon(resolve_request, request_id)

    await anyio.sleep(0.05)

    terminal_handler = cast(_DummyTerminalHandler, handler.terminal_handler)
    assert len(terminal_handler.handled) == total_requests
    assert handler._ongoing_requests == {}

    await manager.stop()
