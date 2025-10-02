import asyncio
from typing import Any

import pytest

from ccproxy.core.plugins.hooks.base import HookContext
from ccproxy.core.plugins.hooks.events import HookEvent
from ccproxy.core.plugins.hooks.thread_manager import BackgroundHookThreadManager


class _Registry:
    def __init__(self, hooks: list[Any]) -> None:
        self._hooks = hooks

    def get(self, event: Any) -> list[Any]:  # pragma: no cover - simple pass-through
        return self._hooks


@pytest.mark.unit
@pytest.mark.asyncio
async def test_background_hook_manager_no_race_on_start(
    caplog: pytest.LogCaptureFixture,
) -> None:
    executed: list[int] = []

    async def hook_fn(ctx: HookContext) -> None:  # noqa: ARG001
        await asyncio.sleep(0.01)
        executed.append(1)

    registry = _Registry([hook_fn])
    manager = BackgroundHookThreadManager()

    manager.start()
    # Immediately emit without any sleep to simulate race window
    # Use an existing HookEvent value to avoid creating a new Enum
    ctx = HookContext(
        event=HookEvent.CUSTOM_EVENT, timestamp=None, data={}, metadata={}
    )
    manager.emit_async(ctx, registry)

    # allow background thread to process
    await asyncio.sleep(0.05)
    manager.stop()

    assert sum(executed) == 1
    logs = "\n".join(
        r.message if hasattr(r, "message") else str(r) for r in caplog.records
    )
    assert "background_thread_not_ready_dropping_task" not in logs
    assert "is bound to a different event loop" not in logs


@pytest.mark.unit
@pytest.mark.asyncio
async def test_background_hook_manager_lazy_start_emit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    executed: list[int] = []

    async def hook_fn(ctx: HookContext) -> None:  # noqa: ARG001
        await asyncio.sleep(0.005)
        executed.append(1)

    registry = _Registry([hook_fn])
    manager = BackgroundHookThreadManager()

    # Do not call start() explicitly; emit triggers lazy start
    ctx = HookContext(
        event=HookEvent.CUSTOM_EVENT, timestamp=None, data={}, metadata={}
    )
    manager.emit_async(ctx, registry)

    await asyncio.sleep(0.05)
    manager.stop()

    assert sum(executed) == 1
    logs = "\n".join(
        r.message if hasattr(r, "message") else str(r) for r in caplog.records
    )
    assert "background_thread_not_ready_dropping_task" not in logs
    assert "is bound to a different event loop" not in logs
