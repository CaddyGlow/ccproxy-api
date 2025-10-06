import asyncio
from datetime import UTC, datetime
from typing import Any

import anyio
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

    ctx = HookContext(
        event=HookEvent.CUSTOM_EVENT, timestamp=datetime.now(UTC), data={}, metadata={}
    )
    await manager.emit_async(ctx, registry)

    await asyncio.sleep(0.05)
    await manager.stop()

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

    ctx = HookContext(
        event=HookEvent.CUSTOM_EVENT, timestamp=datetime.now(UTC), data={}, metadata={}
    )
    await manager.emit_async(ctx, registry)

    await asyncio.sleep(0.05)
    await manager.stop()

    assert sum(executed) == 1
    logs = "\n".join(
        r.message if hasattr(r, "message") else str(r) for r in caplog.records
    )
    assert "background_thread_not_ready_dropping_task" not in logs
    assert "is bound to a different event loop" not in logs


@pytest.mark.unit
@pytest.mark.slow
@pytest.mark.asyncio
async def test_background_hook_manager_stress_high_concurrency() -> None:
    total_events = 250
    processed = 0
    processed_lock = anyio.Lock()

    async def hook_fn(ctx: HookContext) -> None:  # noqa: ARG001
        nonlocal processed
        async with processed_lock:
            processed += 1
        await anyio.sleep(0)

    registry = _Registry([hook_fn])
    manager = BackgroundHookThreadManager()

    async with anyio.create_task_group() as tg:
        for _ in range(total_events):
            ctx = HookContext(
                event=HookEvent.CUSTOM_EVENT,
                timestamp=datetime.now(UTC),
                data={},
                metadata={},
            )
            tg.start_soon(manager.emit_async, ctx, registry)

    with anyio.fail_after(3.0):
        while True:
            async with processed_lock:
                if processed >= total_events:
                    break
            await anyio.sleep(0.01)

    await manager.stop()

    assert processed == total_events
