"""Shared fixtures for permissions integration tests to manage background tasks."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator

import pytest

from ccproxy.api.bootstrap import create_service_container
from ccproxy.core.async_task_manager import start_task_manager, stop_task_manager
from ccproxy.services.container import ServiceContainer


@pytest.fixture(autouse=True)
async def permissions_task_manager() -> AsyncGenerator[ServiceContainer, None]:
    """Ensure the async task manager is available for permission tests."""
    container = ServiceContainer.get_current(strict=False)
    if container is None:
        container = create_service_container()
    await start_task_manager(container=container)
    try:
        yield container
    finally:
        await stop_task_manager(container=container)
        with contextlib.suppress(Exception):
            await container.shutdown()


@pytest.fixture(autouse=True)
def permissions_task_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub managed task creation to regular asyncio tasks for deterministic cleanup."""

    async def _create_managed_task(
        coro,
        *,
        name=None,
        creator=None,
        cleanup_callback=None,
        **_kwargs,
    ):
        task = asyncio.create_task(coro, name=name)
        if cleanup_callback:
            task.add_done_callback(lambda _: cleanup_callback())
        return task

    monkeypatch.setattr(
        "ccproxy.plugins.permissions.service.create_managed_task",
        _create_managed_task,
        raising=False,
    )
