"""Async hook execution manager built on top of the runtime task manager."""

from __future__ import annotations

import asyncio
import inspect
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

import anyio
import structlog

from ccproxy.core.async_runtime import create_task as runtime_create_task
from ccproxy.core.async_task_manager import AsyncTaskManager, create_managed_task

from .base import Hook, HookContext


logger = structlog.get_logger(__name__).bind(component="hook_executor")


@dataclass
class HookTask:
    """Represents a hook execution task."""

    context: HookContext
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)


class BackgroundHookThreadManager:
    """Fire-and-forget hook execution via the async task manager."""

    def __init__(self, task_manager: AsyncTaskManager | None = None) -> None:
        self._task_manager = task_manager
        self._tasks: set[asyncio.Task[Any]] = set()
        self._lock = asyncio.Lock()

    async def emit_async(self, context: HookContext, registry: Any) -> None:
        """Schedule hook execution without blocking the caller."""

        hook_task = HookTask(context=context)

        async def runner() -> None:
            task = asyncio.current_task()
            try:
                await self._execute_task(hook_task, registry)
            finally:
                if task:
                    async with self._lock:
                        self._tasks.discard(task)

        task_name = f"hook_{getattr(context.event, 'value', context.event)}"

        if self._task_manager:
            task = await self._task_manager.create_task(
                runner(), name=task_name, creator="HookManager"
            )
        else:
            task = runtime_create_task(runner(), name=task_name)

        async with self._lock:
            self._tasks.add(task)

    async def stop(self, timeout: float = 5.0) -> None:
        """Cancel any outstanding hook executions."""

        async with self._lock:
            tasks = list(self._tasks)
            self._tasks.clear()

        if not tasks:
            return

        for task in tasks:
            task.cancel()

        try:
            with anyio.move_on_after(timeout):
                await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("hook_executor_stop_error", error=str(exc), exc_info=True)

    async def _execute_task(self, task: HookTask, registry: Any) -> None:
        try:
            hooks = registry.get(task.context.event)
            if not hooks:
                return

            for hook in hooks:
                try:
                    await self._execute_hook(hook, task.context)
                except Exception as exc:
                    logger.error(
                        "background_hook_execution_failed",
                        hook=hook.name,
                        event_type=task.context.event.value
                        if hasattr(task.context.event, "value")
                        else str(task.context.event),
                        error=str(exc),
                        task_id=task.task_id,
                    )
        except Exception as exc:
            logger.error(
                "hook_task_execution_failed", error=str(exc), task_id=task.task_id
            )

    async def _execute_hook(self, hook: Hook, context: HookContext) -> None:
        result = hook(context)
        if inspect.isawaitable(result):
            await result

    # Backwards compatibility helper used by HookManager.shutdown()
    async def shutdown(self) -> None:
        await self.stop()
