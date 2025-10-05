"""Centralized async task management with unified lifecycle control."""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from asyncio import InvalidStateError
from asyncio import Task as AsyncTask
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

import anyio

from ccproxy.core.async_runtime import CancelledError, create_lock
from ccproxy.core.async_runtime import create_task as runtime_create_task
from ccproxy.core.logging import TraceBoundLogger, get_logger


if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from ccproxy.services.container import ServiceContainer


T = TypeVar("T")

logger: TraceBoundLogger = get_logger(__name__)


@dataclass
class TaskInfo:
    """Information about a managed task."""

    task: AsyncTask[Any]
    name: str
    created_at: float
    creator: str | None = None
    cleanup_callback: Callable[[], None] | None = None
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

    @property
    def is_done(self) -> bool:
        return self.task.done()

    @property
    def is_cancelled(self) -> bool:
        return self.task.cancelled()

    def get_exception(self) -> BaseException | None:
        if self.task.done() and not self.task.cancelled():
            with contextlib.suppress(InvalidStateError):
                return self.task.exception()
        return None


class AsyncTaskManager:
    """Centralized manager for async tasks with lifecycle control."""

    def __init__(
        self,
        cleanup_interval: float = 30.0,
        shutdown_timeout: float = 30.0,
        max_tasks: int = 1000,
    ):
        self.cleanup_interval = cleanup_interval
        self.shutdown_timeout = shutdown_timeout
        self.max_tasks = max_tasks

        self._tasks: dict[str, TaskInfo] = {}
        self._lock = create_lock()
        self._started = False
        self._active_tasks = 0
        self._completed_tasks = 0
        self._total_tasks_created = 0

    async def start(self) -> None:
        """Mark the task manager as ready."""
        if self._started:
            logger.warning("task_manager_already_started")
            return

        self._started = True
        logger.debug("task_manager_started")

    async def stop(self) -> None:
        """Cancel all managed tasks and reset state."""
        if not self._started:
            return

        logger.debug("task_manager_stopping", active_tasks=len(self._tasks))

        async with self._lock:
            tasks_to_cancel: list[AsyncTask[Any]] = []
            for info in self._tasks.values():
                if not info.task.done():
                    info.task.cancel()
                tasks_to_cancel.append(info.task)

        if tasks_to_cancel:
            try:
                with anyio.move_on_after(self.shutdown_timeout) as scope:
                    await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
                if scope.cancel_called:
                    logger.warning(
                        "task_cancellation_timeout",
                        timeout=self.shutdown_timeout,
                        remaining_tasks=sum(
                            not task.done() for task in tasks_to_cancel
                        ),
                    )
            except Exception as exc:  # pragma: no cover - defensive
                logger.error(
                    "task_manager_shutdown_error",
                    error=str(exc),
                    error_type=type(exc).__name__,
                    exc_info=True,
                )

        async with self._lock:
            self._tasks.clear()
            self._active_tasks = 0

        self._started = False
        logger.debug("task_manager_stopped")

    async def create_task(
        self,
        coro: Coroutine[Any, Any, T],
        *,
        name: str | None = None,
        creator: str | None = None,
        cleanup_callback: Callable[[], None] | None = None,
    ) -> AsyncTask[T]:
        """Create a managed asyncio task."""
        if not self._started:
            raise RuntimeError("Task manager is not started")

        async with self._lock:
            if self._active_tasks >= self.max_tasks:
                logger.warning(
                    "task_manager_at_capacity",
                    current_tasks=self._active_tasks,
                    max_tasks=self.max_tasks,
                )
                raise RuntimeError(f"Task manager at capacity ({self.max_tasks} tasks)")
            self._active_tasks += 1
            task_name = name or f"managed_task_{self._active_tasks}"

        async def managed() -> T:
            try:
                return await self._wrap_with_exception_handling(coro, task_name)
            finally:
                if cleanup_callback:
                    with contextlib.suppress(Exception):
                        cleanup_callback()
                await self._remove_task(task_id)

        task = runtime_create_task(managed(), name=task_name)
        task_info = TaskInfo(
            task=task,
            name=task_name,
            created_at=time.time(),
            creator=creator,
            cleanup_callback=cleanup_callback,
        )

        task_id = task_info.task_id

        async with self._lock:
            self._tasks[task_id] = task_info
            self._total_tasks_created += 1

        logger.debug(
            "task_created",
            task_id=task_id,
            task_name=task_name,
            creator=creator,
            total_tasks=len(self._tasks),
        )

        return task

    async def _remove_task(self, task_id: str) -> None:
        """Remove a task from tracking when it completes."""
        async with self._lock:
            self._tasks.pop(task_id, None)
            self._active_tasks = max(0, self._active_tasks - 1)
            self._completed_tasks += 1

    async def _wrap_with_exception_handling(
        self, coro: Coroutine[Any, Any, T], task_name: str
    ) -> T:
        try:
            return await coro
        except CancelledError:
            logger.debug("task_cancelled", task_name=task_name)
            raise
        except Exception as e:
            logger.error(
                "task_exception",
                task_name=task_name,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            raise

    async def get_task_stats(self) -> dict[str, Any]:
        async with self._lock:
            active_tasks = sum(1 for t in self._tasks.values() if not t.is_done)
            cancelled_tasks = sum(1 for t in self._tasks.values() if t.is_cancelled)
            failed_tasks = sum(
                1
                for t in self._tasks.values()
                if t.is_done and not t.is_cancelled and t.get_exception()
            )
            return {
                "total_tasks": self._total_tasks_created,
                "active_tasks": active_tasks,
                "cancelled_tasks": cancelled_tasks,
                "failed_tasks": failed_tasks,
                "completed_tasks": self._completed_tasks,
                "started": self._started,
                "max_tasks": self.max_tasks,
            }

    async def list_active_tasks(self) -> list[dict[str, Any]]:
        async with self._lock:
            return [
                {
                    "task_id": info.task_id,
                    "name": info.name,
                    "creator": info.creator,
                    "age_seconds": info.age_seconds,
                    "created_at": info.created_at,
                }
                for info in self._tasks.values()
                if not info.is_done
            ]

    @property
    def is_started(self) -> bool:
        return self._started


# Dependency-injected helpers


def _resolve_task_manager(
    *,
    container: ServiceContainer | None = None,
    task_manager: AsyncTaskManager | None = None,
) -> AsyncTaskManager:
    if task_manager is not None:
        return task_manager

    from ccproxy.services.container import ServiceContainer as _ServiceContainer

    if container is not None:
        resolved_container: _ServiceContainer = container
    else:
        resolved_container_maybe = _ServiceContainer.get_current(strict=False)
        if resolved_container_maybe is None:
            raise RuntimeError(
                "ServiceContainer is not available; provide a container or task manager"
            )
        resolved_container = resolved_container_maybe

    try:
        return resolved_container.get_async_task_manager()
    except Exception as exc:
        raise RuntimeError(
            "AsyncTaskManager is not registered in the provided ServiceContainer"
        ) from exc


async def create_managed_task(
    coro: Coroutine[Any, Any, T],
    *,
    name: str | None = None,
    creator: str | None = None,
    cleanup_callback: Callable[[], None] | None = None,
    container: ServiceContainer | None = None,
    task_manager: AsyncTaskManager | None = None,
) -> AsyncTask[T]:
    manager = _resolve_task_manager(container=container, task_manager=task_manager)
    return await manager.create_task(
        coro, name=name, creator=creator, cleanup_callback=cleanup_callback
    )


async def start_task_manager(
    *,
    container: ServiceContainer | None = None,
    task_manager: AsyncTaskManager | None = None,
) -> None:
    manager = _resolve_task_manager(container=container, task_manager=task_manager)
    await manager.start()


async def stop_task_manager(
    *,
    container: ServiceContainer | None = None,
    task_manager: AsyncTaskManager | None = None,
) -> None:
    manager = _resolve_task_manager(container=container, task_manager=task_manager)
    await manager.stop()


def create_fire_and_forget_task(
    coro: Coroutine[Any, Any, T],
    *,
    name: str | None = None,
    creator: str | None = None,
    container: ServiceContainer | None = None,
    task_manager: AsyncTaskManager | None = None,
) -> None:
    manager = _resolve_task_manager(container=container, task_manager=task_manager)

    if not manager.is_started:
        logger.warning(
            "task_manager_not_started_fire_and_forget",
            name=name,
            creator=creator,
        )
        runtime_create_task(coro, name=name)
        return

    async def _create_managed_task() -> None:
        await manager.create_task(coro, name=name, creator=creator)

    runtime_create_task(_create_managed_task(), name=f"create_{name or 'unnamed'}")
