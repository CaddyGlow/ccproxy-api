"""Centralized async task management for lifecycle control and resource cleanup.

This module provides a centralized task manager that tracks all spawned async tasks,
handles proper cancellation on shutdown, and provides exception handling for
background tasks to prevent resource leaks and unhandled exceptions.
"""

import asyncio
import contextlib
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Optional, TypeVar

from ccproxy.core.logging import TraceBoundLogger, get_logger


if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from ccproxy.services.container import ServiceContainer


T = TypeVar("T")

logger: TraceBoundLogger = get_logger(__name__)


class TaskInfo:
    """Information about a managed task."""

    def __init__(
        self,
        task: asyncio.Task[Any],
        name: str,
        created_at: float,
        creator: str | None = None,
        cleanup_callback: Callable[[], None] | None = None,
    ):
        self.task = task
        self.name = name
        self.created_at = created_at
        self.creator = creator
        self.cleanup_callback = cleanup_callback
        self.task_id = str(uuid.uuid4())

    @property
    def age_seconds(self) -> float:
        """Get the age of the task in seconds."""
        return time.time() - self.created_at

    @property
    def is_done(self) -> bool:
        """Check if the task is done."""
        return self.task.done()

    @property
    def is_cancelled(self) -> bool:
        """Check if the task was cancelled."""
        return self.task.cancelled()

    def get_exception(self) -> BaseException | None:
        """Get the exception if the task failed."""
        if self.task.done() and not self.task.cancelled():
            try:
                return self.task.exception()
            except asyncio.InvalidStateError:
                return None
        return None


class AsyncTaskManager:
    """Centralized manager for async tasks with lifecycle control.

    This class provides:
    - Task registration and tracking
    - Automatic cleanup of completed tasks
    - Graceful shutdown with cancellation
    - Exception handling for background tasks
    - Task monitoring and statistics
    """

    def __init__(
        self,
        cleanup_interval: float = 30.0,
        shutdown_timeout: float = 30.0,
        max_tasks: int = 1000,
    ):
        """Initialize the task manager.

        Args:
            cleanup_interval: Interval for cleaning up completed tasks (seconds)
            shutdown_timeout: Timeout for graceful shutdown (seconds)
            max_tasks: Maximum number of tasks to track (prevents memory leaks)
        """
        self.cleanup_interval = cleanup_interval
        self.shutdown_timeout = shutdown_timeout
        self.max_tasks = max_tasks

        self._tasks: dict[str, TaskInfo] = {}
        self._lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()
        self._cleanup_task: asyncio.Task[None] | None = None
        self._started = False

    async def start(self) -> None:
        """Start the task manager and its cleanup task."""
        if self._started:
            logger.warning("task_manager_already_started")
            return

        self._started = True
        logger.debug("task_manager_starting", cleanup_interval=self.cleanup_interval)

        # Start cleanup task
        self._cleanup_task = asyncio.create_task(
            self._cleanup_loop(), name="task_manager_cleanup"
        )

        logger.debug("task_manager_started")

    async def stop(self) -> None:
        """Stop the task manager and cancel all managed tasks."""
        if not self._started:
            return

        logger.debug("task_manager_stopping", active_tasks=len(self._tasks))
        self._shutdown_event.set()

        # Stop cleanup task first
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task

        # Cancel all managed tasks
        await self._cancel_all_tasks()

        # Clear task registry
        async with self._lock:
            self._tasks.clear()

        self._started = False
        logger.debug("task_manager_stopped")

    async def create_task(
        self,
        coro: Awaitable[T],
        *,
        name: str | None = None,
        creator: str | None = None,
        cleanup_callback: Callable[[], None] | None = None,
    ) -> asyncio.Task[T]:
        """Create a managed task.

        Args:
            coro: Coroutine to execute
            name: Optional name for the task (auto-generated if None)
            creator: Optional creator identifier for debugging
            cleanup_callback: Optional callback to run when task completes

        Returns:
            The created task

        Raises:
            RuntimeError: If task manager is not started or has too many tasks
        """
        if not self._started:
            raise RuntimeError("Task manager is not started")

        # Check task limit
        if len(self._tasks) >= self.max_tasks:
            logger.warning(
                "task_manager_at_capacity",
                current_tasks=len(self._tasks),
                max_tasks=self.max_tasks,
            )
            # Clean up completed tasks to make room
            await self._cleanup_completed_tasks()

            if len(self._tasks) >= self.max_tasks:
                raise RuntimeError(f"Task manager at capacity ({self.max_tasks} tasks)")

        # Generate name if not provided
        if name is None:
            name = f"managed_task_{len(self._tasks)}"

        # Create the task with exception handling
        task = asyncio.create_task(
            self._wrap_with_exception_handling(coro, name),
            name=name,
        )

        # Register the task
        task_info = TaskInfo(
            task=task,
            name=name,
            created_at=time.time(),
            creator=creator,
            cleanup_callback=cleanup_callback,
        )

        async with self._lock:
            self._tasks[task_info.task_id] = task_info

        # Add done callback for automatic cleanup
        task.add_done_callback(lambda t: self._schedule_cleanup_callback(task_info))

        logger.debug(
            "task_created",
            task_id=task_info.task_id,
            task_name=name,
            creator=creator,
            total_tasks=len(self._tasks),
        )

        return task

    async def _wrap_with_exception_handling(
        self, coro: Awaitable[T], task_name: str
    ) -> T:
        """Wrap coroutine with exception handling."""
        try:
            return await coro
        except asyncio.CancelledError:
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

    def _schedule_cleanup_callback(self, task_info: TaskInfo) -> None:
        """Schedule cleanup callback for completed task."""
        try:
            # Run cleanup callback if provided
            if task_info.cleanup_callback:
                task_info.cleanup_callback()
        except Exception as e:
            logger.warning(
                "task_cleanup_callback_failed",
                task_id=task_info.task_id,
                task_name=task_info.name,
                error=str(e),
                exc_info=True,
            )

    async def _cleanup_loop(self) -> None:
        """Background loop for cleaning up completed tasks."""
        logger.debug("task_cleanup_loop_started")

        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(), timeout=self.cleanup_interval
                )
                break  # Shutdown event set
            except TimeoutError:
                pass  # Continue with cleanup

            await self._cleanup_completed_tasks()

        logger.debug("task_cleanup_loop_stopped")

    async def _cleanup_completed_tasks(self) -> None:
        """Clean up completed tasks from the registry."""
        completed_tasks = []

        async with self._lock:
            for task_id, task_info in list(self._tasks.items()):
                if task_info.is_done:
                    completed_tasks.append((task_id, task_info))
                    del self._tasks[task_id]

        if completed_tasks:
            logger.debug(
                "tasks_cleaned_up",
                completed_count=len(completed_tasks),
                remaining_tasks=len(self._tasks),
            )

            # Log any task exceptions
            for task_id, task_info in completed_tasks:
                if task_info.get_exception():
                    logger.warning(
                        "completed_task_had_exception",
                        task_id=task_id,
                        task_name=task_info.name,
                        exception=str(task_info.get_exception()),
                    )

    async def _cancel_all_tasks(self) -> None:
        """Cancel all managed tasks with timeout."""
        if not self._tasks:
            return

        logger.debug("cancelling_all_tasks", task_count=len(self._tasks))

        # Cancel all tasks
        tasks_to_cancel = []
        async with self._lock:
            for task_info in self._tasks.values():
                if not task_info.is_done:
                    task_info.task.cancel()
                    tasks_to_cancel.append(task_info.task)

        if not tasks_to_cancel:
            return

        # Wait for cancellation with timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks_to_cancel, return_exceptions=True),
                timeout=self.shutdown_timeout,
            )
            logger.debug("all_tasks_cancelled_gracefully")
        except TimeoutError:
            logger.warning(
                "task_cancellation_timeout",
                timeout=self.shutdown_timeout,
                remaining_tasks=sum(1 for t in tasks_to_cancel if not t.done()),
            )

    async def get_task_stats(self) -> dict[str, Any]:
        """Get statistics about managed tasks."""
        async with self._lock:
            active_tasks = sum(1 for t in self._tasks.values() if not t.is_done)
            cancelled_tasks = sum(1 for t in self._tasks.values() if t.is_cancelled)
            failed_tasks = sum(
                1
                for t in self._tasks.values()
                if t.is_done and not t.is_cancelled and t.get_exception()
            )

            return {
                "total_tasks": len(self._tasks),
                "active_tasks": active_tasks,
                "cancelled_tasks": cancelled_tasks,
                "failed_tasks": failed_tasks,
                "completed_tasks": len(self._tasks) - active_tasks,
                "started": self._started,
                "max_tasks": self.max_tasks,
            }

    async def list_active_tasks(self) -> list[dict[str, Any]]:
        """Get list of active tasks with details."""
        active_tasks = []

        async with self._lock:
            for task_info in self._tasks.values():
                if not task_info.is_done:
                    active_tasks.append(
                        {
                            "task_id": task_info.task_id,
                            "name": task_info.name,
                            "creator": task_info.creator,
                            "age_seconds": task_info.age_seconds,
                            "created_at": task_info.created_at,
                        }
                    )

        return active_tasks

    @property
    def is_started(self) -> bool:
        """Check if the task manager is started."""
        return self._started


# Dependency-injected access helpers


def _resolve_task_manager(
    *,
    container: Optional["ServiceContainer"] = None,
    task_manager: Optional["AsyncTaskManager"] = None,
) -> "AsyncTaskManager":
    """Resolve the async task manager instance using dependency injection.

    Args:
        container: Optional service container to resolve the manager from
        task_manager: Optional explicit manager instance (takes precedence)

    Returns:
        AsyncTaskManager instance

    Raises:
        RuntimeError: If the manager cannot be resolved
    """

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
    coro: Awaitable[T],
    *,
    name: str | None = None,
    creator: str | None = None,
    cleanup_callback: Callable[[], None] | None = None,
    container: Optional["ServiceContainer"] = None,
    task_manager: Optional["AsyncTaskManager"] = None,
) -> asyncio.Task[T]:
    """Create a managed task using the dependency-injected task manager.

    Args:
        coro: Coroutine to execute
        name: Optional name for the task
        creator: Optional creator identifier
        cleanup_callback: Optional cleanup callback
        container: Optional service container for resolving the task manager
        task_manager: Optional explicit task manager instance

    Returns:
        The created managed task
    """

    manager = _resolve_task_manager(container=container, task_manager=task_manager)
    return await manager.create_task(
        coro, name=name, creator=creator, cleanup_callback=cleanup_callback
    )


async def start_task_manager(
    *,
    container: Optional["ServiceContainer"] = None,
    task_manager: Optional["AsyncTaskManager"] = None,
) -> None:
    """Start the dependency-injected task manager."""

    manager = _resolve_task_manager(container=container, task_manager=task_manager)
    await manager.start()


async def stop_task_manager(
    *,
    container: Optional["ServiceContainer"] = None,
    task_manager: Optional["AsyncTaskManager"] = None,
) -> None:
    """Stop the dependency-injected task manager."""

    manager = _resolve_task_manager(container=container, task_manager=task_manager)
    await manager.stop()


def create_fire_and_forget_task(
    coro: Awaitable[T],
    *,
    name: str | None = None,
    creator: str | None = None,
    container: Optional["ServiceContainer"] = None,
    task_manager: Optional["AsyncTaskManager"] = None,
) -> None:
    """Create a fire-and-forget managed task from a synchronous context.

    This function schedules a coroutine to run as a managed task without
    needing to await it. Useful for calling from synchronous functions
    that need to schedule background work.

    Args:
        coro: Coroutine to execute
        name: Optional name for the task
        creator: Optional creator identifier
        container: Optional service container to resolve the task manager
        task_manager: Optional explicit task manager instance
    """

    manager = _resolve_task_manager(container=container, task_manager=task_manager)

    if not manager.is_started:
        # If task manager isn't started, fall back to regular asyncio.create_task
        logger.warning(
            "task_manager_not_started_fire_and_forget",
            name=name,
            creator=creator,
        )
        asyncio.create_task(coro, name=name)  # type: ignore[arg-type]
        return

    # Schedule the task creation as a fire-and-forget operation
    async def _create_managed_task() -> None:
        try:
            await manager.create_task(coro, name=name, creator=creator)
        except Exception as e:
            logger.error(
                "fire_and_forget_task_creation_failed",
                name=name,
                creator=creator,
                error=str(e),
                exc_info=True,
            )

    # Use asyncio.create_task to schedule the managed task creation
    asyncio.create_task(_create_managed_task(), name=f"create_{name or 'unnamed'}")
