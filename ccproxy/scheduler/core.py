"""Core scheduler for managing periodic tasks."""

import asyncio
from typing import Any

import structlog

from .exceptions import (
    SchedulerError,
    SchedulerShutdownError,
    TaskNotFoundError,
    TaskRegistrationError,
)
from .registry import TaskRegistry, get_task_registry
from .tasks import BaseScheduledTask


logger = structlog.get_logger(__name__)


class Scheduler:
    """
    Scheduler for managing multiple periodic tasks.

    Provides centralized management of scheduled tasks with:
    - Dynamic task registration and configuration
    - Graceful startup and shutdown
    - Task monitoring and status reporting
    - Error handling and recovery
    """

    def __init__(
        self,
        max_concurrent_tasks: int = 10,
        graceful_shutdown_timeout: float = 30.0,
        task_registry: TaskRegistry | None = None,
    ):
        """
        Initialize the scheduler.

        Args:
            max_concurrent_tasks: Maximum number of tasks to run concurrently
            graceful_shutdown_timeout: Timeout for graceful shutdown in seconds
            task_registry: Task registry instance (uses global if None)
        """
        self.max_concurrent_tasks = max_concurrent_tasks
        self.graceful_shutdown_timeout = graceful_shutdown_timeout
        self.task_registry = task_registry or get_task_registry()

        self._running = False
        self._tasks: dict[str, BaseScheduledTask] = {}
        self._semaphore: asyncio.Semaphore | None = None

    async def start(self) -> None:
        """Start the scheduler and all enabled tasks."""
        if self._running:
            logger.warning("scheduler_already_running")
            return

        self._running = True
        self._semaphore = asyncio.Semaphore(self.max_concurrent_tasks)

        logger.info(
            "scheduler_starting",
            max_concurrent_tasks=self.max_concurrent_tasks,
            registered_tasks=self.task_registry.list_tasks(),
        )

        try:
            # No automatic task creation - tasks must be explicitly added
            logger.info(
                "scheduler_started",
                active_tasks=len(self._tasks),
                running_tasks=[
                    name for name, task in self._tasks.items() if task.is_running
                ],
            )
        except Exception as e:
            self._running = False
            logger.error(
                "scheduler_start_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise SchedulerError(f"Failed to start scheduler: {e}") from e

    async def stop(self) -> None:
        """Stop the scheduler and all running tasks."""
        if not self._running:
            return

        self._running = False
        logger.info("scheduler_stopping", active_tasks=len(self._tasks))

        # Stop all tasks
        stop_tasks = []
        for task_name, task in self._tasks.items():
            if task.is_running:
                logger.debug("stopping_task", task_name=task_name)
                stop_tasks.append(task.stop())

        if stop_tasks:
            try:
                # Wait for all tasks to stop gracefully
                await asyncio.wait_for(
                    asyncio.gather(*stop_tasks, return_exceptions=True),
                    timeout=self.graceful_shutdown_timeout,
                )
                logger.info("scheduler_stopped_gracefully")
            except TimeoutError:
                logger.warning(
                    "scheduler_shutdown_timeout",
                    timeout=self.graceful_shutdown_timeout,
                )
                # Tasks should have cancelled themselves, but log the issue
                for task_name, task in self._tasks.items():
                    if task.is_running:
                        logger.warning(
                            "task_still_running_after_shutdown", task_name=task_name
                        )
            except Exception as e:
                logger.error(
                    "scheduler_shutdown_error",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise SchedulerShutdownError(
                    f"Error during scheduler shutdown: {e}"
                ) from e

        self._tasks.clear()
        logger.info("scheduler_stopped")

    async def add_task(
        self,
        task_name: str,
        task_type: str,
        **task_kwargs: Any,
    ) -> None:
        """
        Add and start a task.

        Args:
            task_name: Unique name for this task instance
            task_type: Type of task (must be registered in task registry)
            **task_kwargs: Additional arguments to pass to task constructor

        Raises:
            TaskRegistrationError: If task type is not registered
            SchedulerError: If task name already exists or task creation fails
        """
        if task_name in self._tasks:
            raise SchedulerError(f"Task '{task_name}' already exists")

        if not self.task_registry.is_registered(task_type):
            raise TaskRegistrationError(f"Task type '{task_type}' is not registered")

        try:
            # Get task class and create instance
            task_class = self.task_registry.get(task_type)
            task_instance = task_class(name=task_name, **task_kwargs)

            # Add to our tasks dict
            self._tasks[task_name] = task_instance

            # Start the task if scheduler is running and task is enabled
            if self._running and task_instance.enabled:
                await task_instance.start()
                logger.info(
                    "task_added_and_started",
                    task_name=task_name,
                    task_type=task_type,
                )
            else:
                logger.info(
                    "task_added_not_started",
                    task_name=task_name,
                    task_type=task_type,
                    scheduler_running=self._running,
                    task_enabled=task_instance.enabled,
                )

        except Exception as e:
            # Clean up if task was partially added
            if task_name in self._tasks:
                del self._tasks[task_name]

            logger.error(
                "task_add_failed",
                task_name=task_name,
                task_type=task_type,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise SchedulerError(f"Failed to add task '{task_name}': {e}") from e

    async def remove_task(self, task_name: str) -> None:
        """
        Remove and stop a task.

        Args:
            task_name: Name of task to remove

        Raises:
            TaskNotFoundError: If task does not exist
        """
        if task_name not in self._tasks:
            raise TaskNotFoundError(f"Task '{task_name}' does not exist")

        task = self._tasks[task_name]

        try:
            if task.is_running:
                await task.stop()

            del self._tasks[task_name]
            logger.info("task_removed", task_name=task_name)

        except Exception as e:
            logger.error(
                "task_remove_failed",
                task_name=task_name,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise SchedulerError(f"Failed to remove task '{task_name}': {e}") from e

    def get_task(self, task_name: str) -> BaseScheduledTask:
        """
        Get a task instance by name.

        Args:
            task_name: Name of task to retrieve

        Returns:
            Task instance

        Raises:
            TaskNotFoundError: If task does not exist
        """
        if task_name not in self._tasks:
            raise TaskNotFoundError(f"Task '{task_name}' does not exist")

        return self._tasks[task_name]

    def list_tasks(self) -> list[str]:
        """
        Get list of all task names.

        Returns:
            List of task names
        """
        return list(self._tasks.keys())

    def get_task_status(self, task_name: str) -> dict[str, Any]:
        """
        Get status information for a specific task.

        Args:
            task_name: Name of task

        Returns:
            Task status dictionary

        Raises:
            TaskNotFoundError: If task does not exist
        """
        if task_name not in self._tasks:
            raise TaskNotFoundError(f"Task '{task_name}' does not exist")

        return self._tasks[task_name].get_status()

    def get_scheduler_status(self) -> dict[str, Any]:
        """
        Get overall scheduler status information.

        Returns:
            Scheduler status dictionary
        """
        running_tasks = [name for name, task in self._tasks.items() if task.is_running]

        return {
            "running": self._running,
            "total_tasks": len(self._tasks),
            "running_tasks": len(running_tasks),
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "graceful_shutdown_timeout": self.graceful_shutdown_timeout,
            "task_names": list(self._tasks.keys()),
            "running_task_names": running_tasks,
            "registered_task_types": self.task_registry.list_tasks(),
        }

    @property
    def is_running(self) -> bool:
        """Check if the scheduler is running."""
        return self._running

    @property
    def task_count(self) -> int:
        """Get the number of managed tasks."""
        return len(self._tasks)


# Global scheduler instance
_global_scheduler: Scheduler | None = None


async def get_scheduler() -> Scheduler:
    """
    Get or create the global scheduler instance.

    Returns:
        Global Scheduler instance
    """
    global _global_scheduler

    if _global_scheduler is None:
        _global_scheduler = Scheduler()

    return _global_scheduler


async def start_scheduler() -> None:
    """Start the global scheduler."""
    scheduler = await get_scheduler()
    await scheduler.start()


async def stop_scheduler() -> None:
    """Stop the global scheduler."""
    global _global_scheduler

    if _global_scheduler:
        await _global_scheduler.stop()
        _global_scheduler = None
