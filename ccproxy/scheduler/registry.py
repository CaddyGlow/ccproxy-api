"""Task registry for dynamic task registration and discovery."""

from __future__ import annotations

from typing import Any

import structlog

from .errors import TaskRegistrationError
from .tasks import BaseScheduledTask


logger = structlog.get_logger(__name__)


class TaskRegistry:
    """
    Registry for managing scheduled task registration and discovery.

    Provides a centralized way to register and retrieve scheduled tasks,
    enabling dynamic task management and configuration.
    """

    def __init__(self) -> None:
        """Initialize the task registry."""
        self._tasks: dict[str, type[BaseScheduledTask]] = {}

    def register(self, name: str, task_class: type[BaseScheduledTask]) -> None:
        """
        Register a scheduled task class.

        Args:
            name: Unique name for the task
            task_class: Task class that inherits from BaseScheduledTask

        Raises:
            TaskRegistrationError: If task name is already registered or invalid
        """
        if name in self._tasks:
            raise TaskRegistrationError(f"Task '{name}' is already registered")

        if not issubclass(task_class, BaseScheduledTask):
            raise TaskRegistrationError(
                f"Task class for '{name}' must inherit from BaseScheduledTask"
            )

        self._tasks[name] = task_class
        logger.debug("task_registered", task_name=name, task_class=task_class.__name__)

    def unregister(self, name: str) -> None:
        """
        Unregister a scheduled task.

        Args:
            name: Name of the task to unregister

        Raises:
            TaskRegistrationError: If task is not registered
        """
        if name not in self._tasks:
            raise TaskRegistrationError(f"Task '{name}' is not registered")

        del self._tasks[name]
        logger.debug("task_unregistered", task_name=name)

    def get(self, name: str) -> type[BaseScheduledTask]:
        """
        Get a registered task class by name.

        Args:
            name: Name of the task to retrieve

        Returns:
            Task class

        Raises:
            TaskRegistrationError: If task is not registered
        """
        if name not in self._tasks:
            raise TaskRegistrationError(f"Task '{name}' is not registered")

        return self._tasks[name]

    def list(self) -> list[str]:
        """
        Get list of all registered task names.

        Returns:
            List of registered task names
        """
        return list(self._tasks.keys())

    def has(self, name: str) -> bool:
        """
        Check if a task is registered.

        Args:
            name: Task name to check

        Returns:
            True if task is registered, False otherwise
        """
        return name in self._tasks

    def clear(self) -> None:
        """Clear all registered tasks."""
        self._tasks.clear()
        logger.debug("task_registry_cleared")

    def info(self) -> dict[str, Any]:
        """
        Get information about the current registry state.

        Returns:
            Dictionary with registry information
        """
        return {
            "total_tasks": len(self._tasks),
            "registered_tasks": list(self._tasks.keys()),
            "task_classes": {name: cls.__name__ for name, cls in self._tasks.items()},
        }


# Module-level accessors intentionally omitted.
