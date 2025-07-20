"""Scheduler-specific exceptions."""


class SchedulerError(Exception):
    """Base exception for scheduler-related errors."""

    pass


class TaskRegistrationError(SchedulerError):
    """Raised when task registration fails."""

    pass


class TaskNotFoundError(SchedulerError):
    """Raised when attempting to access a task that doesn't exist."""

    pass


class TaskExecutionError(SchedulerError):
    """Raised when task execution encounters an error."""

    def __init__(self, task_name: str, original_error: Exception):
        self.task_name = task_name
        self.original_error = original_error
        super().__init__(f"Task '{task_name}' execution failed: {original_error}")


class SchedulerShutdownError(SchedulerError):
    """Raised when scheduler shutdown encounters an error."""

    pass
