"""
Scheduler system for periodic tasks.

This module provides a generic, extensible scheduler for managing periodic tasks
in the Claude Code Proxy API. It provides a centralized system that supports:

- Generic task scheduling with configurable intervals
- Task registration and discovery via registry pattern
- Graceful startup and shutdown with FastAPI integration
- Error handling with exponential backoff
- Structured logging and monitoring

Key components:
- Scheduler: Core scheduler engine for task management
- BaseScheduledTask: Abstract base class for all scheduled tasks
- TaskRegistry: Dynamic task registration and discovery system
"""

from .core import Scheduler
from .registry import TaskRegistry, register_task
from .tasks import (
    BaseScheduledTask,
    PricingCacheUpdateTask,
    PushgatewayTask,
    StatsPrintingTask,
)


# Task registration is now handled in manager.py during scheduler startup
# to avoid side effects during module imports (e.g., CLI help display)

__all__ = [
    "Scheduler",
    "TaskRegistry",
    "BaseScheduledTask",
    "PushgatewayTask",
    "StatsPrintingTask",
    "PricingCacheUpdateTask",
]
