"""
Scheduler system for periodic tasks.

This module provides a generic, extensible scheduler for managing periodic tasks
in the CCProxy API. It provides a centralized system that supports:

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
from .registry import TaskRegistry
from .tasks import BaseScheduledTask


__all__ = [
    "Scheduler",
    "TaskRegistry",
    "BaseScheduledTask",
]
