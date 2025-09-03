"""Hook system for CCProxy.

This package provides a flexible, event-driven hook system that enables
metrics collection, analytics, logging, and custom provider behaviors
without modifying core code.

Key components:
- HookEvent: Enumeration of all supported events
- HookContext: Context data passed to hooks
- Hook: Protocol for hook implementations
- HookRegistry: Registry for managing hooks
- HookManager: Manager for executing hooks
"""

from .base import Hook, HookContext
from .events import HookEvent
from .manager import HookManager
from .registry import HookRegistry


__all__ = ["Hook", "HookContext", "HookEvent", "HookManager", "HookRegistry"]
