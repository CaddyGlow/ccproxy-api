"""Central registry for all hooks"""

from collections import defaultdict

import structlog

from .base import Hook
from .events import HookEvent


class HookRegistry:
    """Central registry for all hooks"""

    def __init__(self) -> None:
        self._hooks: dict[HookEvent, list[Hook]] = defaultdict(list)
        self._logger = structlog.get_logger(__name__)

    def register(self, hook: Hook) -> None:
        """Register a hook for its events"""
        for event in hook.events:
            self._hooks[event].append(hook)
            self._logger.info(f"Registered {hook.name} for {event}")

    def unregister(self, hook: Hook) -> None:
        """Remove a hook from all events"""
        for event in hook.events:
            if hook in self._hooks[event]:
                self._hooks[event].remove(hook)

    def get_hooks(self, event: HookEvent) -> list[Hook]:
        """Get all hooks for an event"""
        return self._hooks.get(event, [])
