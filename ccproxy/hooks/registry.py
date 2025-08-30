"""Central registry for all hooks"""

from collections import defaultdict

import structlog
from sortedcontainers import SortedList

from .base import Hook
from .events import HookEvent


class HookRegistry:
    """Central registry for all hooks with priority-based ordering."""

    def __init__(self) -> None:
        # Use SortedList for automatic priority ordering
        # Key function sorts by (priority, registration_order)
        self._hooks: dict[HookEvent, SortedList[Hook]] = defaultdict(
            lambda: SortedList(
                key=lambda h: (
                    getattr(h, "priority", 500),
                    self._registration_order.get(h, 0),
                )
            )
        )
        self._registration_order: dict[Hook, int] = {}
        self._next_order = 0
        self._logger = structlog.get_logger(__name__)

    def register(self, hook: Hook) -> None:
        """Register a hook for its events with priority ordering"""
        priority = getattr(
            hook, "priority", 500
        )  # Default priority for backward compatibility

        # Track registration order for stable sorting
        if hook not in self._registration_order:
            self._registration_order[hook] = self._next_order
            self._next_order += 1

        for event in hook.events:
            self._hooks[event].add(hook)
            self._logger.info(
                "hook_registered",
                name=hook.name,
                hook_event=event.value if hasattr(event, "value") else str(event),
                priority=priority,
            )

    def unregister(self, hook: Hook) -> None:
        """Remove a hook from all events"""
        for event in hook.events:
            try:
                self._hooks[event].remove(hook)
                self._logger.info(
                    "hook_unregistered",
                    name=hook.name,
                    hook_event=event.value if hasattr(event, "value") else str(event),
                )
            except ValueError:
                pass  # Hook not in list, ignore

        # Clean up registration order tracking
        if hook in self._registration_order:
            del self._registration_order[hook]

    def get_hooks(self, event: HookEvent) -> list[Hook]:
        """Get all hooks for an event in priority order"""
        return list(self._hooks.get(event, []))

    def get_hooks_summary(self) -> dict[str, list[dict[str, any]]]:
        """Get summary of all registered hooks organized by event.

        Returns:
            Dictionary mapping event names to lists of hook info
        """
        summary = {}
        for event, hooks in self._hooks.items():
            event_name = event.value if hasattr(event, "value") else str(event)
            summary[event_name] = [
                {
                    "name": hook.name,
                    "priority": getattr(hook, "priority", 500),
                }
                for hook in hooks
            ]
        return summary
