"""Central registry for all hooks"""

from collections import defaultdict
from typing import Any

import structlog
from sortedcontainers import SortedList  # type: ignore[import-untyped]

from .base import Hook
from .events import HookEvent


class HookRegistry:
    """Central registry for all hooks with priority-based ordering."""

    def __init__(self) -> None:
        # Use SortedList for automatic priority ordering
        # Key function sorts by (priority, registration_order)
        self._hooks: dict[HookEvent, Any] = defaultdict(
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
        # Batch logging for registration/unregistration
        self._pending_registrations: list[tuple[str, str, int]] = []
        self._pending_unregistrations: list[tuple[str, str]] = []

    def register(self, hook: Hook) -> None:
        """Register a hook for its events with priority ordering"""
        priority = getattr(
            hook, "priority", 500
        )  # Default priority for backward compatibility

        # Track registration order for stable sorting
        if hook not in self._registration_order:
            self._registration_order[hook] = self._next_order
            self._next_order += 1

        events_registered = []
        for event in hook.events:
            self._hooks[event].add(hook)
            event_name = event.value if hasattr(event, "value") else str(event)
            events_registered.append(event_name)
            # Log individual registrations at DEBUG level
            # self._logger.debug(
            #     "hook_registered",
            #     name=hook.name,
            #     hook_event=event_name,
            #     priority=priority,
            # )

        # Log summary at DEBUG; a global summary will be logged elsewhere at INFO
        if len(events_registered) > 0:
            self._logger.debug(
                "hook_registered",
                name=hook.name,
                events=events_registered,
                event_count=len(events_registered),
                priority=priority,
            )

    def unregister(self, hook: Hook) -> None:
        """Remove a hook from all events"""
        events_unregistered = []
        for event in hook.events:
            try:
                self._hooks[event].remove(hook)
                event_name = event.value if hasattr(event, "value") else str(event)
                events_unregistered.append(event_name)
                # Log individual unregistrations at DEBUG level
                # self._logger.debug(
                #     "hook_unregistered",
                #     name=hook.name,
                #     hook_event=event_name,
                # )
            except ValueError:
                pass  # Hook not in list, ignore

        # Log summary at INFO level only if multiple events
        if len(events_unregistered) > 1:
            self._logger.info(
                "hook_unregistered_summary",
                name=hook.name,
                events=events_unregistered,
                event_count=len(events_unregistered),
            )
        elif events_unregistered:
            # Single event - log at DEBUG level to reduce verbosity
            self._logger.debug(
                "hook_unregistered_single",
                name=hook.name,
                hook_event=events_unregistered[0],
            )

        # Clean up registration order tracking
        if hook in self._registration_order:
            del self._registration_order[hook]

    def get(self, event: HookEvent) -> list[Hook]:
        """Get all hooks for an event in priority order"""
        return list(self._hooks.get(event, []))

    def list(self) -> dict[str, list[dict[str, Any]]]:
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

    def has(self, event: HookEvent) -> bool:
        """Check if any hook is registered for the event."""
        hooks = self._hooks.get(event)
        return bool(hooks and len(hooks) > 0)

    def clear(self) -> None:
        """Clear all registered hooks and reset ordering (testing or shutdown)."""
        self._hooks.clear()
        self._registration_order.clear()
        self._next_order = 0


# Module-level accessor intentionally omitted.
