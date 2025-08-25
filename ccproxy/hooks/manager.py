"""Hook execution manager for CCProxy.

This module provides the HookManager class which handles the execution of hooks
for various events in the system. It ensures proper error isolation and supports
both async and sync hooks.
"""

import asyncio
from datetime import datetime
from typing import Any

import structlog

from .base import Hook, HookContext
from .events import HookEvent
from .registry import HookRegistry


class HookManager:
    """Manages hook execution with error isolation and async/sync support.

    The HookManager is responsible for emitting events to registered hooks
    and ensuring that hook failures don't crash the system. It handles both
    async and sync hooks by running sync hooks in a thread pool.
    """

    def __init__(self, registry: HookRegistry):
        """Initialize the hook manager.

        Args:
            registry: The hook registry to get hooks from
        """
        self._registry = registry
        self._logger = structlog.get_logger(__name__)

    async def emit(
        self, event: HookEvent, data: dict[str, Any] | None = None, **kwargs: Any
    ) -> None:
        """Emit an event to all registered hooks.

        Creates a HookContext with the provided data and emits it to all
        hooks registered for the given event. Handles errors gracefully
        to ensure one failing hook doesn't affect others.

        Args:
            event: The event to emit
            data: Optional data dictionary to include in context
            **kwargs: Additional context fields (request, response, provider, etc.)
        """
        context = HookContext(
            event=event,
            timestamp=datetime.utcnow(),
            data=data or {},
            metadata={},
            **kwargs,
        )

        hooks = self._registry.get_hooks(event)
        if not hooks:
            return

        # Execute all hooks, catching errors
        for hook in hooks:
            try:
                await self._execute_hook(hook, context)
            except Exception as e:
                self._logger.error(f"Hook {hook.name} failed for {event}", error=str(e))
                # Continue executing other hooks

    async def emit_with_context(self, context: HookContext) -> None:
        """Emit an event using a pre-built HookContext.

        This is useful when you need to build the context with specific metadata
        before emitting the event.

        Args:
            context: The HookContext to emit
        """
        hooks = self._registry.get_hooks(context.event)
        if not hooks:
            return

        # Execute all hooks, catching errors
        for hook in hooks:
            try:
                await self._execute_hook(hook, context)
            except Exception as e:
                self._logger.error(
                    f"Hook {hook.name} failed for {context.event}", error=str(e)
                )
                # Continue executing other hooks

    async def _execute_hook(self, hook: Hook, context: HookContext) -> None:
        """Execute a single hook with proper async/sync handling.

        Determines if the hook is async or sync and executes it appropriately.
        Sync hooks are run in a thread pool to avoid blocking the async event loop.

        Args:
            hook: The hook to execute
            context: The context to pass to the hook
        """
        result = hook(context)
        if asyncio.iscoroutine(result):
            await result
        # If result is None, it was a sync hook and we're done
