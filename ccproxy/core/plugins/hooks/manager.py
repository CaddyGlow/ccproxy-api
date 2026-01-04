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
from .thread_manager import BackgroundHookThreadManager


class HookManager:
    """Manages hook execution with error isolation and async/sync support.

    The HookManager is responsible for emitting events to registered hooks
    and ensuring that hook failures don't crash the system. It handles both
    async and sync hooks by running sync hooks in a thread pool.
    """

    def __init__(
        self,
        registry: HookRegistry,
        background_manager: BackgroundHookThreadManager | None = None,
    ):
        """Initialize the hook manager.

        Args:
            registry: The hook registry to get hooks from
            background_manager: Optional background thread manager for fire-and-forget execution
        """
        self._registry = registry
        self._background_manager = background_manager
        self._logger = structlog.get_logger(__name__)

    async def emit(
        self,
        event: HookEvent,
        data: dict[str, Any] | None = None,
        fire_and_forget: bool = True,
        **kwargs: Any,
    ) -> None:
        """Emit an event to all registered hooks.

        Creates a HookContext with the provided data and emits it to all
        hooks registered for the given event. Handles errors gracefully
        to ensure one failing hook doesn't affect others.

        Args:
            event: The event to emit
            data: Optional data dictionary to include in context
            fire_and_forget: If True, execute hooks in background thread (default)
            **kwargs: Additional context fields (request, response, provider, etc.)
        """
        context = HookContext(
            event=event,
            timestamp=datetime.utcnow(),
            data=data or {},
            metadata={},
            **kwargs,
        )

        if fire_and_forget and self._background_manager:
            # Execute in background thread - non-blocking
            self._background_manager.emit_async(context, self._registry)
            return
        elif fire_and_forget and not self._background_manager:
            # No background manager available, log warning and fall back to sync
            self._logger.warning(
                "fire_and_forget_requested_but_no_background_manager_available"
            )
        # Fall through to synchronous execution

        # Synchronous execution (legacy behavior)
        hooks = self._registry.get(event)
        if not hooks:
            return

        # Log execution order if debug logging enabled
        self._logger.debug(
            "hook_execution_order",
            hook_event=event.value if hasattr(event, "value") else str(event),
            hooks=[
                {"name": h.name, "priority": getattr(h, "priority", 500)} for h in hooks
            ],
        )

        # Execute all hooks in priority order, catching errors
        for hook in hooks:
            try:
                await self._execute_hook(hook, context)
            except Exception as e:
                self._logger.error(
                    "hook_execution_failed",
                    hook=hook.name,
                    hook_event=event.value if hasattr(event, "value") else str(event),
                    priority=getattr(hook, "priority", 500),
                    error=str(e),
                )
                # Continue executing other hooks

    async def emit_with_context(
        self, context: HookContext, fire_and_forget: bool = True
    ) -> None:
        """Emit an event using a pre-built HookContext.

        This is useful when you need to build the context with specific metadata
        before emitting the event.

        Args:
            context: The HookContext to emit
            fire_and_forget: If True, execute hooks in background thread (default)
        """
        if fire_and_forget and self._background_manager:
            # Execute in background thread - non-blocking
            self._background_manager.emit_async(context, self._registry)
            return
        elif fire_and_forget and not self._background_manager:
            # No background manager available, log warning and fall back to sync
            self._logger.warning(
                "fire_and_forget_requested_but_no_background_manager_available"
            )
        # Fall through to synchronous execution

        # Synchronous execution (legacy behavior)
        hooks = self._registry.get(context.event)
        if not hooks:
            return

        # Log execution order if debug logging enabled
        self._logger.debug(
            "hook_execution_order",
            hook_event=context.event.value
            if hasattr(context.event, "value")
            else str(context.event),
            hooks=[
                {"name": h.name, "priority": getattr(h, "priority", 500)} for h in hooks
            ],
        )

        # Execute all hooks in priority order, catching errors
        for hook in hooks:
            try:
                await self._execute_hook(hook, context)
            except Exception as e:
                self._logger.error(
                    "hook_execution_failed",
                    hook=hook.name,
                    hook_event=context.event.value
                    if hasattr(context.event, "value")
                    else str(context.event),
                    priority=getattr(hook, "priority", 500),
                    error=str(e),
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

    def shutdown(self) -> None:
        """Shutdown the background hook processing.

        This method should be called during application shutdown to ensure
        proper cleanup of the background thread.
        """
        if self._background_manager:
            self._background_manager.stop()
