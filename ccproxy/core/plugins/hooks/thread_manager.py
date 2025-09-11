"""Background thread manager for async hook execution."""

import asyncio
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog

from .base import Hook, HookContext


logger = structlog.get_logger(__name__)


@dataclass
class HookTask:
    """Represents a hook execution task."""

    context: HookContext
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)


class BackgroundHookThreadManager:
    """Manages a dedicated async thread for hook execution."""

    def __init__(self) -> None:
        """Initialize the background thread manager."""
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._queue: asyncio.Queue[tuple[HookTask, Any]] | None = None
        self._shutdown_event: asyncio.Event | None = None
        self._running = False
        self._logger = logger.bind(component="background_hook_thread")

    def start(self) -> None:
        """Start the background thread with its own event loop."""
        if self._running:
            return

        self._logger.info("starting_background_hook_thread")

        # Create and start the background thread
        self._thread = threading.Thread(
            target=self._run_background_loop, name="hook-background-thread", daemon=True
        )
        self._thread.start()

        # Wait a moment for the thread to initialize
        time.sleep(0.01)
        self._running = True

        self._logger.info("background_hook_thread_started")

    def stop(self, timeout: float = 5.0) -> None:
        """Gracefully shutdown the background thread."""
        if not self._running:
            return

        self._logger.info("stopping_background_hook_thread")

        # Signal shutdown to the background loop
        if self._loop and self._shutdown_event:
            self._loop.call_soon_threadsafe(self._shutdown_event.set)

        # Wait for thread to complete
        if self._thread:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                self._logger.warning("background_thread_shutdown_timeout")

        self._running = False
        self._loop = None
        self._thread = None
        self._queue = None
        self._shutdown_event = None

        self._logger.info("background_hook_thread_stopped")

    def emit_async(self, context: HookContext, registry: Any) -> None:
        """Queue a hook task for background execution.

        Args:
            context: Hook context to execute
            registry: Hook registry to get hooks from
        """
        if not self._running:
            self.start()

        if not self._loop or not self._queue:
            self._logger.warning("background_thread_not_ready_dropping_task")
            return

        task = HookTask(context=context)

        # Add task to queue in a thread-safe way
        try:
            self._loop.call_soon_threadsafe(self._add_task_to_queue, task, registry)
        except Exception as e:
            self._logger.error("failed_to_queue_hook_task", error=str(e))

    def _add_task_to_queue(self, task: HookTask, registry: Any) -> None:
        """Add task to queue (called from background thread)."""
        if self._queue:
            try:
                self._queue.put_nowait((task, registry))
            except asyncio.QueueFull:
                self._logger.warning("hook_task_queue_full_dropping_task")

    def _run_background_loop(self) -> None:
        """Run the background event loop for hook processing."""
        try:
            # Create new event loop for this thread
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            # Create queue and shutdown event
            self._queue = asyncio.Queue[tuple[HookTask, Any]](maxsize=1000)
            self._shutdown_event = asyncio.Event()

            # Run the processing loop
            self._loop.run_until_complete(self._process_tasks())
        except Exception as e:
            logger.error("background_hook_thread_error", error=str(e))
        finally:
            if self._loop:
                self._loop.close()

    async def _process_tasks(self) -> None:
        """Main task processing loop."""
        self._logger.debug("background_hook_processor_started")

        while self._shutdown_event and not self._shutdown_event.is_set():
            try:
                # Wait for either a task or shutdown signal
                if not self._queue:
                    break
                task_data = await asyncio.wait_for(self._queue.get(), timeout=0.1)

                task, registry = task_data
                await self._execute_task(task, registry)

            except TimeoutError:
                # Normal timeout, continue loop
                continue
            except Exception as e:
                self._logger.error("hook_task_processing_error", error=str(e))

        self._logger.debug("background_hook_processor_stopped")

    async def _execute_task(self, task: HookTask, registry: Any) -> None:
        """Execute a single hook task.

        Args:
            task: The hook task to execute
            registry: Hook registry to get hooks from
        """
        try:
            hooks = registry.get(task.context.event)
            if not hooks:
                return

            # Execute all hooks for this event
            for hook in hooks:
                try:
                    await self._execute_hook(hook, task.context)
                except Exception as e:
                    self._logger.error(
                        "background_hook_execution_failed",
                        hook=hook.name,
                        event=task.context.event.value
                        if hasattr(task.context.event, "value")
                        else str(task.context.event),
                        error=str(e),
                        task_id=task.task_id,
                    )
        except Exception as e:
            self._logger.error(
                "hook_task_execution_failed", error=str(e), task_id=task.task_id
            )

    async def _execute_hook(self, hook: Hook, context: HookContext) -> None:
        """Execute a single hook with proper async/sync handling.

        Args:
            hook: The hook to execute
            context: The context to pass to the hook
        """
        result = hook(context)
        if asyncio.iscoroutine(result):
            await result
        # If result is None, it was a sync hook and we're done
