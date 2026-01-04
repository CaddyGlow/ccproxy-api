"""Pricing plugin scheduled tasks."""

import asyncio
import contextlib
import random
import time
from abc import ABC, abstractmethod
from typing import Any

from ccproxy.core.async_task_manager import create_managed_task
from ccproxy.core.logging import get_plugin_logger

from .service import PricingService


logger = get_plugin_logger(__name__)


class BaseScheduledTask(ABC):
    """
    Abstract base class for all scheduled tasks.

    Provides common functionality for task lifecycle management, error handling,
    and exponential backoff for failed executions.
    """

    def __init__(
        self,
        name: str,
        interval_seconds: float,
        enabled: bool = True,
        max_backoff_seconds: float = 300.0,
        jitter_factor: float = 0.25,
    ):
        """
        Initialize scheduled task.

        Args:
            name: Human-readable task name
            interval_seconds: Interval between task executions in seconds
            enabled: Whether the task is enabled
            max_backoff_seconds: Maximum backoff delay for failed tasks
            jitter_factor: Jitter factor for backoff randomization (0.0-1.0)
        """
        self.name = name
        self.interval_seconds = max(1.0, interval_seconds)
        self.enabled = enabled
        self.max_backoff_seconds = max_backoff_seconds
        self.jitter_factor = min(1.0, max(0.0, jitter_factor))

        # Task state
        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()
        self._consecutive_failures = 0
        self._last_success_time: float | None = None
        self._next_run_time: float | None = None

    @abstractmethod
    async def run(self) -> bool:
        """
        Execute the task logic.

        Returns:
            True if task completed successfully, False otherwise
        """

    async def setup(self) -> None:  # noqa: B027
        """
        Optional setup hook called before the task starts running.

        Override this method to perform any initialization required by the task.
        """
        pass

    async def teardown(self) -> None:  # noqa: B027
        """
        Optional teardown hook called when the task stops.

        Override this method to perform any cleanup required by the task.
        """
        pass

    def _calculate_next_run_delay(self, failed: bool = False) -> float:
        """Calculate delay until next task execution with exponential backoff."""
        if not failed:
            # Normal interval with jitter
            base_delay = self.interval_seconds
            jitter = random.uniform(-self.jitter_factor, self.jitter_factor)
            return float(base_delay * (1 + jitter))

        # Exponential backoff for failures
        backoff_factor = min(2**self._consecutive_failures, 32)
        backoff_delay = min(
            self.interval_seconds * backoff_factor, self.max_backoff_seconds
        )

        # Add jitter to prevent thundering herd
        jitter = random.uniform(-self.jitter_factor, self.jitter_factor)
        return float(backoff_delay * (1 + jitter))

    async def _run_with_error_handling(self) -> bool:
        """Execute task with error handling and metrics."""
        start_time = time.time()

        try:
            success = await self.run()

            if success:
                self._consecutive_failures = 0
                self._last_success_time = start_time
                logger.debug(
                    "scheduled_task_success",
                    task_name=self.name,
                    duration=time.time() - start_time,
                )
            else:
                self._consecutive_failures += 1
                logger.warning(
                    "scheduled_task_failed",
                    task_name=self.name,
                    consecutive_failures=self._consecutive_failures,
                    duration=time.time() - start_time,
                )

            return success

        except Exception as e:
            self._consecutive_failures += 1
            logger.error(
                "scheduled_task_error",
                task_name=self.name,
                error=str(e),
                error_type=type(e).__name__,
                consecutive_failures=self._consecutive_failures,
                duration=time.time() - start_time,
                exc_info=e,
            )
            return False

    async def _task_loop(self) -> None:
        """Main task execution loop."""
        logger.info("scheduled_task_starting", task_name=self.name)

        try:
            # Run setup
            with contextlib.suppress(Exception):
                await self.setup()

            while not self._stop_event.is_set():
                # Execute task
                success = await self._run_with_error_handling()

                # Calculate next run delay
                delay = self._calculate_next_run_delay(failed=not success)
                self._next_run_time = time.time() + delay

                # Wait for next execution or stop event
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=delay)
                    break  # Stop event was set
                except TimeoutError:
                    continue  # Time to run again

        finally:
            # Run teardown
            with contextlib.suppress(Exception):
                await self.teardown()

            logger.info("scheduled_task_stopped", task_name=self.name)

    async def start(self) -> None:
        """Start the scheduled task."""
        if not self.enabled:
            logger.info("scheduled_task_disabled", task_name=self.name)
            return

        if self._task and not self._task.done():
            logger.warning("scheduled_task_already_running", task_name=self.name)
            return

        self._stop_event.clear()
        self._task = await create_managed_task(
            self._task_loop(), name=f"scheduled_task_{self.name}"
        )

    async def stop(self, timeout: float = 10.0) -> None:
        """Stop the scheduled task."""
        if not self._task:
            return

        logger.info("scheduled_task_stopping", task_name=self.name)

        # Signal stop
        self._stop_event.set()

        # Wait for task to complete
        try:
            await asyncio.wait_for(self._task, timeout=timeout)
        except TimeoutError:
            logger.warning(
                "scheduled_task_stop_timeout", task_name=self.name, timeout=timeout
            )
            if not self._task.done():
                self._task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._task

        self._task = None

    def is_running(self) -> bool:
        """Check if task is currently running."""
        return self._task is not None and not self._task.done()

    def get_status(self) -> dict[str, Any]:
        """Get current task status information."""
        now = time.time()
        return {
            "name": self.name,
            "enabled": self.enabled,
            "running": self.is_running(),
            "consecutive_failures": self._consecutive_failures,
            "last_success_time": self._last_success_time,
            "last_success_ago_seconds": (
                now - self._last_success_time if self._last_success_time else None
            ),
            "next_run_time": self._next_run_time,
            "next_run_in_seconds": (
                self._next_run_time - now if self._next_run_time else None
            ),
            "interval_seconds": self.interval_seconds,
        }


class PricingCacheUpdateTask(BaseScheduledTask):
    """Task for updating pricing cache periodically."""

    def __init__(
        self,
        name: str,
        interval_seconds: float,
        pricing_service: PricingService,
        enabled: bool = True,
        force_refresh_on_startup: bool = False,
    ):
        """
        Initialize pricing cache update task.

        Args:
            name: Task name
            interval_seconds: Interval between pricing updates
            pricing_service: Pricing service instance
            enabled: Whether task is enabled
            force_refresh_on_startup: Whether to force refresh on first run
        """
        super().__init__(
            name=name,
            interval_seconds=interval_seconds,
            enabled=enabled,
        )
        self.pricing_service = pricing_service
        self.force_refresh_on_startup = force_refresh_on_startup
        self._first_run = True

    async def run(self) -> bool:
        """Execute pricing cache update."""
        try:
            if not self.pricing_service.config.enabled:
                logger.debug("pricing_service_disabled", task_name=self.name)
                return True  # Not a failure, just disabled

            # Force refresh on first run if configured
            force_refresh = self._first_run and self.force_refresh_on_startup
            self._first_run = False

            if force_refresh:
                logger.info("pricing_update_force_refresh_startup", task_name=self.name)
                success = await self.pricing_service.force_refresh_pricing()
            else:
                # Regular update check
                pricing_data = await self.pricing_service.get_current_pricing(
                    force_refresh=False
                )
                success = pricing_data is not None

            if success:
                logger.debug("pricing_update_success", task_name=self.name)
            else:
                logger.warning("pricing_update_failed", task_name=self.name)

            return success

        except Exception as e:
            logger.error(
                "pricing_update_task_error",
                task_name=self.name,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=e,
            )
            return False
