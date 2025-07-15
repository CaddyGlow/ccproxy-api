"""
Scheduler for periodic observability tasks.

This module provides a background scheduler for periodic tasks like:
- Pushing metrics to Pushgateway
- Flushing metrics to storage
- Cleaning up old metrics data

Key features:
- Async task scheduling with configurable intervals
- Graceful shutdown handling
- Error handling with backoff
- Integration with observability settings
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any, Optional

import structlog

from ccproxy.config.observability import ObservabilitySettings


logger = structlog.get_logger(__name__)


class ObservabilityScheduler:
    """
    Background scheduler for periodic observability tasks.

    Manages periodic tasks like pushing metrics to Pushgateway,
    flushing metrics to storage, and other maintenance tasks.
    """

    def __init__(self, settings: ObservabilitySettings):
        """
        Initialize scheduler with observability settings.

        Args:
            settings: Observability configuration settings
        """
        self.settings = settings
        self._running = False
        self._tasks: list[asyncio.Task[Any]] = []
        self._pushgateway_interval = 30.0  # seconds
        self._metrics_instance: Any | None = None

    async def start(self) -> None:
        """Start the scheduler and background tasks."""
        if self._running:
            return

        self._running = True
        logger.info("observability_scheduler_start")

        # Initialize metrics instance
        await self._init_metrics()

        # Start periodic tasks
        if self.settings.pushgateway_enabled and self.settings.pushgateway_url:
            task = asyncio.create_task(self._pushgateway_task())
            self._tasks.append(task)

    async def stop(self) -> None:
        """Stop the scheduler and cancel all tasks."""
        if not self._running:
            return

        self._running = False
        logger.info("observability_scheduler_stop")

        # Cancel all tasks
        for task in self._tasks:
            task.cancel()

        # Wait for tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        self._tasks.clear()

    async def _init_metrics(self) -> None:
        """Initialize metrics instance."""
        try:
            from .metrics import get_metrics

            self._metrics_instance = get_metrics()
        except Exception as e:
            logger.error("scheduler_metrics_init_failed", error=str(e))

    async def _pushgateway_task(self) -> None:
        """Periodic task to push metrics to Pushgateway."""
        logger.info(
            "pushgateway_task_start",
            interval=self._pushgateway_interval,
            url=self.settings.pushgateway_url,
            job=self.settings.pushgateway_job,
        )

        while self._running:
            try:
                if (
                    self._metrics_instance
                    and self._metrics_instance.is_pushgateway_enabled()
                ):
                    success = self._metrics_instance.push_to_gateway()
                    if not success:
                        logger.warning("pushgateway_push_failed")

                # Wait for next interval
                await asyncio.sleep(self._pushgateway_interval)

            except asyncio.CancelledError:
                logger.info("pushgateway_task_cancelled")
                break
            except Exception as e:
                logger.error("pushgateway_task_error", error=str(e))
                # Backoff on error
                backoff_time = min(self._pushgateway_interval * 2, 60.0)
                await asyncio.sleep(backoff_time)

    def set_pushgateway_interval(self, interval: float) -> None:
        """
        Set the interval for pushing metrics to Pushgateway.

        Args:
            interval: Interval in seconds
        """
        self._pushgateway_interval = max(1.0, interval)
        logger.info("pushgateway_interval_updated", interval=self._pushgateway_interval)


# Global scheduler instance
_global_scheduler: ObservabilityScheduler | None = None


async def get_scheduler() -> ObservabilityScheduler:
    """Get or create global scheduler instance."""
    global _global_scheduler

    if _global_scheduler is None:
        # Import here to avoid circular imports
        from ccproxy.config.settings import get_settings

        settings = get_settings()
        _global_scheduler = ObservabilityScheduler(settings.observability)
        await _global_scheduler.start()

    return _global_scheduler


async def start_scheduler() -> None:
    """Start the global observability scheduler."""
    scheduler = await get_scheduler()
    await scheduler.start()


async def stop_scheduler() -> None:
    """Stop the global observability scheduler."""
    global _global_scheduler

    if _global_scheduler:
        await _global_scheduler.stop()
        _global_scheduler = None


@asynccontextmanager
async def scheduler_context() -> Any:
    """
    Context manager for scheduler lifecycle management.

    Yields:
        ObservabilityScheduler instance
    """
    scheduler = await get_scheduler()
    await scheduler.start()

    try:
        yield scheduler
    finally:
        await scheduler.stop()
