"""Scheduler management for FastAPI integration."""

import structlog

from ccproxy.config.settings import Settings

from .core import Scheduler
from .registry import register_task
from .tasks import (
    PricingCacheUpdateTask,
    PushgatewayTask,
    StatsPrintingTask,
    VersionUpdateCheckTask,
)


logger = structlog.get_logger(__name__)


async def setup_scheduler_tasks(scheduler: Scheduler, settings: Settings) -> None:
    """
    Setup and configure all scheduler tasks based on settings.

    Args:
        scheduler: Scheduler instance
        settings: Application settings
    """
    scheduler_config = settings.scheduler

    if not scheduler_config.enabled:
        logger.info("scheduler_disabled")
        return

    # Add pushgateway task if enabled
    if scheduler_config.pushgateway_enabled:
        try:
            await scheduler.add_task(
                task_name="pushgateway",
                task_type="pushgateway",
                interval_seconds=scheduler_config.pushgateway_interval_seconds,
                enabled=True,
                max_backoff_seconds=scheduler_config.pushgateway_max_backoff_seconds,
            )
            logger.info(
                "pushgateway_task_added",
                interval_seconds=scheduler_config.pushgateway_interval_seconds,
            )
        except Exception as e:
            logger.error(
                "pushgateway_task_add_failed",
                error=str(e),
                error_type=type(e).__name__,
            )

    # Add stats printing task if enabled
    if scheduler_config.stats_printing_enabled:
        try:
            await scheduler.add_task(
                task_name="stats_printing",
                task_type="stats_printing",
                interval_seconds=scheduler_config.stats_printing_interval_seconds,
                enabled=True,
            )
            logger.info(
                "stats_printing_task_added",
                interval_seconds=scheduler_config.stats_printing_interval_seconds,
            )
        except Exception as e:
            logger.error(
                "stats_printing_task_add_failed",
                error=str(e),
                error_type=type(e).__name__,
            )

    # Add pricing cache update task if enabled
    if scheduler_config.pricing_update_enabled:
        try:
            # Convert hours to seconds
            interval_seconds = scheduler_config.pricing_update_interval_hours * 3600

            await scheduler.add_task(
                task_name="pricing_cache_update",
                task_type="pricing_cache_update",
                interval_seconds=interval_seconds,
                enabled=True,
                force_refresh_on_startup=scheduler_config.pricing_force_refresh_on_startup,
            )
            logger.debug(
                "pricing_update_task_added",
                interval_hours=scheduler_config.pricing_update_interval_hours,
                force_refresh_on_startup=scheduler_config.pricing_force_refresh_on_startup,
            )
        except Exception as e:
            logger.error(
                "pricing_update_task_add_failed",
                error=str(e),
                error_type=type(e).__name__,
            )

    # Add version update check task if enabled
    if scheduler_config.version_check_enabled:
        try:
            # Convert hours to seconds
            interval_seconds = scheduler_config.version_check_interval_hours * 3600

            await scheduler.add_task(
                task_name="version_update_check",
                task_type="version_update_check",
                interval_seconds=interval_seconds,
                enabled=True,
                startup_max_age_hours=scheduler_config.version_check_startup_max_age_hours,
            )
            logger.debug(
                "version_check_task_added",
                interval_hours=scheduler_config.version_check_interval_hours,
                startup_max_age_hours=scheduler_config.version_check_startup_max_age_hours,
            )
        except Exception as e:
            logger.error(
                "version_check_task_add_failed",
                error=str(e),
                error_type=type(e).__name__,
            )


def _register_default_tasks() -> None:
    """Register default task types in the global registry."""
    from .registry import get_task_registry

    registry = get_task_registry()

    # Only register if not already registered
    if not registry.is_registered("pushgateway"):
        register_task("pushgateway", PushgatewayTask)
    if not registry.is_registered("stats_printing"):
        register_task("stats_printing", StatsPrintingTask)
    if not registry.is_registered("pricing_cache_update"):
        register_task("pricing_cache_update", PricingCacheUpdateTask)
    if not registry.is_registered("version_update_check"):
        register_task("version_update_check", VersionUpdateCheckTask)


async def start_scheduler(settings: Settings) -> Scheduler | None:
    """
    Start the scheduler with configured tasks.

    Args:
        settings: Application settings

    Returns:
        Scheduler instance if successful, None otherwise
    """
    try:
        if not settings.scheduler.enabled:
            logger.info("scheduler_disabled")
            return None

        # Register task types (only when actually starting scheduler)
        _register_default_tasks()

        # Create scheduler with settings
        scheduler = Scheduler(
            max_concurrent_tasks=settings.scheduler.max_concurrent_tasks,
            graceful_shutdown_timeout=settings.scheduler.graceful_shutdown_timeout,
        )

        # Start the scheduler
        await scheduler.start()

        # Setup tasks based on configuration
        await setup_scheduler_tasks(scheduler, settings)

        logger.info(
            "scheduler_started",
            max_concurrent_tasks=settings.scheduler.max_concurrent_tasks,
            active_tasks=scheduler.task_count,
            running_tasks=len(
                [
                    name
                    for name in scheduler.list_tasks()
                    if scheduler.get_task(name).is_running
                ]
            ),
        )

        return scheduler

    except Exception as e:
        logger.error(
            "scheduler_start_failed",
            error=str(e),
            error_type=type(e).__name__,
        )
        return None


async def stop_scheduler(scheduler: Scheduler | None) -> None:
    """
    Stop the scheduler gracefully.

    Args:
        scheduler: Scheduler instance to stop
    """
    if scheduler is None:
        return

    try:
        await scheduler.stop()
    except Exception as e:
        logger.error(
            "scheduler_stop_failed",
            error=str(e),
            error_type=type(e).__name__,
        )
