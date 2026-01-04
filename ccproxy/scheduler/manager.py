"""Scheduler management for FastAPI integration."""

import structlog

from ccproxy.config.settings import Settings
from ccproxy.services.container import ServiceContainer

from .core import Scheduler
from .errors import SchedulerError, TaskRegistrationError
from .registry import TaskRegistry
from .tasks import PoolStatsTask, VersionUpdateCheckTask


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
        logger.debug("scheduler_disabled")
        return

    # Log network features status
    logger.debug(
        "network_features_status",
        pricing_updates_enabled=scheduler_config.pricing_update_enabled,
        version_check_enabled=scheduler_config.version_check_enabled,
        message=(
            "Network features disabled by default for privacy"
            if not scheduler_config.pricing_update_enabled
            and not scheduler_config.version_check_enabled
            else "Some network features are enabled"
        ),
    )

    if (
        hasattr(scheduler_config, "stats_printing_enabled")
        and scheduler_config.stats_printing_enabled
    ):
        logger.debug(
            "stats_printing_task_skipped",
            message="Stats printing is handled by plugin",
        )

    if scheduler_config.pricing_update_enabled:
        logger.debug(
            "pricing_update_task_handled_by_plugin",
            message="Pricing updates managed by plugin",
            interval_hours=scheduler_config.pricing_update_interval_hours,
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
                version_check_cache_ttl_hours=scheduler_config.version_check_cache_ttl_hours,
            )
            logger.debug(
                "version_check_task_added",
                interval_hours=scheduler_config.version_check_interval_hours,
                version_check_cache_ttl_hours=scheduler_config.version_check_cache_ttl_hours,
            )
        except TaskRegistrationError as e:
            logger.error(
                "version_check_task_registration_failed",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=e,
            )
        except Exception as e:
            logger.error(
                "version_check_task_add_failed",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=e,
            )


def _register_default_tasks(registry: TaskRegistry, settings: Settings) -> None:
    """Register default task types in the global registry based on configuration."""
    # Registry is provided by DI

    # Always register core tasks (not metrics-related)
    if not registry.has("version_update_check"):
        registry.register("version_update_check", VersionUpdateCheckTask)
    if not registry.has("pool_stats"):
        registry.register("pool_stats", PoolStatsTask)


async def start_scheduler(
    settings: Settings, container: ServiceContainer
) -> Scheduler | None:
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

        # Resolve registry from DI and register task types
        registry = container.get_task_registry()
        _register_default_tasks(registry, settings)

        # Create scheduler with settings
        scheduler = Scheduler(
            max_concurrent_tasks=settings.scheduler.max_concurrent_tasks,
            graceful_shutdown_timeout=settings.scheduler.graceful_shutdown_timeout,
            task_registry=registry,
        )

        # Start the scheduler
        await scheduler.start()

        # Setup tasks based on configuration
        await setup_scheduler_tasks(scheduler, settings)

        task_names = scheduler.list_tasks()
        logger.debug(
            "scheduler_started",
            max_concurrent_tasks=settings.scheduler.max_concurrent_tasks,
            active_tasks=scheduler.task_count,
            running_tasks=len(
                [name for name in task_names if scheduler.get_task(name).is_running]
            ),
            names=task_names,
        )

        return scheduler

    except SchedulerError as e:
        logger.error(
            "scheduler_start_scheduler_error",
            error=str(e),
            error_type=type(e).__name__,
            exc_info=e,
        )
        return None
    except Exception as e:
        logger.error(
            "scheduler_start_failed",
            error=str(e),
            error_type=type(e).__name__,
            exc_info=e,
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
    except SchedulerError as e:
        logger.error(
            "scheduler_stop_scheduler_error",
            error=str(e),
            error_type=type(e).__name__,
            exc_info=e,
        )
    except Exception as e:
        logger.error(
            "scheduler_stop_failed",
            error=str(e),
            error_type=type(e).__name__,
            exc_info=e,
        )
