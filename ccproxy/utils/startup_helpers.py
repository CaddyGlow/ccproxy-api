"""Startup utility functions for application lifecycle management.

This module contains simple utility functions to extract and organize
the complex startup logic from the main lifespan function, following
the KISS principle and avoiding overengineering.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from fastapi import FastAPI

from ccproxy.scheduler.errors import SchedulerError
from ccproxy.scheduler.manager import start_scheduler, stop_scheduler


# DuckDB storage initialization is handled by the duckdb_storage plugin.


# get_permission_service is imported locally to avoid circular imports

if TYPE_CHECKING:
    from ccproxy.config.settings import Settings

logger = structlog.get_logger(__name__)


async def check_version_updates_startup(app: FastAPI, settings: Settings) -> None:
    """Trigger version update check at startup.

    Manually runs the version check task once during application startup,
    before the scheduler starts managing periodic checks.

    Args:
        app: FastAPI application instance
        settings: Application settings
    """
    # Skip version check if disabled by settings
    if not settings.scheduler.version_check_enabled:
        logger.debug("version_check_startup_disabled")
        return

    try:
        # Import locally to avoid circular imports and create task instance
        from ccproxy.scheduler.tasks import VersionUpdateCheckTask

        # Create a temporary task instance for startup check
        version_task = VersionUpdateCheckTask(
            name="version_check_startup",
            interval_seconds=settings.scheduler.version_check_interval_hours * 3600,
            enabled=True,
            version_check_cache_ttl_hours=settings.scheduler.version_check_cache_ttl_hours,
            skip_first_scheduled_run=False,
        )

        # Run the version check once and wait for it to complete
        success = await version_task.run()

        if success:
            logger.debug("version_check_startup_completed")
        else:
            logger.debug("version_check_startup_failed")

    except (ImportError, ModuleNotFoundError) as e:
        logger.debug(
            "version_check_startup_import_error",
            error=str(e),
            error_type=type(e).__name__,
        )
    except Exception as e:
        logger.debug(
            "version_check_startup_unexpected_error",
            error=str(e),
            error_type=type(e).__name__,
        )


async def check_claude_cli_startup(app: FastAPI, settings: Settings) -> None:
    """Check Claude CLI availability at startup.

    Args:
        app: FastAPI application instance
        settings: Application settings
    """
    # Claude CLI check is now handled by the plugin
    pass


# DuckDB storage startup/shutdown handled by plugin


async def setup_scheduler_startup(app: FastAPI, settings: Settings) -> None:
    """Start scheduler system and configure tasks.

    Args:
        app: FastAPI application instance
        settings: Application settings
    """
    try:
        # Use DI container to resolve registry and dependencies
        container = app.state.service_container
        scheduler = await start_scheduler(settings, container)
        app.state.scheduler = scheduler
        logger.debug("scheduler_initialized")

        # Add session pool stats task if session manager is available
        if (
            scheduler
            and hasattr(app.state, "session_manager")
            and app.state.session_manager
        ):
            try:
                # Add session pool stats task that runs every minute
                await scheduler.add_task(
                    task_name="session_pool_stats",
                    task_type="pool_stats",
                    interval_seconds=60,  # Every minute
                    enabled=True,
                    pool_manager=app.state.session_manager,
                )
                logger.debug("session_pool_stats_task_added", interval_seconds=60)
            except (ImportError, ModuleNotFoundError) as e:
                logger.error(
                    "session_pool_stats_task_add_import_error",
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=e,
                )
            except Exception as e:
                logger.error(
                    "session_pool_stats_task_add_unexpected_error",
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=e,
                )
    except SchedulerError as e:
        logger.error("scheduler_initialization_failed", error=str(e))
        # Continue startup even if scheduler fails (graceful degradation)


async def setup_scheduler_shutdown(app: FastAPI) -> None:
    """Stop scheduler system.

    Args:
        app: FastAPI application instance
    """
    try:
        scheduler = getattr(app.state, "scheduler", None)
        await stop_scheduler(scheduler)
        logger.debug("scheduler_stopped_lifespan")
    except SchedulerError as e:
        logger.error("scheduler_stop_failed", error=str(e))


async def setup_session_manager_shutdown(app: FastAPI) -> None:
    """Shutdown Claude SDK session manager if it was created.

    Args:
        app: FastAPI application instance
    """
    if hasattr(app.state, "session_manager") and app.state.session_manager:
        try:
            await app.state.session_manager.shutdown()
            logger.debug("claude_sdk_session_manager_shutdown")
        except (ImportError, ModuleNotFoundError) as e:
            logger.error(
                "claude_sdk_session_manager_shutdown_import_error",
                error=str(e),
                exc_info=e,
            )
        except Exception as e:
            logger.error(
                "claude_sdk_session_manager_shutdown_unexpected_error",
                error=str(e),
                exc_info=e,
            )


async def initialize_service_container_startup(
    app: FastAPI, settings: Settings
) -> None:
    """Initialize service container and proxy client.

    Args:
        app: FastAPI application instance
        settings: Application settings
    """
    try:
        # Create HTTP client for proxy
        from ccproxy.services.container import ServiceContainer

        # Reuse ServiceContainer from app state or create new one
        if hasattr(app.state, "service_container"):
            container = app.state.service_container
        else:
            logger.debug("creating_new_service_container")
            container = ServiceContainer(settings)
            app.state.service_container = container

        # Metrics are now handled by the metrics plugin
        app.state.metrics = None

        logger.debug("service_container_initialized")
    except (ImportError, ModuleNotFoundError) as e:
        logger.error(
            "service_container_initialization_import_error", error=str(e), exc_info=e
        )
    except Exception as e:
        logger.error(
            "service_container_initialization_unexpected_error",
            error=str(e),
            exc_info=e,
        )
        # Continue startup even if service container fails (graceful degradation)
