"""Base scheduled task classes and task implementations."""

import asyncio
import contextlib
import random
import time
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any

import structlog

from ccproxy.core.async_task_manager import create_managed_task
from ccproxy.scheduler.errors import SchedulerError
from ccproxy.utils.version_checker import (
    VersionCheckState,
    compare_versions,
    fetch_latest_github_version,
    get_current_version,
    get_version_check_state_path,
    load_check_state,
    save_check_state,
)


logger = structlog.get_logger(__name__)


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

        self._consecutive_failures = 0
        self._last_run_time: float = 0
        self._running = False
        self._task: asyncio.Task[Any] | None = None

    @abstractmethod
    async def run(self) -> bool:
        """
        Execute the scheduled task.

        Returns:
            True if execution was successful, False otherwise
        """
        pass

    async def setup(self) -> None:
        """
        Perform any setup required before task execution starts.

        Called once when the task is first started. Override if needed.
        Default implementation does nothing.
        """
        # Default implementation - subclasses can override if needed
        return

    async def cleanup(self) -> None:
        """
        Perform any cleanup required after task execution stops.

        Called once when the task is stopped. Override if needed.
        Default implementation does nothing.
        """
        # Default implementation - subclasses can override if needed
        return

    def calculate_next_delay(self) -> float:
        """
        Calculate the delay before the next task execution.

        Returns exponential backoff delay for failed tasks, or normal interval
        for successful tasks, with optional jitter.

        Returns:
            Delay in seconds before next execution
        """
        if self._consecutive_failures == 0:
            base_delay = self.interval_seconds
        else:
            # Exponential backoff: interval * (2 ^ failures)
            base_delay = self.interval_seconds * (2**self._consecutive_failures)
            base_delay = min(base_delay, self.max_backoff_seconds)

        # Add jitter to prevent thundering herd
        if self.jitter_factor > 0:
            jitter = base_delay * self.jitter_factor * (random.random() - 0.5)
            base_delay += jitter

        return max(1.0, base_delay)

    async def start(self) -> None:
        """Start the scheduled task execution loop."""
        if self._running or not self.enabled:
            return

        self._running = True
        logger.debug("task_starting", task_name=self.name)

        try:
            await self.setup()
            self._task = await create_managed_task(
                self._run_loop(),
                name=f"scheduled_task_{self.name}",
                creator="BaseScheduledTask",
            )
            logger.debug("task_started", task_name=self.name)
        except SchedulerError as e:
            self._running = False
            logger.error(
                "task_start_scheduler_error",
                task_name=self.name,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=e,
            )
            raise
        except Exception as e:
            self._running = False
            logger.error(
                "task_start_failed",
                task_name=self.name,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=e,
            )
            raise

    async def stop(self) -> None:
        """Stop the scheduled task execution loop."""
        if not self._running:
            return

        self._running = False
        logger.debug("task_stopping", task_name=self.name)

        # Cancel the running task
        if self._task and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

        try:
            await self.cleanup()
            logger.debug("task_stopped", task_name=self.name)
        except SchedulerError as e:
            logger.error(
                "task_cleanup_scheduler_error",
                task_name=self.name,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=e,
            )
        except Exception as e:
            logger.error(
                "task_cleanup_failed",
                task_name=self.name,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=e,
            )

    async def _run_loop(self) -> None:
        """Main execution loop for the scheduled task."""
        while self._running:
            try:
                start_time = time.time()

                # Execute the task
                success = await self.run()

                execution_time = time.time() - start_time

                if success:
                    self._consecutive_failures = 0
                    logger.debug(
                        "task_execution_success",
                        task_name=self.name,
                        execution_time=execution_time,
                    )
                else:
                    self._consecutive_failures += 1
                    logger.warning(
                        "task_execution_failed",
                        task_name=self.name,
                        consecutive_failures=self._consecutive_failures,
                        execution_time=execution_time,
                    )

                self._last_run_time = time.time()

                # Calculate delay before next execution
                delay = self.calculate_next_delay()

                if not success and self._consecutive_failures > 1:
                    logger.info(
                        "task_backoff_delay",
                        task_name=self.name,
                        consecutive_failures=self._consecutive_failures,
                        delay=delay,
                        max_backoff=self.max_backoff_seconds,
                    )

                # Wait for next execution or cancellation
                await asyncio.sleep(delay)

            except asyncio.CancelledError:
                logger.debug("task_cancelled", task_name=self.name)
                break
            except TimeoutError as e:
                self._consecutive_failures += 1
                logger.error(
                    "task_execution_timeout_error",
                    task_name=self.name,
                    error=str(e),
                    error_type=type(e).__name__,
                    consecutive_failures=self._consecutive_failures,
                    exc_info=e,
                )
                # Use backoff delay for exceptions too
                backoff_delay = self.calculate_next_delay()
                await asyncio.sleep(backoff_delay)
            except SchedulerError as e:
                self._consecutive_failures += 1
                logger.error(
                    "task_execution_scheduler_error",
                    task_name=self.name,
                    error=str(e),
                    error_type=type(e).__name__,
                    consecutive_failures=self._consecutive_failures,
                    exc_info=e,
                )
                # Use backoff delay for exceptions too
                backoff_delay = self.calculate_next_delay()
                await asyncio.sleep(backoff_delay)
            except Exception as e:
                self._consecutive_failures += 1
                logger.error(
                    "task_execution_error",
                    task_name=self.name,
                    error=str(e),
                    error_type=type(e).__name__,
                    consecutive_failures=self._consecutive_failures,
                    exc_info=e,
                )
                # Use backoff delay for exceptions too
                backoff_delay = self.calculate_next_delay()
                await asyncio.sleep(backoff_delay)

    @property
    def is_running(self) -> bool:
        """Check if the task is currently running."""
        return self._running

    @property
    def consecutive_failures(self) -> int:
        """Get the number of consecutive failures."""
        return self._consecutive_failures

    @property
    def last_run_time(self) -> float:
        """Get the timestamp of the last execution."""
        return self._last_run_time

    def get_status(self) -> dict[str, Any]:
        """
        Get current task status information.

        Returns:
            Dictionary with task status details
        """
        return {
            "name": self.name,
            "enabled": self.enabled,
            "running": self.is_running,
            "interval_seconds": self.interval_seconds,
            "consecutive_failures": self.consecutive_failures,
            "last_run_time": self.last_run_time,
            "next_delay": self.calculate_next_delay() if self.is_running else None,
        }


class PoolStatsTask(BaseScheduledTask):
    """Task for displaying pool statistics periodically."""

    def __init__(
        self,
        name: str,
        interval_seconds: float,
        enabled: bool = True,
        pool_manager: Any | None = None,
    ):
        """
        Initialize pool stats task.

        Args:
            name: Task name
            interval_seconds: Interval between stats display
            enabled: Whether task is enabled
            pool_manager: Injected pool manager instance
        """
        super().__init__(
            name=name,
            interval_seconds=interval_seconds,
            enabled=enabled,
        )
        self._pool_manager = pool_manager

    async def setup(self) -> None:
        """Initialize pool manager instance if not injected."""
        if self._pool_manager is None:
            logger.warning(
                "pool_stats_task_no_manager",
                task_name=self.name,
                message="Pool manager not injected, task will be disabled",
            )

    async def run(self) -> bool:
        """Display pool statistics."""
        try:
            if not self._pool_manager:
                return True  # Not an error, just no pool manager available

            # Get general pool stats (if available)
            general_pool = getattr(self._pool_manager, "_pool", None)
            general_stats = None
            if general_pool:
                general_stats = general_pool.get_stats()

            # Get session pool stats
            session_pool = getattr(self._pool_manager, "_session_pool", None)
            session_stats = None
            if session_pool:
                session_stats = await session_pool.get_stats()

            # Log pool statistics
            logger.debug(
                "pool_stats_report",
                task_name=self.name,
                general_pool={
                    "enabled": bool(general_pool),
                    "total_clients": general_stats.total_clients
                    if general_stats
                    else 0,
                    "available_clients": general_stats.available_clients
                    if general_stats
                    else 0,
                    "active_clients": general_stats.active_clients
                    if general_stats
                    else 0,
                    "connections_created": general_stats.connections_created
                    if general_stats
                    else 0,
                    "connections_closed": general_stats.connections_closed
                    if general_stats
                    else 0,
                    "acquire_count": general_stats.acquire_count
                    if general_stats
                    else 0,
                    "release_count": general_stats.release_count
                    if general_stats
                    else 0,
                    "health_check_failures": general_stats.health_check_failures
                    if general_stats
                    else 0,
                }
                if general_pool
                else None,
                session_pool={
                    "enabled": session_stats.get("enabled", False)
                    if session_stats
                    else False,
                    "total_sessions": session_stats.get("total_sessions", 0)
                    if session_stats
                    else 0,
                    "active_sessions": session_stats.get("active_sessions", 0)
                    if session_stats
                    else 0,
                    "max_sessions": session_stats.get("max_sessions", 0)
                    if session_stats
                    else 0,
                    "total_messages": session_stats.get("total_messages", 0)
                    if session_stats
                    else 0,
                    "session_ttl": session_stats.get("session_ttl", 0)
                    if session_stats
                    else 0,
                }
                if session_pool
                else None,
            )

            return True

        except Exception as e:
            logger.error(
                "pool_stats_task_error",
                task_name=self.name,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=e,
            )
            return False


class VersionUpdateCheckTask(BaseScheduledTask):
    """Task for checking version updates periodically."""

    def __init__(
        self,
        name: str,
        interval_seconds: float,
        enabled: bool = True,
        version_check_cache_ttl_hours: float = 1.0,
        *,
        skip_first_scheduled_run: bool = True,
    ):
        """
        Initialize version update check task.

        Args:
            name: Task name
            interval_seconds: Interval between version checks
            enabled: Whether task is enabled
            version_check_cache_ttl_hours: Maximum cache age (hours) used at startup before contacting GitHub
            skip_first_scheduled_run: If True, first scheduled loop execution is skipped
        """
        super().__init__(
            name=name,
            interval_seconds=interval_seconds,
            enabled=enabled,
        )
        self.version_check_cache_ttl_hours = version_check_cache_ttl_hours
        # Mark first scheduled execution; allow skipping to avoid duplicate run after startup
        self._first_run = True
        self._skip_first_run = skip_first_scheduled_run

    def _log_version_comparison(
        self, current_version: str, latest_version: str, *, source: str | None = None
    ) -> None:
        """
        Log version comparison results with appropriate warning level.

        Args:
            current_version: Current version string
            latest_version: Latest version string
        """
        if compare_versions(current_version, latest_version):
            logger.warning(
                "version_update_available",
                task_name=self.name,
                current_version=current_version,
                latest_version=latest_version,
                source=source,
                message=(f"New version available: {latest_version}"),
            )
        else:
            logger.debug(
                "version_check_complete_no_update",
                task_name=self.name,
                current_version=current_version,
                latest_version=latest_version,
                source=source,
                message=(
                    f"No update: latest_version={latest_version} "
                    f"current_version={current_version}"
                ),
            )

    async def run(self) -> bool:
        """Execute version update check."""
        try:
            logger.debug(
                "version_check_task_run_start",
                task_name=self.name,
                first_run=self._first_run,
            )
            state_path = get_version_check_state_path()
            current_time = datetime.now(UTC)

            # Skip first scheduled run to avoid duplicate check after startup
            if self._first_run and self._skip_first_run:
                self._first_run = False
                logger.debug(
                    "version_check_first_run_skipped",
                    task_name=self.name,
                    message="Skipping first scheduled run since startup check already completed",
                )
                return True

            # Determine freshness window using configured cache TTL
            # Applies to both startup and scheduled runs to avoid unnecessary network calls
            max_age_hours = self.version_check_cache_ttl_hours

            # Load previous state if available
            prev_state: VersionCheckState | None = await load_check_state(state_path)
            latest_version: str | None = None
            source: str | None = None

            # If we have a recent state within the freshness window, avoid network call
            if prev_state is not None:
                age_hours = (
                    current_time - prev_state.last_check_at
                ).total_seconds() / 3600.0
                if age_hours < max_age_hours:
                    logger.debug(
                        "version_check_cache_fresh",
                        task_name=self.name,
                        age_hours=round(age_hours, 3),
                        max_age_hours=max_age_hours,
                    )
                    latest_version = prev_state.latest_version_found
                    source = "cache"
                else:
                    logger.debug(
                        "version_check_cache_stale",
                        task_name=self.name,
                        age_hours=round(age_hours, 3),
                        max_age_hours=max_age_hours,
                    )

            # Fetch only if we don't have a fresh cached version
            if latest_version is None:
                latest_version = await fetch_latest_github_version()
                if latest_version is None:
                    logger.warning("version_check_fetch_failed", task_name=self.name)
                    return False
                # Persist refreshed state
                new_state = VersionCheckState(
                    last_check_at=current_time,
                    latest_version_found=latest_version,
                )
                await save_check_state(state_path, new_state)
                source = "network"
            else:
                # Ensure state file at least exists; if it didn't, we wouldn't be here
                pass

            # Compare versions and log result
            current_version = get_current_version()
            self._log_version_comparison(current_version, latest_version, source=source)

            # Mark first run as complete
            if self._first_run:
                self._first_run = False

            return True

        except ImportError as e:
            logger.error(
                "version_check_task_import_error",
                task_name=self.name,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=e,
            )
            return False

        except Exception as e:
            logger.error(
                "version_check_task_error",
                task_name=self.name,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=e,
            )
            return False


# Test helper task exposed for tests that import from this module
class MockScheduledTask(BaseScheduledTask):
    """Minimal mock task used by tests for registration and lifecycle checks."""

    async def run(self) -> bool:
        return True
