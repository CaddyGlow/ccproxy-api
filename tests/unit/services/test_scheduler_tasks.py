"""Unit tests for individual scheduler task implementations."""

import asyncio
from collections.abc import AsyncGenerator
from datetime import UTC
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from ccproxy.api.bootstrap import create_service_container
from ccproxy.core.async_task_manager import start_task_manager, stop_task_manager
from ccproxy.scheduler.tasks import (
    BaseScheduledTask,
    # PushgatewayTask removed - functionality moved to metrics plugin
    # StatsPrintingTask removed - functionality moved to metrics plugin
    VersionUpdateCheckTask,
)
from ccproxy.services.container import ServiceContainer


class TestBaseScheduledTask:
    """Test the BaseScheduledTask abstract base class."""

    @pytest.fixture
    async def task_manager_lifecycle(self) -> AsyncGenerator[None, None]:
        """Start and stop the task manager for tests that need it."""
        container = ServiceContainer.get_current(strict=False)
        if container is None:
            container = create_service_container()
        await start_task_manager(container=container)
        try:
            yield
        finally:
            await stop_task_manager(container=container)

    class ConcreteTask(BaseScheduledTask):
        """Concrete implementation for testing."""

        def __init__(self, should_succeed: bool = True, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.should_succeed = should_succeed
            self.run_count = 0
            self.setup_called = False
            self.cleanup_called = False

        async def run(self) -> bool:
            """Mock run implementation."""
            self.run_count += 1
            return self.should_succeed

        async def setup(self) -> None:
            """Mock setup implementation."""
            self.setup_called = True

        async def cleanup(self) -> None:
            """Mock cleanup implementation."""
            self.cleanup_called = True

    @pytest.mark.asyncio
    async def test_task_lifecycle(self, task_manager_lifecycle: None) -> None:
        """Test task start and stop lifecycle."""
        task = self.ConcreteTask(
            name="test_task",
            interval_seconds=0.1,
            enabled=True,
        )

        assert not task.is_running
        assert task.run_count == 0

        # Start the task
        await task.start()
        assert task.is_running
        # Verify setup was called during start
        assert task.setup_called  # type: ignore[unreachable]

        # Wait a short moment to let the task run at least once
        # Since we have a very short interval (0.1s), the task should run immediately
        while task.run_count == 0:
            await asyncio.sleep(0.01)  # Very short sleep to yield control

        assert task.run_count > 0

        # Stop the task - now waits deterministically for completion
        await task.stop()
        assert not task.is_running
        assert task.cleanup_called

    @pytest.mark.asyncio
    async def test_task_disabled_does_not_start(self) -> None:
        """Test disabled task doesn't start."""
        task = self.ConcreteTask(
            name="disabled_task",
            interval_seconds=0.1,
            enabled=False,
        )

        await task.start()
        assert not task.is_running
        assert not task.setup_called

    @pytest.mark.asyncio
    async def test_task_backoff_calculation(self) -> None:
        """Test exponential backoff calculation."""
        task = self.ConcreteTask(
            name="backoff_test",
            interval_seconds=10.0,
            enabled=True,
            max_backoff_seconds=60.0,
            jitter_factor=0.0,  # No jitter for predictable testing
        )

        # No failures - should use normal interval
        delay = task.calculate_next_delay()
        assert delay == 10.0

        # Simulate failures
        task._consecutive_failures = 1
        delay = task.calculate_next_delay()
        assert delay == 20.0  # 10 * 2^1

        task._consecutive_failures = 2
        delay = task.calculate_next_delay()
        assert delay == 40.0  # 10 * 2^2

        task._consecutive_failures = 3
        delay = task.calculate_next_delay()
        assert delay == 60.0  # Capped at max_backoff_seconds

        task._consecutive_failures = 10
        delay = task.calculate_next_delay()
        assert delay == 60.0  # Still capped

    def test_task_backoff_with_jitter(self) -> None:
        """Test backoff calculation includes jitter."""
        task = self.ConcreteTask(
            name="jitter_test",
            interval_seconds=10.0,
            enabled=True,
            jitter_factor=0.25,
        )

        delays = []
        for _ in range(10):
            delay = task.calculate_next_delay()
            delays.append(delay)

        # With jitter, delays should vary around base interval
        assert min(delays) >= 7.5  # 10 - (10 * 0.25 / 2)
        assert max(delays) <= 12.5  # 10 + (10 * 0.25 / 2)
        assert len(set(delays)) > 1  # Should have variation

    def test_task_status_info(self) -> None:
        """Test task status information."""
        task = self.ConcreteTask(
            name="status_test",
            interval_seconds=30.0,
            enabled=True,
        )

        status = task.get_status()
        assert status["name"] == "status_test"
        assert status["enabled"] is True
        assert status["running"] is False
        assert status["interval_seconds"] == 30.0
        assert status["consecutive_failures"] == 0
        assert status["last_run_time"] == 0
        assert status["next_delay"] is None  # Not running

    @pytest.mark.asyncio
    async def test_task_failure_tracking(self) -> None:
        """Test failure tracking using manual counter manipulation."""
        task = self.ConcreteTask(
            name="failure_test",
            interval_seconds=1.0,  # Slow interval to avoid timing issues
            enabled=True,
            should_succeed=True,
        )

        # Test initial state
        assert task.consecutive_failures == 0

        # Manually test failure counter increment (simulates run loop behavior)
        task._consecutive_failures = 3
        assert task.consecutive_failures == 3

        # Test backoff calculation with failures
        delay = task.calculate_next_delay()
        assert delay > task.interval_seconds  # Should be higher due to backoff

        # Test reset on success (simulates successful run)
        task._consecutive_failures = 0
        assert task.consecutive_failures == 0

        # Test normal interval with no failures
        delay = task.calculate_next_delay()
        assert delay >= task.interval_seconds  # Should be normal interval (+jitter)


# TestPushgatewayTask removed - functionality moved to metrics plugin
# The metrics plugin now has its own tests for the PushgatewayTask


# TestStatsPrintingTask removed - functionality moved to metrics plugin


# class TestStatsPrintingTask:
#     """Test StatsPrintingTask specific functionality."""
#
#     @pytest.mark.asyncio
#     async def test_stats_printing_task_setup(self) -> None:
#         """Test StatsPrintingTask setup process."""
#         with (
#             patch("ccproxy.config.settings.get_settings") as mock_get_settings,
#             patch("ccproxy.observability.metrics.get_metrics") as mock_get_metrics,
#             patch(
#                 "ccproxy.observability.stats_printer.get_stats_collector"
#             ) as mock_get_stats,
#         ):
#             # Setup mocks
#             mock_settings = MagicMock()
#             mock_settings.observability = MagicMock()
#             mock_get_settings.return_value = mock_settings
#
#             mock_metrics = MagicMock()
#             mock_get_metrics.return_value = mock_metrics
#
#             mock_stats_collector = AsyncMock()
#             mock_get_stats.return_value = mock_stats_collector
#
#             task = StatsPrintingTask(
#                 name="stats_setup_test",
#                 interval_seconds=60.0,
#                 enabled=True,
#             )
#
#             await task.setup()
#             assert task._stats_collector_instance is not None
#             assert task._metrics_instance is not None
#
#             await task.cleanup()
#
#     @pytest.mark.asyncio
#     async def test_stats_printing_task_run_success(self) -> None:
#         """Test successful stats printing task execution."""
#         with (
#             patch("ccproxy.config.settings.get_settings") as mock_get_settings,
#             patch("ccproxy.observability.metrics.get_metrics") as mock_get_metrics,
#             patch(
#                 "ccproxy.observability.stats_printer.get_stats_collector"
#             ) as mock_get_stats,
#         ):
#             # Setup mocks
#             mock_settings = MagicMock()
#             mock_settings.observability = MagicMock()
#             mock_get_settings.return_value = mock_settings
#
#             mock_metrics = MagicMock()
#             mock_get_metrics.return_value = mock_metrics
#
#             mock_stats_collector = AsyncMock()
#             mock_get_stats.return_value = mock_stats_collector
#
#             task = StatsPrintingTask(
#                 name="stats_success_test",
#                 interval_seconds=60.0,
#                 enabled=True,
#             )
#
#             await task.setup()
#             result = await task.run()
#
#             assert result is True
#             mock_stats_collector.print_stats.assert_called_once()
#
#             await task.cleanup()
#
#     @pytest.mark.asyncio
#     async def test_stats_printing_task_error_handling(self) -> None:
#         """Test stats printing task error handling."""
#         with (
#             patch("ccproxy.config.settings.get_settings") as mock_get_settings,
#             patch("ccproxy.observability.metrics.get_metrics") as mock_get_metrics,
#             patch(
#                 "ccproxy.observability.stats_printer.get_stats_collector"
#             ) as mock_get_stats,
#         ):
#             # Setup mocks
#             mock_settings = MagicMock()
#             mock_settings.observability = MagicMock()
#             mock_get_settings.return_value = mock_settings
#
#             mock_metrics = MagicMock()
#             mock_get_metrics.return_value = mock_metrics
#
#             mock_stats_collector = AsyncMock()
#             mock_stats_collector.print_stats.side_effect = Exception("Print error")
#             mock_get_stats.return_value = mock_stats_collector
#
#             task = StatsPrintingTask(
#                 name="stats_error_test",
#                 interval_seconds=60.0,
#                 enabled=True,
#             )
#
#             await task.setup()
#             result = await task.run()
#
#             assert result is False
#             mock_stats_collector.print_stats.assert_called_once()
#
#             await task.cleanup()
#
#
class TestVersionUpdateCheckTask:
    """Test VersionUpdateCheckTask specific functionality."""

    @pytest.mark.asyncio
    async def test_version_check_task_creation(self) -> None:
        """Test VersionUpdateCheckTask creation."""
        task = VersionUpdateCheckTask(
            name="version_check_test",
            interval_seconds=3600.0,
            enabled=True,
            version_check_cache_ttl_hours=2.0,
            skip_first_scheduled_run=False,
        )

        assert task.name == "version_check_test"
        assert task.interval_seconds == 3600.0
        assert task.enabled is True
        assert task.version_check_cache_ttl_hours == 2.0
        assert task._first_run is True

    @pytest.mark.asyncio
    async def test_version_check_task_successful_check_with_update(self) -> None:
        """Test version check task when update is available."""
        with (
            patch(
                "ccproxy.scheduler.tasks.fetch_latest_github_version",
                new_callable=AsyncMock,
            ) as mock_fetch,
            patch("ccproxy.scheduler.tasks.get_current_version") as mock_current,
            patch("ccproxy.scheduler.tasks.compare_versions") as mock_compare,
            patch(
                "ccproxy.scheduler.tasks.save_check_state", new_callable=AsyncMock
            ) as mock_save,
            patch(
                "ccproxy.scheduler.tasks.load_check_state", new_callable=AsyncMock
            ) as mock_load,
            patch("ccproxy.scheduler.tasks.get_version_check_state_path") as mock_path,
        ):
            # Mock successful fetch with newer version
            mock_fetch.return_value = "1.5.0"
            mock_current.return_value = "1.0.0"
            mock_compare.return_value = True  # Update available
            mock_load.return_value = None  # No previous state
            mock_save.return_value = None
            mock_path.return_value = "/tmp/version_check.json"

            task = VersionUpdateCheckTask(
                name="version_update_test",
                interval_seconds=3600.0,
                enabled=True,
                version_check_cache_ttl_hours=1.0,
                skip_first_scheduled_run=False,
            )

            result = await task.run()

            assert result is True
            mock_fetch.assert_called_once()
            mock_current.assert_called_once()
            mock_compare.assert_called_once_with("1.0.0", "1.5.0")
            mock_save.assert_called_once()

    @pytest.mark.asyncio
    async def test_version_check_task_no_update_available(self) -> None:
        """Test version check task when no update is available."""
        with (
            patch(
                "ccproxy.scheduler.tasks.fetch_latest_github_version",
                new_callable=AsyncMock,
            ) as mock_fetch,
            patch("ccproxy.scheduler.tasks.get_current_version") as mock_current,
            patch("ccproxy.scheduler.tasks.compare_versions") as mock_compare,
            patch(
                "ccproxy.scheduler.tasks.save_check_state", new_callable=AsyncMock
            ) as mock_save,
            patch(
                "ccproxy.scheduler.tasks.load_check_state", new_callable=AsyncMock
            ) as mock_load,
            patch("ccproxy.scheduler.tasks.get_version_check_state_path") as mock_path,
        ):
            # Mock successful fetch with same version
            mock_fetch.return_value = "1.0.0"
            mock_current.return_value = "1.0.0"
            mock_compare.return_value = False  # No update available
            mock_load.return_value = None  # No previous state
            mock_save.return_value = None
            mock_path.return_value = "/tmp/version_check.json"

            task = VersionUpdateCheckTask(
                name="version_no_update_test",
                interval_seconds=3600.0,
                enabled=True,
                version_check_cache_ttl_hours=1.0,
                skip_first_scheduled_run=False,
            )

            result = await task.run()

            assert result is True
            mock_fetch.assert_called_once()
            mock_current.assert_called_once()
            mock_compare.assert_called_once_with("1.0.0", "1.0.0")
            mock_save.assert_called_once()

    @pytest.mark.asyncio
    async def test_version_check_task_fetch_failure(self) -> None:
        """Test version check task when GitHub fetch fails."""
        with (
            patch(
                "ccproxy.scheduler.tasks.fetch_latest_github_version",
                new_callable=AsyncMock,
            ) as mock_fetch,
            patch(
                "ccproxy.scheduler.tasks.load_check_state", new_callable=AsyncMock
            ) as mock_load,
            patch("ccproxy.scheduler.tasks.get_version_check_state_path") as mock_path,
        ):
            # Mock failed fetch
            mock_fetch.return_value = None
            mock_load.return_value = None  # No previous state
            mock_path.return_value = "/tmp/version_check.json"

            task = VersionUpdateCheckTask(
                name="version_fetch_fail_test",
                interval_seconds=3600.0,
                enabled=True,
                version_check_cache_ttl_hours=1.0,
                skip_first_scheduled_run=False,
            )

            result = await task.run()

            assert result is False
            mock_fetch.assert_called_once()

    @pytest.mark.asyncio
    async def test_version_check_task_startup_skip_recent_check(self) -> None:
        """Test version check task skips startup check if recent check exists."""
        from datetime import datetime

        from ccproxy.utils.version_checker import VersionCheckState

        # Create recent state (30 minutes ago)
        recent_time = datetime.now(UTC).replace(minute=30)
        recent_state = VersionCheckState(
            last_check_at=recent_time,
            latest_version_found="1.0.0",
        )

        with (
            patch(
                "ccproxy.scheduler.tasks.load_check_state", new_callable=AsyncMock
            ) as mock_load,
            patch(
                "ccproxy.scheduler.tasks.fetch_latest_github_version",
                new_callable=AsyncMock,
            ) as mock_fetch,
            patch("ccproxy.scheduler.tasks.get_version_check_state_path") as mock_path,
        ):
            mock_load.return_value = recent_state
            mock_path.return_value = "/tmp/version_check.json"

            task = VersionUpdateCheckTask(
                name="version_skip_test",
                interval_seconds=3600.0,
                enabled=True,
                version_check_cache_ttl_hours=1.0,  # 1 hour threshold
                skip_first_scheduled_run=False,
            )

            result = await task.run()

            # Should skip because recent check was less than 1 hour ago
            assert result is True
            mock_load.assert_called_once()
            mock_fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_version_check_task_startup_run_old_check(self) -> None:
        """Test version check task runs startup check if old check exists."""
        from datetime import datetime, timedelta

        from ccproxy.utils.version_checker import VersionCheckState

        # Create old state (2 hours ago)
        old_time = datetime.now(UTC) - timedelta(hours=2)
        old_state = VersionCheckState(
            last_check_at=old_time,
            latest_version_found="1.0.0",
        )

        with (
            patch(
                "ccproxy.scheduler.tasks.load_check_state", new_callable=AsyncMock
            ) as mock_load,
            patch(
                "ccproxy.scheduler.tasks.fetch_latest_github_version",
                new_callable=AsyncMock,
            ) as mock_fetch,
            patch("ccproxy.scheduler.tasks.get_current_version") as mock_current,
            patch("ccproxy.scheduler.tasks.compare_versions") as mock_compare,
            patch(
                "ccproxy.scheduler.tasks.save_check_state", new_callable=AsyncMock
            ) as mock_save,
            patch("ccproxy.scheduler.tasks.get_version_check_state_path") as mock_path,
        ):
            mock_load.return_value = old_state
            mock_fetch.return_value = "1.1.0"
            mock_current.return_value = "1.0.0"
            mock_compare.return_value = True
            mock_save.return_value = None
            mock_path.return_value = "/tmp/version_check.json"

            task = VersionUpdateCheckTask(
                name="version_old_test",
                interval_seconds=3600.0,
                enabled=True,
                version_check_cache_ttl_hours=1.0,  # 1 hour threshold
                skip_first_scheduled_run=False,
            )

            result = await task.run()

            # Should run because old check was more than 1 hour ago
            assert result is True
            mock_load.assert_called_once()
            mock_fetch.assert_called_once()

    @pytest.mark.asyncio
    async def test_version_check_task_second_run_normal(self) -> None:
        """Test version check task second run behavior (non-startup)."""
        with (
            patch(
                "ccproxy.scheduler.tasks.load_check_state", new_callable=AsyncMock
            ) as mock_load,
            patch(
                "ccproxy.scheduler.tasks.fetch_latest_github_version",
                new_callable=AsyncMock,
            ) as mock_fetch,
            patch("ccproxy.scheduler.tasks.get_current_version") as mock_current,
            patch("ccproxy.scheduler.tasks.compare_versions") as mock_compare,
            patch(
                "ccproxy.scheduler.tasks.save_check_state", new_callable=AsyncMock
            ) as mock_save,
            patch("ccproxy.scheduler.tasks.get_version_check_state_path") as mock_path,
        ):
            mock_load.return_value = None  # No previous state
            mock_fetch.return_value = "1.1.0"
            mock_current.return_value = "1.0.0"
            mock_compare.return_value = True
            mock_save.return_value = None
            mock_path.return_value = "/tmp/version_check.json"

            task = VersionUpdateCheckTask(
                name="version_second_run_test",
                interval_seconds=3600.0,
                enabled=True,
                version_check_cache_ttl_hours=1.0,
                skip_first_scheduled_run=False,
            )

            # First run (startup logic)
            result1 = await task.run()
            assert result1 is True
            assert task._first_run is False

            # Second run (normal logic, should skip startup checks)
            result2 = await task.run()
            assert result2 is True

            # Should have called fetch twice (once for each run)
            assert mock_fetch.call_count == 2

    @pytest.mark.asyncio
    async def test_version_check_task_error_handling(self) -> None:
        """Test version check task error handling."""
        with (
            patch(
                "ccproxy.scheduler.tasks.load_check_state", new_callable=AsyncMock
            ) as mock_load,
            patch("ccproxy.scheduler.tasks.get_version_check_state_path") as mock_path,
        ):
            # Mock exception during load
            mock_load.side_effect = Exception("File system error")
            mock_path.return_value = "/tmp/version_check.json"

            task = VersionUpdateCheckTask(
                name="version_error_test",
                interval_seconds=3600.0,
                enabled=True,
                version_check_cache_ttl_hours=1.0,
                skip_first_scheduled_run=False,
            )

            result = await task.run()

            assert result is False
            mock_load.assert_called_once()

    @pytest.mark.asyncio
    async def test_version_check_cache_invalidated_on_version_change(self) -> None:
        """Cache is ignored when local version differs from cached information."""

        from datetime import datetime

        from ccproxy.utils.version_checker import VersionCheckState

        cached_time = datetime.now(UTC)
        cached_state = VersionCheckState(
            last_check_at=cached_time,
            latest_version_found="1.1.0",
            running_version="1.1.0",
        )

        with (
            patch(
                "ccproxy.scheduler.tasks.load_check_state", new_callable=AsyncMock
            ) as mock_load,
            patch(
                "ccproxy.scheduler.tasks.fetch_latest_github_version",
                new_callable=AsyncMock,
            ) as mock_fetch,
            patch("ccproxy.scheduler.tasks.get_current_version") as mock_current,
            patch("ccproxy.scheduler.tasks.compare_versions") as mock_compare,
            patch(
                "ccproxy.scheduler.tasks.save_check_state", new_callable=AsyncMock
            ) as mock_save,
            patch("ccproxy.scheduler.tasks.get_version_check_state_path") as mock_path,
        ):
            mock_load.return_value = cached_state
            mock_fetch.return_value = "1.2.0"
            mock_current.return_value = "2.0.0"
            mock_compare.return_value = True
            mock_save.return_value = None
            mock_path.return_value = "/tmp/version_check.json"

            task = VersionUpdateCheckTask(
                name="version_cache_invalidation_test",
                interval_seconds=3600.0,
                enabled=True,
                version_check_cache_ttl_hours=6.0,
                skip_first_scheduled_run=False,
            )

            result = await task.run()

            assert result is True
            mock_fetch.assert_awaited_once()
            mock_compare.assert_called_once_with("2.0.0", "1.2.0")
            mock_save.assert_awaited_once()

            saved_state = mock_save.await_args.args[1]  # type: ignore[union-attr]
            assert isinstance(saved_state, VersionCheckState)
            assert saved_state.running_version == "2.0.0"

    @pytest.mark.asyncio
    async def test_version_check_dev_branch_resolution(self) -> None:
        """Dev versions use branch commit comparison for updates."""

        current_dev_version = "0.2.0.dev37+gabcdef12"
        branch_name = "main"
        branch_commit = "abcdef1234567890abcdef1234567890abcdef12"

        with (
            patch(
                "ccproxy.scheduler.tasks.load_check_state", new_callable=AsyncMock
            ) as mock_load,
            patch(
                "ccproxy.scheduler.tasks.fetch_latest_github_version",
                new_callable=AsyncMock,
            ) as mock_release_fetch,
            patch(
                "ccproxy.scheduler.tasks.fetch_latest_branch_commit",
                new_callable=AsyncMock,
            ) as mock_branch_fetch,
            patch(
                "ccproxy.scheduler.tasks.resolve_branch_for_commit",
                new_callable=AsyncMock,
            ) as mock_branch_resolve,
            patch("ccproxy.scheduler.tasks.get_current_version") as mock_current,
            patch(
                "ccproxy.scheduler.tasks.save_check_state", new_callable=AsyncMock
            ) as mock_save,
            patch("ccproxy.scheduler.tasks.get_version_check_state_path") as mock_path,
            patch("ccproxy.scheduler.tasks.logger") as mock_logger,
        ):
            mock_load.return_value = None
            mock_branch_resolve.return_value = branch_name
            mock_branch_fetch.return_value = branch_commit
            mock_current.return_value = current_dev_version
            mock_save.return_value = None
            mock_path.return_value = "/tmp/version_check.json"

            task = VersionUpdateCheckTask(
                name="version_branch_test",
                interval_seconds=3600.0,
                enabled=True,
                version_check_cache_ttl_hours=1.0,
                skip_first_scheduled_run=False,
            )

            result = await task.run()

            assert result is True
            mock_branch_resolve.assert_awaited_once_with("abcdef12")
            mock_branch_fetch.assert_awaited_once_with(branch_name)
            mock_release_fetch.assert_not_called()
            mock_save.assert_awaited_once()
            mock_logger.warning.assert_not_called()

            saved_state = mock_save.await_args.args[1]  # type: ignore[union-attr]

            from ccproxy.utils.version_checker import VersionCheckState

            assert isinstance(saved_state, VersionCheckState)
            assert saved_state.latest_branch_name == branch_name
            assert saved_state.latest_branch_commit == branch_commit


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
