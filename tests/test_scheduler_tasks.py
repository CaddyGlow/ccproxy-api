"""Unit tests for individual scheduler task implementations."""

import asyncio
from datetime import UTC
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccproxy.scheduler.tasks import (
    BaseScheduledTask,
    PricingCacheUpdateTask,
    PushgatewayTask,
    StatsPrintingTask,
    VersionUpdateCheckTask,
)


class TestBaseScheduledTask:
    """Test the BaseScheduledTask abstract base class."""

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
    async def test_task_lifecycle(self) -> None:
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

        # Let it run a few times
        await asyncio.sleep(0.25)
        assert task.run_count > 0

        # Stop the task
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


class TestPushgatewayTask:
    """Test PushgatewayTask specific functionality."""

    @pytest.mark.asyncio
    async def test_pushgateway_task_setup(self) -> None:
        """Test PushgatewayTask setup process."""
        with patch("ccproxy.observability.metrics.get_metrics") as mock_get_metrics:
            mock_metrics = MagicMock()
            mock_get_metrics.return_value = mock_metrics

            task = PushgatewayTask(
                name="pg_setup_test",
                interval_seconds=60.0,
                enabled=True,
            )

            await task.setup()
            assert task._metrics_instance is not None
            mock_get_metrics.assert_called_once()

            await task.cleanup()

    @pytest.mark.asyncio
    async def test_pushgateway_task_run_success(self) -> None:
        """Test successful pushgateway task execution."""
        with patch("ccproxy.observability.metrics.get_metrics") as mock_get_metrics:
            mock_metrics = MagicMock()
            mock_metrics.is_pushgateway_enabled.return_value = True
            mock_metrics.push_to_gateway.return_value = True
            mock_get_metrics.return_value = mock_metrics

            task = PushgatewayTask(
                name="pg_success_test",
                interval_seconds=60.0,
                enabled=True,
            )

            await task.setup()
            result = await task.run()

            assert result is True
            mock_metrics.push_to_gateway.assert_called_once()

            await task.cleanup()

    @pytest.mark.asyncio
    async def test_pushgateway_task_disabled(self) -> None:
        """Test pushgateway task when disabled."""
        with patch("ccproxy.observability.metrics.get_metrics") as mock_get_metrics:
            mock_metrics = MagicMock()
            mock_metrics.is_pushgateway_enabled.return_value = False
            mock_get_metrics.return_value = mock_metrics

            task = PushgatewayTask(
                name="pg_disabled_test",
                interval_seconds=60.0,
                enabled=True,
            )

            await task.setup()
            result = await task.run()

            # Should return True (not an error) but not call push_to_gateway
            assert result is True
            mock_metrics.push_to_gateway.assert_not_called()

            await task.cleanup()

    @pytest.mark.asyncio
    async def test_pushgateway_task_error_handling(self) -> None:
        """Test pushgateway task error handling."""
        with patch("ccproxy.observability.metrics.get_metrics") as mock_get_metrics:
            mock_metrics = MagicMock()
            mock_metrics.is_pushgateway_enabled.return_value = True
            mock_metrics.push_to_gateway.side_effect = Exception("Network error")
            mock_get_metrics.return_value = mock_metrics

            task = PushgatewayTask(
                name="pg_error_test",
                interval_seconds=60.0,
                enabled=True,
            )

            await task.setup()
            result = await task.run()

            assert result is False
            mock_metrics.push_to_gateway.assert_called_once()

            await task.cleanup()


class TestStatsPrintingTask:
    """Test StatsPrintingTask specific functionality."""

    @pytest.mark.asyncio
    async def test_stats_printing_task_setup(self) -> None:
        """Test StatsPrintingTask setup process."""
        with (
            patch("ccproxy.config.settings.get_settings") as mock_get_settings,
            patch("ccproxy.observability.metrics.get_metrics") as mock_get_metrics,
            patch(
                "ccproxy.observability.stats_printer.get_stats_collector"
            ) as mock_get_stats,
        ):
            # Setup mocks
            mock_settings = MagicMock()
            mock_settings.observability = MagicMock()
            mock_get_settings.return_value = mock_settings

            mock_metrics = MagicMock()
            mock_get_metrics.return_value = mock_metrics

            mock_stats_collector = AsyncMock()
            mock_get_stats.return_value = mock_stats_collector

            task = StatsPrintingTask(
                name="stats_setup_test",
                interval_seconds=60.0,
                enabled=True,
            )

            await task.setup()
            assert task._stats_collector_instance is not None
            assert task._metrics_instance is not None

            await task.cleanup()

    @pytest.mark.asyncio
    async def test_stats_printing_task_run_success(self) -> None:
        """Test successful stats printing task execution."""
        with (
            patch("ccproxy.config.settings.get_settings") as mock_get_settings,
            patch("ccproxy.observability.metrics.get_metrics") as mock_get_metrics,
            patch(
                "ccproxy.observability.stats_printer.get_stats_collector"
            ) as mock_get_stats,
        ):
            # Setup mocks
            mock_settings = MagicMock()
            mock_settings.observability = MagicMock()
            mock_get_settings.return_value = mock_settings

            mock_metrics = MagicMock()
            mock_get_metrics.return_value = mock_metrics

            mock_stats_collector = AsyncMock()
            mock_get_stats.return_value = mock_stats_collector

            task = StatsPrintingTask(
                name="stats_success_test",
                interval_seconds=60.0,
                enabled=True,
            )

            await task.setup()
            result = await task.run()

            assert result is True
            mock_stats_collector.print_stats.assert_called_once()

            await task.cleanup()

    @pytest.mark.asyncio
    async def test_stats_printing_task_error_handling(self) -> None:
        """Test stats printing task error handling."""
        with (
            patch("ccproxy.config.settings.get_settings") as mock_get_settings,
            patch("ccproxy.observability.metrics.get_metrics") as mock_get_metrics,
            patch(
                "ccproxy.observability.stats_printer.get_stats_collector"
            ) as mock_get_stats,
        ):
            # Setup mocks
            mock_settings = MagicMock()
            mock_settings.observability = MagicMock()
            mock_get_settings.return_value = mock_settings

            mock_metrics = MagicMock()
            mock_get_metrics.return_value = mock_metrics

            mock_stats_collector = AsyncMock()
            mock_stats_collector.print_stats.side_effect = Exception("Print error")
            mock_get_stats.return_value = mock_stats_collector

            task = StatsPrintingTask(
                name="stats_error_test",
                interval_seconds=60.0,
                enabled=True,
            )

            await task.setup()
            result = await task.run()

            assert result is False
            mock_stats_collector.print_stats.assert_called_once()

            await task.cleanup()


class TestPricingCacheUpdateTask:
    """Test PricingCacheUpdateTask specific functionality."""

    @pytest.mark.asyncio
    async def test_pricing_task_setup(self) -> None:
        """Test PricingCacheUpdateTask setup process."""
        with patch(
            "ccproxy.pricing.updater.PricingUpdater"
        ) as mock_pricing_updater_class:
            mock_pricing_updater = AsyncMock()
            mock_pricing_updater_class.return_value = mock_pricing_updater

            task = PricingCacheUpdateTask(
                name="pricing_setup_test",
                interval_seconds=3600.0,
                enabled=True,
            )

            await task.setup()
            assert task._pricing_updater is not None
            mock_pricing_updater_class.assert_called_once()

            await task.cleanup()

    @pytest.mark.asyncio
    async def test_pricing_task_force_refresh_on_startup(self) -> None:
        """Test pricing task force refresh on startup."""
        with patch(
            "ccproxy.pricing.updater.PricingUpdater"
        ) as mock_pricing_updater_class:
            mock_pricing_updater = AsyncMock()
            mock_pricing_updater.force_refresh.return_value = True
            mock_pricing_updater_class.return_value = mock_pricing_updater

            task = PricingCacheUpdateTask(
                name="pricing_force_test",
                interval_seconds=3600.0,
                enabled=True,
                force_refresh_on_startup=True,
            )

            await task.setup()

            # First run should force refresh
            result = await task.run()
            assert result is True
            mock_pricing_updater.force_refresh.assert_called_once()

            await task.cleanup()

    @pytest.mark.asyncio
    async def test_pricing_task_regular_update(self) -> None:
        """Test pricing task regular update behavior."""
        with patch(
            "ccproxy.pricing.updater.PricingUpdater"
        ) as mock_pricing_updater_class:
            mock_pricing_updater = AsyncMock()
            mock_pricing_updater.get_current_pricing.return_value = {
                "model": "claude-3"
            }
            mock_pricing_updater_class.return_value = mock_pricing_updater

            task = PricingCacheUpdateTask(
                name="pricing_regular_test",
                interval_seconds=3600.0,
                enabled=True,
                force_refresh_on_startup=False,
            )

            await task.setup()

            # Regular run should check current pricing
            result = await task.run()
            assert result is True
            mock_pricing_updater.get_current_pricing.assert_called_once_with(
                force_refresh=False
            )

            await task.cleanup()

    @pytest.mark.asyncio
    async def test_pricing_task_startup_then_regular(self) -> None:
        """Test pricing task startup behavior then regular behavior."""
        with patch(
            "ccproxy.pricing.updater.PricingUpdater"
        ) as mock_pricing_updater_class:
            mock_pricing_updater = AsyncMock()
            mock_pricing_updater.force_refresh.return_value = True
            mock_pricing_updater.get_current_pricing.return_value = {
                "model": "claude-3"
            }
            mock_pricing_updater_class.return_value = mock_pricing_updater

            task = PricingCacheUpdateTask(
                name="pricing_transition_test",
                interval_seconds=3600.0,
                enabled=True,
                force_refresh_on_startup=True,
            )

            await task.setup()

            # First run should force refresh
            result1 = await task.run()
            assert result1 is True
            mock_pricing_updater.force_refresh.assert_called_once()

            # Second run should do regular update
            result2 = await task.run()
            assert result2 is True
            mock_pricing_updater.get_current_pricing.assert_called_once_with(
                force_refresh=False
            )

            await task.cleanup()

    @pytest.mark.asyncio
    async def test_pricing_task_error_handling(self) -> None:
        """Test pricing task error handling."""
        with patch(
            "ccproxy.pricing.updater.PricingUpdater"
        ) as mock_pricing_updater_class:
            mock_pricing_updater = AsyncMock()
            mock_pricing_updater.get_current_pricing.side_effect = Exception(
                "Update error"
            )
            mock_pricing_updater_class.return_value = mock_pricing_updater

            task = PricingCacheUpdateTask(
                name="pricing_error_test",
                interval_seconds=3600.0,
                enabled=True,
                force_refresh_on_startup=False,
            )

            await task.setup()
            result = await task.run()

            assert result is False
            mock_pricing_updater.get_current_pricing.assert_called_once()

            await task.cleanup()

    @pytest.mark.asyncio
    async def test_pricing_task_no_data_returned(self) -> None:
        """Test pricing task when no data is returned."""
        with patch(
            "ccproxy.pricing.updater.PricingUpdater"
        ) as mock_pricing_updater_class:
            mock_pricing_updater = AsyncMock()
            mock_pricing_updater.get_current_pricing.return_value = None
            mock_pricing_updater_class.return_value = mock_pricing_updater

            task = PricingCacheUpdateTask(
                name="pricing_no_data_test",
                interval_seconds=3600.0,
                enabled=True,
                force_refresh_on_startup=False,
            )

            await task.setup()
            result = await task.run()

            # Should return False when no data is returned
            assert result is False
            mock_pricing_updater.get_current_pricing.assert_called_once()

            await task.cleanup()


class TestVersionUpdateCheckTask:
    """Test VersionUpdateCheckTask specific functionality."""

    @pytest.mark.asyncio
    async def test_version_check_task_creation(self) -> None:
        """Test VersionUpdateCheckTask creation."""
        task = VersionUpdateCheckTask(
            name="version_check_test",
            interval_seconds=3600.0,
            enabled=True,
            startup_max_age_hours=2.0,
        )

        assert task.name == "version_check_test"
        assert task.interval_seconds == 3600.0
        assert task.enabled is True
        assert task.startup_max_age_hours == 2.0
        assert task._first_run is True

    @pytest.mark.asyncio
    async def test_version_check_task_successful_check_with_update(self) -> None:
        """Test version check task when update is available."""
        with (
            patch(
                "ccproxy.utils.version_checker.fetch_latest_github_version"
            ) as mock_fetch,
            patch("ccproxy.utils.version_checker.get_current_version") as mock_current,
            patch("ccproxy.utils.version_checker.compare_versions") as mock_compare,
            patch("ccproxy.utils.version_checker.save_check_state") as mock_save,
            patch("ccproxy.utils.version_checker.load_check_state") as mock_load,
        ):
            # Mock successful fetch with newer version
            mock_fetch.return_value = "1.5.0"
            mock_current.return_value = "1.0.0"
            mock_compare.return_value = True  # Update available
            mock_load.return_value = None  # No previous state
            mock_save.return_value = None

            task = VersionUpdateCheckTask(
                name="version_update_test",
                interval_seconds=3600.0,
                enabled=True,
                startup_max_age_hours=1.0,
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
                "ccproxy.utils.version_checker.fetch_latest_github_version"
            ) as mock_fetch,
            patch("ccproxy.utils.version_checker.get_current_version") as mock_current,
            patch("ccproxy.utils.version_checker.compare_versions") as mock_compare,
            patch("ccproxy.utils.version_checker.save_check_state") as mock_save,
            patch("ccproxy.utils.version_checker.load_check_state") as mock_load,
        ):
            # Mock successful fetch with same version
            mock_fetch.return_value = "1.0.0"
            mock_current.return_value = "1.0.0"
            mock_compare.return_value = False  # No update available
            mock_load.return_value = None  # No previous state
            mock_save.return_value = None

            task = VersionUpdateCheckTask(
                name="version_no_update_test",
                interval_seconds=3600.0,
                enabled=True,
                startup_max_age_hours=1.0,
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
                "ccproxy.utils.version_checker.fetch_latest_github_version"
            ) as mock_fetch,
            patch("ccproxy.utils.version_checker.load_check_state") as mock_load,
        ):
            # Mock failed fetch
            mock_fetch.return_value = None
            mock_load.return_value = None  # No previous state

            task = VersionUpdateCheckTask(
                name="version_fetch_fail_test",
                interval_seconds=3600.0,
                enabled=True,
                startup_max_age_hours=1.0,
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
            patch("ccproxy.utils.version_checker.load_check_state") as mock_load,
            patch(
                "ccproxy.utils.version_checker.fetch_latest_github_version"
            ) as mock_fetch,
        ):
            mock_load.return_value = recent_state

            task = VersionUpdateCheckTask(
                name="version_skip_test",
                interval_seconds=3600.0,
                enabled=True,
                startup_max_age_hours=1.0,  # 1 hour threshold
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
            patch("ccproxy.utils.version_checker.load_check_state") as mock_load,
            patch(
                "ccproxy.utils.version_checker.fetch_latest_github_version"
            ) as mock_fetch,
            patch("ccproxy.utils.version_checker.get_current_version") as mock_current,
            patch("ccproxy.utils.version_checker.compare_versions") as mock_compare,
            patch("ccproxy.utils.version_checker.save_check_state") as mock_save,
        ):
            mock_load.return_value = old_state
            mock_fetch.return_value = "1.1.0"
            mock_current.return_value = "1.0.0"
            mock_compare.return_value = True
            mock_save.return_value = None

            task = VersionUpdateCheckTask(
                name="version_old_test",
                interval_seconds=3600.0,
                enabled=True,
                startup_max_age_hours=1.0,  # 1 hour threshold
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
            patch("ccproxy.utils.version_checker.load_check_state") as mock_load,
            patch(
                "ccproxy.utils.version_checker.fetch_latest_github_version"
            ) as mock_fetch,
            patch("ccproxy.utils.version_checker.get_current_version") as mock_current,
            patch("ccproxy.utils.version_checker.compare_versions") as mock_compare,
            patch("ccproxy.utils.version_checker.save_check_state") as mock_save,
        ):
            mock_load.return_value = None  # No previous state
            mock_fetch.return_value = "1.1.0"
            mock_current.return_value = "1.0.0"
            mock_compare.return_value = True
            mock_save.return_value = None

            task = VersionUpdateCheckTask(
                name="version_second_run_test",
                interval_seconds=3600.0,
                enabled=True,
                startup_max_age_hours=1.0,
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
            patch("ccproxy.utils.version_checker.load_check_state") as mock_load,
        ):
            # Mock exception during load
            mock_load.side_effect = Exception("File system error")

            task = VersionUpdateCheckTask(
                name="version_error_test",
                interval_seconds=3600.0,
                enabled=True,
                startup_max_age_hours=1.0,
            )

            result = await task.run()

            assert result is False
            mock_load.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
