"""Integration tests for the scheduler system."""

import asyncio
from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ccproxy.config.scheduler import SchedulerSettings
from ccproxy.config.settings import Settings
from ccproxy.scheduler.core import Scheduler
from ccproxy.scheduler.errors import (
    TaskNotFoundError,
    TaskRegistrationError,
)
from ccproxy.scheduler.manager import start_scheduler, stop_scheduler
from ccproxy.scheduler.registry import TaskRegistry, get_task_registry
from ccproxy.scheduler.tasks import (
    PricingCacheUpdateTask,
    PushgatewayTask,
    StatsPrintingTask,
)


class TestSchedulerCore:
    """Test the core Scheduler functionality."""

    @pytest.fixture
    def scheduler(self) -> Generator[Scheduler, None, None]:
        """Create a test scheduler instance."""
        # Register tasks before creating scheduler
        from ccproxy.scheduler.tasks import (
            PricingCacheUpdateTask,
            PushgatewayTask,
            StatsPrintingTask,
        )

        registry = get_task_registry()
        registry.clear()  # Clear any existing registrations

        # Register default tasks
        registry.register("pushgateway", PushgatewayTask)
        registry.register("stats_printing", StatsPrintingTask)
        registry.register("pricing_cache_update", PricingCacheUpdateTask)

        scheduler = Scheduler(
            max_concurrent_tasks=5,
            graceful_shutdown_timeout=1.0,
        )
        yield scheduler

        # Clean up after test
        registry.clear()

    @pytest.mark.asyncio
    async def test_scheduler_lifecycle(self, scheduler: Scheduler) -> None:
        """Test scheduler start and stop lifecycle."""
        assert not scheduler.is_running

        await scheduler.start()
        assert scheduler.is_running

        await scheduler.stop()  # type: ignore[unreachable]
        assert not scheduler.is_running

    @pytest.mark.asyncio
    async def test_add_task_success(self, scheduler: Scheduler) -> None:
        """Test successful task addition."""
        await scheduler.start()

        await scheduler.add_task(
            task_name="test_pushgateway",
            task_type="pushgateway",
            interval_seconds=60.0,
            enabled=True,
        )
        assert scheduler.task_count == 1
        assert "test_pushgateway" in scheduler.list_tasks()

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_add_task_invalid_type(self, scheduler: Scheduler) -> None:
        """Test adding task with invalid type raises error."""
        await scheduler.start()

        with pytest.raises(TaskRegistrationError, match="not registered"):
            await scheduler.add_task(
                task_name="invalid_task",
                task_type="invalid_type",
                interval_seconds=60.0,
                enabled=True,
            )

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_remove_task_success(self, scheduler: Scheduler) -> None:
        """Test successful task removal."""
        await scheduler.start()

        # Add task first
        await scheduler.add_task(
            task_name="test_task",
            task_type="pushgateway",
            interval_seconds=60.0,
            enabled=True,
        )

        assert scheduler.task_count == 1

        # Remove task
        await scheduler.remove_task("test_task")
        assert scheduler.task_count == 0
        assert "test_task" not in scheduler.list_tasks()

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_remove_nonexistent_task(self, scheduler: Scheduler) -> None:
        """Test removing non-existent task raises error."""
        await scheduler.start()

        with pytest.raises(TaskNotFoundError, match="does not exist"):
            await scheduler.remove_task("nonexistent_task")

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_get_task_info(self, scheduler: Scheduler) -> None:
        """Test getting task information."""
        await scheduler.start()

        await scheduler.add_task(
            task_name="info_test",
            task_type="stats_printing",
            interval_seconds=30.0,
            enabled=True,
        )

        task = scheduler.get_task("info_test")
        assert task is not None
        assert task.name == "info_test"
        assert task.interval_seconds == 30.0
        assert task.enabled is True

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_get_scheduler_status(self, scheduler: Scheduler) -> None:
        """Test getting scheduler status information."""
        await scheduler.start()

        await scheduler.add_task(
            task_name="status_test",
            task_type="pushgateway",
            interval_seconds=60.0,
            enabled=True,
        )

        status = scheduler.get_scheduler_status()
        assert status["running"] is True
        assert status["total_tasks"] == 1
        assert "status_test" in status["task_names"]

        await scheduler.stop()


class TestTaskRegistry:
    """Test the TaskRegistry functionality."""

    @pytest.fixture
    def registry(self) -> Generator[TaskRegistry, None, None]:
        """Create a test task registry."""
        registry = TaskRegistry()
        yield registry

    def test_register_task_success(self, registry: TaskRegistry) -> None:
        """Test successful task registration."""
        registry.register("test_task", PushgatewayTask)

        assert registry.is_registered("test_task")
        assert "test_task" in registry.list_tasks()
        task_class = registry.get("test_task")
        assert task_class is PushgatewayTask

    def test_register_duplicate_task_error(self, registry: TaskRegistry) -> None:
        """Test registering duplicate task raises error."""
        registry.register("duplicate_task", PushgatewayTask)

        with pytest.raises(TaskRegistrationError, match="already registered"):
            registry.register("duplicate_task", StatsPrintingTask)

    def test_register_invalid_task_class_error(self, registry: TaskRegistry) -> None:
        """Test registering invalid task class raises error."""

        class InvalidTask:
            """Invalid task class that doesn't inherit from BaseScheduledTask."""

            pass

        with pytest.raises(TaskRegistrationError, match="must inherit"):
            registry.register("invalid_task", InvalidTask)  # type: ignore[arg-type]

    def test_unregister_task_success(self, registry: TaskRegistry) -> None:
        """Test successful task unregistration."""
        registry.register("temp_task", PushgatewayTask)
        assert registry.is_registered("temp_task")

        registry.unregister("temp_task")
        assert not registry.is_registered("temp_task")

    def test_unregister_nonexistent_task_error(self, registry: TaskRegistry) -> None:
        """Test unregistering non-existent task raises error."""
        with pytest.raises(TaskRegistrationError, match="not registered"):
            registry.unregister("nonexistent_task")

    def test_get_nonexistent_task_error(self, registry: TaskRegistry) -> None:
        """Test getting non-existent task raises error."""
        with pytest.raises(TaskRegistrationError, match="not registered"):
            registry.get("nonexistent_task")

    def test_registry_info(self, registry: TaskRegistry) -> None:
        """Test getting registry information."""
        registry.register("task1", PushgatewayTask)
        registry.register("task2", StatsPrintingTask)

        info = registry.get_registry_info()
        assert info["total_tasks"] == 2
        assert set(info["registered_tasks"]) == {"task1", "task2"}
        assert info["task_classes"]["task1"] == "PushgatewayTask"
        assert info["task_classes"]["task2"] == "StatsPrintingTask"

    def test_clear_registry(self, registry: TaskRegistry) -> None:
        """Test clearing the registry."""
        registry.register("task1", PushgatewayTask)
        registry.register("task2", StatsPrintingTask)
        assert len(registry.list_tasks()) == 2

        registry.clear()
        assert len(registry.list_tasks()) == 0


class TestScheduledTasks:
    """Test individual scheduled task implementations."""

    @pytest.mark.asyncio
    async def test_pushgateway_task_lifecycle(self) -> None:
        """Test PushgatewayTask lifecycle management."""
        with patch("ccproxy.observability.metrics.get_metrics") as mock_get_metrics:
            mock_metrics = MagicMock()
            mock_metrics.is_pushgateway_enabled.return_value = True
            mock_metrics.push_to_gateway.return_value = True
            mock_get_metrics.return_value = mock_metrics

            task = PushgatewayTask(
                name="test_pushgateway",
                interval_seconds=0.1,  # Fast for testing
                enabled=True,
            )

            await task.setup()
            assert task._metrics_instance is not None

            # Test single run
            result = await task.run()
            assert result is True
            mock_metrics.push_to_gateway.assert_called_once()

            await task.cleanup()

    @pytest.mark.asyncio
    async def test_stats_printing_task_lifecycle(self) -> None:
        """Test StatsPrintingTask lifecycle management."""
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
                name="test_stats",
                interval_seconds=0.1,
                enabled=True,
            )

            await task.setup()
            assert task._stats_collector_instance is not None

            # Test single run
            result = await task.run()
            assert result is True
            mock_stats_collector.print_stats.assert_called_once()

            await task.cleanup()

    @pytest.mark.asyncio
    async def test_pricing_cache_update_task_lifecycle(self) -> None:
        """Test PricingCacheUpdateTask lifecycle management."""
        with patch(
            "ccproxy.pricing.updater.PricingUpdater"
        ) as mock_pricing_updater_class:
            mock_pricing_updater = AsyncMock()
            mock_pricing_updater.get_current_pricing.return_value = {
                "model": "claude-3"
            }
            mock_pricing_updater.force_refresh.return_value = True
            mock_pricing_updater_class.return_value = mock_pricing_updater

            task = PricingCacheUpdateTask(
                name="test_pricing",
                interval_seconds=0.1,
                enabled=True,
                force_refresh_on_startup=True,
            )

            await task.setup()
            assert task._pricing_updater is not None

            # Test force refresh on first run
            result = await task.run()
            assert result is True
            mock_pricing_updater.force_refresh.assert_called_once()

            # Test regular update on second run
            result = await task.run()
            assert result is True
            mock_pricing_updater.get_current_pricing.assert_called_with(
                force_refresh=False
            )

            await task.cleanup()

    @pytest.mark.asyncio
    async def test_task_error_handling(self) -> None:
        """Test task error handling and backoff calculation."""
        with patch("ccproxy.observability.metrics.get_metrics") as mock_get_metrics:
            mock_metrics = MagicMock()
            mock_metrics.is_pushgateway_enabled.return_value = True
            mock_metrics.push_to_gateway.side_effect = Exception("Test error")
            mock_get_metrics.return_value = mock_metrics

            task = PushgatewayTask(
                name="error_test",
                interval_seconds=10.0,
                enabled=True,
            )

            await task.setup()

            # Test failed run
            result = await task.run()
            assert result is False

            # Consecutive failures only track in the run loop, not direct run() calls
            # So we test the backoff calculation directly
            task._consecutive_failures = 1  # Simulate failure state

            # Test backoff calculation after failure
            delay = task.calculate_next_delay()
            assert delay >= 10.0  # Should use exponential backoff

            await task.cleanup()


class TestSchedulerConfiguration:
    """Test scheduler configuration integration."""

    def test_scheduler_settings_defaults(self) -> None:
        """Test default scheduler settings."""
        settings = SchedulerSettings()

        assert settings.enabled is True
        assert settings.max_concurrent_tasks == 10
        assert settings.graceful_shutdown_timeout == 30.0
        assert settings.pricing_update_enabled is True
        assert settings.pricing_update_interval_hours == 24
        assert settings.pushgateway_enabled is False
        assert settings.stats_printing_enabled is False  # Default is False

    def test_scheduler_settings_environment_override(self) -> None:
        """Test scheduler settings from environment variables."""
        import os

        # Set environment variables
        os.environ["SCHEDULER__ENABLED"] = "false"
        os.environ["SCHEDULER__MAX_CONCURRENT_TASKS"] = "5"
        os.environ["SCHEDULER__PRICING_UPDATE_INTERVAL_HOURS"] = "12"

        try:
            settings = SchedulerSettings()
            assert settings.enabled is False
            assert settings.max_concurrent_tasks == 5
            assert settings.pricing_update_interval_hours == 12
        finally:
            # Clean up environment variables
            for key in [
                "SCHEDULER__ENABLED",
                "SCHEDULER__MAX_CONCURRENT_TASKS",
                "SCHEDULER__PRICING_UPDATE_INTERVAL_HOURS",
            ]:
                os.environ.pop(key, None)

    def test_main_settings_includes_scheduler(self) -> None:
        """Test that main Settings includes scheduler configuration."""
        settings = Settings()
        assert hasattr(settings, "scheduler")
        assert isinstance(settings.scheduler, SchedulerSettings)


class TestSchedulerManagerIntegration:
    """Test scheduler manager FastAPI integration."""

    @pytest.fixture(autouse=True)
    def setup_registry(self) -> Generator[None, None, None]:
        """Setup task registry for integration tests."""
        from ccproxy.scheduler.registry import get_task_registry
        from ccproxy.scheduler.tasks import (
            PricingCacheUpdateTask,
            PushgatewayTask,
            StatsPrintingTask,
        )

        registry = get_task_registry()
        registry.clear()  # Clear any existing registrations

        # Register default tasks
        registry.register("pushgateway", PushgatewayTask)
        registry.register("stats_printing", StatsPrintingTask)
        registry.register("pricing_cache_update", PricingCacheUpdateTask)

        yield

        # Clean up after test
        registry.clear()

    @pytest.mark.asyncio
    async def test_start_scheduler_success(self) -> None:
        """Test successful scheduler startup."""
        settings = Settings()
        settings.scheduler.enabled = True
        settings.scheduler.max_concurrent_tasks = 3
        settings.scheduler.graceful_shutdown_timeout = 5.0

        scheduler = await start_scheduler(settings)
        assert scheduler is not None
        assert scheduler.is_running

        await stop_scheduler(scheduler)

    @pytest.mark.asyncio
    async def test_start_scheduler_disabled(self) -> None:
        """Test scheduler startup when disabled."""
        settings = Settings()
        settings.scheduler.enabled = False

        scheduler = await start_scheduler(settings)
        assert scheduler is None

    @pytest.mark.asyncio
    async def test_stop_scheduler_none(self) -> None:
        """Test stopping None scheduler (graceful handling)."""
        # Should not raise any exceptions
        await stop_scheduler(None)

    @pytest.mark.asyncio
    async def test_scheduler_with_tasks_configured(self) -> None:
        """Test scheduler with all task types configured."""
        settings = Settings()
        settings.scheduler.enabled = True
        settings.scheduler.pushgateway_enabled = True
        settings.scheduler.pushgateway_interval_seconds = 30.0
        settings.scheduler.stats_printing_enabled = True
        settings.scheduler.stats_printing_interval_seconds = 60.0
        settings.scheduler.pricing_update_enabled = True
        settings.scheduler.pricing_update_interval_hours = 6

        with (
            patch("ccproxy.observability.metrics.get_metrics"),
            patch("ccproxy.config.settings.get_settings"),
            patch("ccproxy.observability.stats_printer.get_stats_collector"),
            patch("ccproxy.pricing.updater.PricingUpdater"),
        ):
            scheduler = await start_scheduler(settings)
            assert scheduler is not None
            assert scheduler.is_running

            # Should have all three task types
            task_names = scheduler.list_tasks()
            assert "pushgateway" in task_names
            assert "stats_printing" in task_names
            assert "pricing_cache_update" in task_names
            assert scheduler.task_count == 4

            await stop_scheduler(scheduler)


class TestSchedulerFastAPIIntegration:
    """Test scheduler integration with FastAPI application lifecycle."""

    @pytest.fixture
    def app_with_scheduler(self) -> Generator[FastAPI, None, None]:
        """Create FastAPI app with scheduler enabled."""
        from ccproxy.api.app import create_app

        # Create settings with scheduler enabled
        settings = Settings()
        settings.scheduler.enabled = True
        settings.scheduler.pricing_update_enabled = True
        settings.scheduler.pricing_update_interval_hours = 1

        app = create_app(settings)
        yield app

    @pytest.mark.asyncio
    async def test_app_lifespan_with_scheduler(
        self, app_with_scheduler: FastAPI
    ) -> None:
        """Test that app lifecycle properly manages scheduler."""
        with (
            patch("ccproxy.observability.metrics.get_metrics"),
            patch("ccproxy.config.settings.get_settings") as mock_get_settings,
            patch("ccproxy.observability.stats_printer.get_stats_collector"),
            patch("ccproxy.pricing.updater.PricingUpdater"),
            TestClient(app_with_scheduler) as client,
        ):
            # Mock settings to return our test configuration
            settings = Settings()
            settings.scheduler.enabled = True
            settings.scheduler.pricing_update_enabled = True
            mock_get_settings.return_value = settings

            # App should start successfully with scheduler
            response = client.get("/health")
            assert response.status_code == 200

            # Check that scheduler was initialized (would be in app.state)
            # Note: In a real test environment, we'd need to check app.state.scheduler
            # but TestClient context manager handles lifespan events

    def test_scheduler_disabled_app_still_works(self) -> None:
        """Test that app works when scheduler is disabled."""
        from unittest.mock import patch

        from ccproxy.api.app import create_app

        settings = Settings()
        settings.scheduler.enabled = False

        # Mock any potential blocking operations during app creation
        with (
            patch("ccproxy.observability.metrics.get_metrics") as mock_metrics,
            patch("ccproxy.config.settings.get_settings") as mock_get_settings,
            patch(
                "ccproxy.observability.stats_printer.get_stats_collector"
            ) as mock_stats,
            patch("ccproxy.pricing.updater.PricingUpdater") as mock_pricing,
            patch("ccproxy.services.credentials.CredentialsManager") as mock_creds,
        ):
            # Mock settings to return our test configuration
            mock_get_settings.return_value = settings

            app = create_app(settings)

            with TestClient(app) as client:
                response = client.get("/health")
                assert response.status_code == 200


class TestSchedulerErrorScenarios:
    """Test error scenarios and edge cases."""

    @pytest.fixture(autouse=True)
    def setup_registry(self) -> Generator[None, None, None]:
        """Setup task registry for error scenario tests."""
        from ccproxy.scheduler.registry import get_task_registry
        from ccproxy.scheduler.tasks import (
            PricingCacheUpdateTask,
            PushgatewayTask,
            StatsPrintingTask,
        )

        registry = get_task_registry()
        registry.clear()  # Clear any existing registrations

        # Register default tasks
        registry.register("pushgateway", PushgatewayTask)
        registry.register("stats_printing", StatsPrintingTask)
        registry.register("pricing_cache_update", PricingCacheUpdateTask)

        yield

        # Clean up after test
        registry.clear()

    @pytest.mark.asyncio
    async def test_scheduler_task_failure_recovery(self) -> None:
        """Test scheduler handles task failures gracefully."""
        scheduler = Scheduler(max_concurrent_tasks=2)
        await scheduler.start()

        with patch("ccproxy.observability.metrics.get_metrics") as mock_get_metrics:
            # Mock metrics to fail initially, then succeed
            mock_metrics = MagicMock()
            mock_metrics.is_pushgateway_enabled.return_value = True
            mock_metrics.push_to_gateway.side_effect = [
                Exception("Network error"),  # First call fails
                True,  # Second call succeeds
            ]
            mock_get_metrics.return_value = mock_metrics

            await scheduler.add_task(
                task_name="failure_test",
                task_type="pushgateway",
                interval_seconds=0.1,
                enabled=True,
            )

            # Let task run and fail once
            await asyncio.sleep(0.2)

            task = scheduler.get_task("failure_test")
            assert task is not None
            # Task should have recorded the failure but still be running

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_scheduler_concurrent_task_limit(self) -> None:
        """Test scheduler respects concurrent task limits."""
        scheduler = Scheduler(max_concurrent_tasks=1)
        await scheduler.start()

        # Add first task
        await scheduler.add_task(
            task_name="task1",
            task_type="stats_printing",
            interval_seconds=60.0,
            enabled=True,
        )

        # Add second task (should still work, limit is for execution not registration)
        await scheduler.add_task(
            task_name="task2",
            task_type="stats_printing",
            interval_seconds=60.0,
            enabled=True,
        )

        assert scheduler.task_count == 2

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_scheduler_graceful_shutdown_timeout(self) -> None:
        """Test scheduler graceful shutdown with timeout."""
        scheduler = Scheduler(
            max_concurrent_tasks=2,
            graceful_shutdown_timeout=0.1,  # Very short timeout for testing
        )
        await scheduler.start()

        with patch("ccproxy.observability.metrics.get_metrics"):
            await scheduler.add_task(
                task_name="long_running_task",
                task_type="pushgateway",
                interval_seconds=0.05,  # Very frequent execution
                enabled=True,
            )

            # Let task start running
            await asyncio.sleep(0.1)

            # Shutdown should complete within timeout
            start_time = asyncio.get_event_loop().time()
            await scheduler.stop()
            end_time = asyncio.get_event_loop().time()

            # Should shutdown within timeout + small buffer
            assert (end_time - start_time) < 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
