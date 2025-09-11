"""Integration tests for the scheduler system."""

from collections.abc import AsyncGenerator, Generator
from typing import Any

import pytest

from ccproxy.config.settings import Settings
from ccproxy.config.utils import SchedulerSettings
from ccproxy.core.async_task_manager import start_task_manager, stop_task_manager
from ccproxy.scheduler.core import Scheduler
from ccproxy.scheduler.errors import (
    TaskNotFoundError,
    TaskRegistrationError,
)
from ccproxy.scheduler.registry import TaskRegistry
from ccproxy.scheduler.tasks import (
    # PushgatewayTask removed - functionality moved to metrics plugin
    # StatsPrintingTask removed - functionality moved to metrics plugin
    BaseScheduledTask,
)


# Mock task for testing since PushgatewayTask moved to metrics plugin
class MockScheduledTask(BaseScheduledTask):
    """Mock scheduled task for testing."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.run_count = 0

    async def run(self) -> bool:
        self.run_count += 1
        return True


class TestSchedulerCore:
    """Test the core Scheduler functionality."""

    @pytest.fixture
    async def task_manager_lifecycle(self) -> AsyncGenerator[None, None]:
        """Start and stop the task manager for tests that need it."""
        await start_task_manager()
        try:
            yield
        finally:
            await stop_task_manager()

    @pytest.fixture
    def scheduler(self) -> Generator[Scheduler, None, None]:
        """Create a test scheduler instance."""
        registry = TaskRegistry()
        registry.clear()  # ensure clean

        # Register mock task for testing (neutral name, not tied to core plugins)
        registry.register("custom_task", MockScheduledTask)
        # registry.register("stats_printing", StatsPrintingTask)  # removed

        scheduler = Scheduler(
            task_registry=registry,
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

        await scheduler.stop()
        assert not scheduler.is_running

    @pytest.mark.asyncio
    async def test_add_task_success(
        self, scheduler: Scheduler, task_manager_lifecycle: None
    ) -> None:
        """Test successful task addition."""
        await scheduler.start()

        await scheduler.add_task(
            task_name="test_custom",
            task_type="custom_task",
            interval_seconds=60.0,
            enabled=True,
        )
        assert scheduler.task_count == 1
        assert "test_custom" in scheduler.list_tasks()

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_add_task_invalid_type(
        self, scheduler: Scheduler, task_manager_lifecycle: None
    ) -> None:
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
    async def test_remove_task_success(
        self, scheduler: Scheduler, task_manager_lifecycle: None
    ) -> None:
        """Test successful task removal."""
        await scheduler.start()

        # Add task first
        await scheduler.add_task(
            task_name="test_task",
            task_type="custom_task",
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
    async def test_remove_nonexistent_task(
        self, scheduler: Scheduler, task_manager_lifecycle: None
    ) -> None:
        """Test removing non-existent task raises error."""
        await scheduler.start()

        with pytest.raises(TaskNotFoundError, match="does not exist"):
            await scheduler.remove_task("nonexistent_task")

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_get_task_info(
        self, scheduler: Scheduler, task_manager_lifecycle: None
    ) -> None:
        """Test getting task information."""
        await scheduler.start()

        await scheduler.add_task(
            task_name="info_test",
            task_type="custom_task",
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
    async def test_get_scheduler_status(
        self, scheduler: Scheduler, task_manager_lifecycle: None
    ) -> None:
        """Test getting scheduler status information."""
        await scheduler.start()

        await scheduler.add_task(
            task_name="status_test",
            task_type="custom_task",
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
        registry.register("test_task", MockScheduledTask)

        assert registry.has("test_task")
        assert "test_task" in registry.list()
        task_class = registry.get("test_task")
        assert task_class is MockScheduledTask

    def test_register_duplicate_task_error(self, registry: TaskRegistry) -> None:
        """Test registering duplicate task raises error."""
        registry.register("duplicate_task", MockScheduledTask)

        with pytest.raises(TaskRegistrationError, match="already registered"):
            registry.register(
                "duplicate_task", MockScheduledTask
            )  # Changed from StatsPrintingTask

    def test_register_invalid_task_class_error(self, registry: TaskRegistry) -> None:
        """Test registering invalid task class raises error."""

        class InvalidTask:
            """Invalid task class that doesn't inherit from BaseScheduledTask."""

            pass

        with pytest.raises(TaskRegistrationError, match="must inherit"):
            registry.register("invalid_task", InvalidTask)  # type: ignore[arg-type]

    def test_unregister_task_success(self, registry: TaskRegistry) -> None:
        """Test successful task unregistration."""
        registry.register("temp_task", MockScheduledTask)
        assert registry.has("temp_task")

        registry.unregister("temp_task")
        assert not registry.has("temp_task")

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
        registry.register("task1", MockScheduledTask)
        registry.register("task2", MockScheduledTask)  # Changed from StatsPrintingTask

        info = registry.info()
        assert info["total_tasks"] == 2
        assert set(info["registered_tasks"]) == {"task1", "task2"}
        assert info["task_classes"]["task1"] == "MockScheduledTask"
        assert (
            info["task_classes"]["task2"] == "MockScheduledTask"
        )  # Changed from StatsPrintingTask

    def test_clear_registry(self, registry: TaskRegistry) -> None:
        """Test clearing the registry."""
        registry.register("task1", MockScheduledTask)
        registry.register("task2", MockScheduledTask)  # Changed from StatsPrintingTask
        assert len(registry.list()) == 2

        registry.clear()
        assert len(registry.list()) == 0


class TestScheduledTasks:
    """Test individual scheduled task implementations."""

    @pytest.mark.asyncio
    async def test_mock_task_lifecycle(self) -> None:
        """Test MockScheduledTask lifecycle management."""
        task = MockScheduledTask(
            name="test_mock",
            interval_seconds=0.1,  # Fast for testing
            enabled=True,
        )

        await task.setup()

        # Test single run
        result = await task.run()
        assert result is True
        assert task.run_count == 1

        await task.cleanup()

    # StatsPrintingTask test removed - functionality moved to metrics plugin

    @pytest.mark.asyncio
    async def test_task_error_handling(self) -> None:
        """Test task failure path and backoff calculation without observability."""

        class FailingTask(BaseScheduledTask):
            async def run(self) -> bool:
                return False

        task = FailingTask(
            name="error_test",
            interval_seconds=10.0,
            enabled=True,
        )

        await task.setup()

        # Test failed run
        result = await task.run()
        assert result is False

        # Simulate failure state and verify backoff
        task._consecutive_failures = 1
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
        assert settings.pricing_update_enabled is True  # Enabled by default for privacy
        assert settings.pricing_update_interval_hours == 24
        # Pushgateway settings moved to metrics plugin; not part of scheduler
        assert settings.stats_printing_enabled is False  # Disabled by default
        assert settings.version_check_enabled is True  # Enabled by default for privacy

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
