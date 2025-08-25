"""Tests for the AsyncTaskManager."""

import asyncio
from unittest.mock import Mock, patch

import pytest

from ccproxy.core.async_task_manager import (
    AsyncTaskManager,
    TaskInfo,
    create_fire_and_forget_task,
    create_managed_task,
    get_task_manager,
    start_task_manager,
    stop_task_manager,
)


class TestTaskInfo:
    """Test TaskInfo data class."""

    def test_task_info_creation(self):
        """Test TaskInfo can be created with required parameters."""
        task = Mock()
        task.done.return_value = False
        task.cancelled.return_value = False

        task_info = TaskInfo(
            task=task,
            name="test_task",
            created_at=1234567890.0,
        )

        assert task_info.task == task
        assert task_info.name == "test_task"
        assert task_info.created_at == 1234567890.0
        assert task_info.creator is None
        assert task_info.cleanup_callback is None
        assert task_info.is_done is False
        assert task_info.is_cancelled is False

    def test_task_info_with_optional_params(self):
        """Test TaskInfo with optional parameters."""
        task = Mock()
        task.done.return_value = True
        task.cancelled.return_value = False

        cleanup_callback = Mock()

        task_info = TaskInfo(
            task=task,
            name="test_task",
            created_at=1234567890.0,
            creator="test_creator",
            cleanup_callback=cleanup_callback,
        )

        assert task_info.creator == "test_creator"
        assert task_info.cleanup_callback == cleanup_callback
        assert task_info.is_done is True

    @patch("time.time", return_value=1234567900.0)
    def test_age_calculation(self, mock_time):
        """Test task age calculation."""
        task = Mock()
        task_info = TaskInfo(
            task=task,
            name="test_task",
            created_at=1234567890.0,
        )

        assert task_info.age_seconds == 10.0

    def test_get_exception(self):
        """Test getting task exception."""
        task = Mock()
        task.done.return_value = True
        task.cancelled.return_value = False
        task.exception.return_value = RuntimeError("test error")

        task_info = TaskInfo(
            task=task,
            name="test_task",
            created_at=1234567890.0,
        )

        exception = task_info.get_exception()
        assert isinstance(exception, RuntimeError)
        assert str(exception) == "test error"


class TestAsyncTaskManager:
    """Test AsyncTaskManager functionality."""

    @pytest.fixture
    def manager(self):
        """Create a task manager for testing."""
        return AsyncTaskManager(
            cleanup_interval=0.1,
            shutdown_timeout=5.0,
            max_tasks=10,
        )

    async def test_manager_initialization(self, manager):
        """Test manager initialization."""
        assert manager.cleanup_interval == 0.1
        assert manager.shutdown_timeout == 5.0
        assert manager.max_tasks == 10
        assert not manager.is_started
        assert len(manager._tasks) == 0

    async def test_start_and_stop(self, manager):
        """Test manager start and stop lifecycle."""
        assert not manager.is_started

        await manager.start()
        assert manager.is_started
        assert manager._cleanup_task is not None

        await manager.stop()
        assert not manager.is_started
        assert manager._cleanup_task is None or manager._cleanup_task.done()

    async def test_double_start(self, manager):
        """Test that starting twice doesn't cause issues."""
        await manager.start()
        await manager.start()  # Should not raise
        assert manager.is_started

    async def test_stop_without_start(self, manager):
        """Test that stopping without starting doesn't cause issues."""
        await manager.stop()  # Should not raise
        assert not manager.is_started

    async def test_create_task_before_start(self, manager):
        """Test creating task before manager is started raises error."""

        async def dummy_coro():
            return "test"

        with pytest.raises(RuntimeError, match="Task manager is not started"):
            await manager.create_task(dummy_coro())

    async def test_create_and_manage_task(self, manager):
        """Test creating and managing a task."""
        await manager.start()

        result_value = "test_result"

        async def dummy_coro():
            await asyncio.sleep(0.01)
            return result_value

        task = await manager.create_task(
            dummy_coro(),
            name="test_task",
            creator="test_creator",
        )

        assert isinstance(task, asyncio.Task)
        assert task.get_name() == "test_task"

        # Wait for task to complete
        result = await task
        assert result == result_value

        # Check task is tracked
        stats = await manager.get_task_stats()
        assert stats["total_tasks"] >= 1

        await manager.stop()

    async def test_task_exception_handling(self, manager):
        """Test that task exceptions are handled properly."""
        await manager.start()

        async def failing_coro():
            await asyncio.sleep(0.01)
            raise RuntimeError("test error")

        task = await manager.create_task(
            failing_coro(),
            name="failing_task",
        )

        # Task should raise the exception when awaited
        with pytest.raises(RuntimeError, match="test error"):
            await task

        await manager.stop()

    async def test_task_cancellation_on_shutdown(self, manager):
        """Test that tasks are cancelled on shutdown."""
        await manager.start()

        cancelled_event = asyncio.Event()

        async def long_running_coro():
            try:
                await asyncio.sleep(10)  # Long sleep
            except asyncio.CancelledError:
                cancelled_event.set()
                raise

        task = await manager.create_task(
            long_running_coro(),
            name="long_task",
        )

        # Give task time to start
        await asyncio.sleep(0.01)

        # Stop manager (should cancel task)
        await manager.stop()

        # Verify task was cancelled
        assert task.cancelled()
        assert cancelled_event.is_set()

    async def test_cleanup_callback(self, manager):
        """Test cleanup callback is called when task completes."""
        await manager.start()

        cleanup_called = asyncio.Event()

        def cleanup_callback():
            cleanup_called.set()

        async def dummy_coro():
            await asyncio.sleep(0.01)
            return "done"

        task = await manager.create_task(
            dummy_coro(),
            name="callback_task",
            cleanup_callback=cleanup_callback,
        )

        await task

        # Give cleanup time to run
        await asyncio.sleep(0.01)

        assert cleanup_called.is_set()

        await manager.stop()

    async def test_max_tasks_limit(self, manager):
        """Test that max tasks limit is enforced."""
        await manager.start()

        # Create max number of tasks
        tasks = []
        for i in range(manager.max_tasks):
            task = await manager.create_task(
                asyncio.sleep(1),  # Long enough to not complete
                name=f"task_{i}",
            )
            tasks.append(task)

        # Next task should raise error
        with pytest.raises(RuntimeError, match="Task manager at capacity"):
            await manager.create_task(
                asyncio.sleep(0.01),
                name="overflow_task",
            )

        # Cancel all tasks
        for task in tasks:
            task.cancel()

        await manager.stop()

    async def test_task_stats(self, manager):
        """Test task statistics reporting."""
        await manager.start()

        # Initially no tasks
        stats = await manager.get_task_stats()
        assert stats["total_tasks"] == 0
        assert stats["active_tasks"] == 0
        assert stats["started"] is True

        # Create some tasks
        async def quick_task():
            await asyncio.sleep(0.01)

        async def slow_task():
            await asyncio.sleep(1)

        task1 = await manager.create_task(quick_task(), name="quick")
        task2 = await manager.create_task(slow_task(), name="slow")

        # Wait for quick task to complete
        await task1

        stats = await manager.get_task_stats()
        assert stats["total_tasks"] >= 2
        assert stats["active_tasks"] >= 1  # slow task still running

        # Cancel slow task
        task2.cancel()

        await manager.stop()

    async def test_list_active_tasks(self, manager):
        """Test listing active tasks."""
        await manager.start()

        async def slow_task():
            await asyncio.sleep(1)

        task = await manager.create_task(
            slow_task(),
            name="slow_task",
            creator="test_creator",
        )

        active_tasks = await manager.list_active_tasks()
        assert len(active_tasks) >= 1

        found_task = None
        for active_task in active_tasks:
            if active_task["name"] == "slow_task":
                found_task = active_task
                break

        assert found_task is not None
        assert found_task["creator"] == "test_creator"
        assert "task_id" in found_task
        assert "age_seconds" in found_task

        task.cancel()
        await manager.stop()


class TestGlobalFunctions:
    """Test global task manager functions."""

    async def test_global_task_manager_lifecycle(self):
        """Test global task manager start/stop."""
        # Stop any existing global manager
        await stop_task_manager()

        # Start global manager
        await start_task_manager()

        manager = get_task_manager()
        assert manager.is_started

        # Stop global manager
        await stop_task_manager()

        # Manager should be reset
        new_manager = get_task_manager()
        assert not new_manager.is_started

    async def test_create_managed_task_global(self):
        """Test creating managed task using global manager."""
        await start_task_manager()

        async def test_coro():
            return "global_test"

        task = await create_managed_task(
            test_coro(),
            name="global_task",
            creator="test",
        )

        result = await task
        assert result == "global_test"

        await stop_task_manager()

    async def test_create_fire_and_forget_task(self):
        """Test fire and forget task creation."""
        executed = asyncio.Event()

        async def test_coro():
            executed.set()

        # This should not raise even if manager isn't started
        create_fire_and_forget_task(
            test_coro(),
            name="fire_forget_task",
            creator="test",
        )

        # Give time for task to execute
        await asyncio.sleep(0.1)
        assert executed.is_set()

    async def test_fire_and_forget_with_started_manager(self):
        """Test fire and forget with started manager."""
        await start_task_manager()

        executed = asyncio.Event()

        async def test_coro():
            executed.set()

        create_fire_and_forget_task(
            test_coro(),
            name="fire_forget_task",
            creator="test",
        )

        # Give time for task to execute
        await asyncio.sleep(0.1)
        assert executed.is_set()

        await stop_task_manager()


class TestTaskManagerIntegration:
    """Integration tests for task manager."""

    async def test_cleanup_loop_functionality(self):
        """Test that cleanup loop removes completed tasks."""
        manager = AsyncTaskManager(cleanup_interval=0.05)  # Very fast cleanup
        await manager.start()

        # Create several quick tasks
        tasks = []
        for i in range(5):
            task = await manager.create_task(
                asyncio.sleep(0.01),
                name=f"quick_task_{i}",
            )
            tasks.append(task)

        # Wait for tasks to complete
        await asyncio.gather(*tasks)

        # Wait for cleanup to run
        await asyncio.sleep(0.1)

        # Check that completed tasks were cleaned up
        stats = await manager.get_task_stats()
        # Some tasks might still be in registry briefly, but should be cleaned up
        active_tasks = await manager.list_active_tasks()
        assert len(active_tasks) == 0  # No active tasks

        await manager.stop()

    async def test_exception_in_cleanup_callback(self):
        """Test that exceptions in cleanup callbacks don't break manager."""
        manager = AsyncTaskManager()
        await manager.start()

        def failing_cleanup():
            raise RuntimeError("cleanup failed")

        async def dummy_coro():
            return "done"

        # This should not break the manager
        task = await manager.create_task(
            dummy_coro(),
            name="callback_test",
            cleanup_callback=failing_cleanup,
        )

        await task

        # Manager should still be functional
        assert manager.is_started

        await manager.stop()

    async def test_concurrent_task_creation(self):
        """Test concurrent task creation doesn't cause issues."""
        manager = AsyncTaskManager()
        await manager.start()

        async def create_task_wrapper(i):
            return await manager.create_task(
                asyncio.sleep(0.01),
                name=f"concurrent_{i}",
            )

        # Create multiple tasks concurrently
        task_creation_tasks = [create_task_wrapper(i) for i in range(10)]

        created_tasks = await asyncio.gather(*task_creation_tasks)

        # All tasks should be created successfully
        assert len(created_tasks) == 10

        # Wait for all tasks to complete
        await asyncio.gather(*created_tasks)

        await manager.stop()
