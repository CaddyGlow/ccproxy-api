"""Tests for hook ordering and priority system."""

import asyncio
from datetime import datetime
from typing import Any

import pytest

from ccproxy.core.plugins.hooks import HookEvent, HookManager, HookRegistry
from ccproxy.core.plugins.hooks.base import Hook, HookContext
from ccproxy.core.plugins.hooks.layers import HookLayer


class MockHook(Hook):
    """Mock hook that records execution order."""

    def __init__(self, name: str, priority: int, execution_log: list[str]):
        self._name = name
        self._priority = priority
        self._execution_log = execution_log
        self._events = [HookEvent.REQUEST_STARTED]

    @property
    def name(self) -> str:
        return self._name

    @property
    def priority(self) -> int:
        return self._priority

    @property
    def events(self) -> list[HookEvent]:
        return self._events

    async def __call__(self, context: HookContext) -> None:
        """Record execution in the log."""
        self._execution_log.append(self.name)
        # Simulate some async work
        await asyncio.sleep(0.001)


class DataModifyingHook(Hook):
    """Test hook that modifies context data."""

    def __init__(self, name: str, priority: int, field: str, value: Any):
        self._name = name
        self._priority = priority
        self._field = field
        self._value = value
        self._events = [HookEvent.REQUEST_STARTED]

    @property
    def name(self) -> str:
        return self._name

    @property
    def priority(self) -> int:
        return self._priority

    @property
    def events(self) -> list[HookEvent]:
        return self._events

    async def __call__(self, context: HookContext) -> None:
        """Modify context data."""
        context.data[self._field] = self._value
        context.metadata[f"{self._field}_modified_by"] = self.name


class MockHookOrdering:
    """Test hook priority and ordering functionality."""

    @pytest.mark.asyncio
    async def test_hooks_execute_in_priority_order(self) -> None:
        """Test that hooks execute in priority order."""
        registry = HookRegistry()
        manager = HookManager(registry)
        execution_log: list[str] = []

        # Register hooks in random order with different priorities
        hook_high = MockHook("high_priority", 100, execution_log)
        hook_medium = MockHook("medium_priority", 500, execution_log)
        hook_low = MockHook("low_priority", 900, execution_log)

        # Register in non-priority order
        registry.register(hook_medium)
        registry.register(hook_low)
        registry.register(hook_high)

        # Emit event
        await manager.emit(HookEvent.REQUEST_STARTED, {"test": "data"})

        # Verify execution order
        assert execution_log == ["high_priority", "medium_priority", "low_priority"]

    @pytest.mark.asyncio
    async def test_hooks_with_same_priority_maintain_registration_order(self) -> None:
        """Test that hooks with same priority execute in registration order."""
        registry = HookRegistry()
        manager = HookManager(registry)
        execution_log: list[str] = []

        # Create hooks with same priority
        hook1 = MockHook("hook1", 500, execution_log)
        hook2 = MockHook("hook2", 500, execution_log)
        hook3 = MockHook("hook3", 500, execution_log)

        # Register in specific order
        registry.register(hook1)
        registry.register(hook2)
        registry.register(hook3)

        # Emit event
        await manager.emit(HookEvent.REQUEST_STARTED, {"test": "data"})

        # Verify registration order is maintained
        assert execution_log == ["hook1", "hook2", "hook3"]

    @pytest.mark.asyncio
    async def test_hook_layers_ordering(self) -> None:
        """Test that standard hook layers work correctly."""
        registry = HookRegistry()
        manager = HookManager(registry)
        execution_log: list[str] = []

        # Create hooks for different layers
        critical_hook = MockHook("critical", HookLayer.CRITICAL, execution_log)
        auth_hook = MockHook("auth", HookLayer.AUTH, execution_log)
        enrichment_hook = MockHook("enrichment", HookLayer.ENRICHMENT, execution_log)
        processing_hook = MockHook("processing", HookLayer.PROCESSING, execution_log)
        observation_hook = MockHook("observation", HookLayer.OBSERVATION, execution_log)
        cleanup_hook = MockHook("cleanup", HookLayer.CLEANUP, execution_log)

        # Register in random order
        registry.register(observation_hook)
        registry.register(auth_hook)
        registry.register(cleanup_hook)
        registry.register(critical_hook)
        registry.register(processing_hook)
        registry.register(enrichment_hook)

        # Emit event
        await manager.emit(HookEvent.REQUEST_STARTED, {"test": "data"})

        # Verify layer ordering
        assert execution_log == [
            "critical",
            "auth",
            "enrichment",
            "processing",
            "observation",
            "cleanup",
        ]

    @pytest.mark.asyncio
    async def test_data_modification_in_order(self) -> None:
        """Test that hooks can modify data and later hooks see changes."""
        registry = HookRegistry()
        manager = HookManager(registry)

        # Create hooks that modify data in sequence
        hook1 = DataModifyingHook("enricher1", HookLayer.ENRICHMENT, "user_id", "123")
        hook2 = DataModifyingHook(
            "enricher2", HookLayer.ENRICHMENT + 10, "user_name", "test_user"
        )
        hook3 = DataModifyingHook("processor", HookLayer.PROCESSING, "processed", True)

        registry.register(hook3)  # Register out of order
        registry.register(hook1)
        registry.register(hook2)

        # Create context and emit
        context = HookContext(
            event=HookEvent.REQUEST_STARTED,
            timestamp=datetime.utcnow(),
            data={},
            metadata={},
        )

        await manager.emit_with_context(context)

        # Verify data modifications happened in priority order
        assert context.data == {
            "user_id": "123",
            "user_name": "test_user",
            "processed": True,
        }
        assert context.metadata == {
            "user_id_modified_by": "enricher1",
            "user_name_modified_by": "enricher2",
            "processed_modified_by": "processor",
        }

    @pytest.mark.asyncio
    async def test_hook_registry_summary(self) -> None:
        """Test that registry provides correct summary of hooks."""
        registry = HookRegistry()

        # Register some hooks
        hook1 = MockHook("hook1", 100, [])
        hook2 = MockHook("hook2", 500, [])
        hook3 = MockHook("hook3", 700, [])

        registry.register(hook1)
        registry.register(hook2)
        registry.register(hook3)

        # Get summary
        summary = registry.list()

        # Verify summary structure
        assert HookEvent.REQUEST_STARTED.value in summary
        hooks = summary[HookEvent.REQUEST_STARTED.value]
        assert len(hooks) == 3

        # Verify hooks are in priority order in summary
        assert hooks[0]["name"] == "hook1"
        assert hooks[0]["priority"] == 100
        assert hooks[1]["name"] == "hook2"
        assert hooks[1]["priority"] == 500
        assert hooks[2]["name"] == "hook3"
        assert hooks[2]["priority"] == 700

    @pytest.mark.asyncio
    async def test_hook_failure_doesnt_stop_others(self) -> None:
        """Test that one hook failing doesn't prevent others from executing."""

        class FailingHook(Hook):
            """Hook that raises an exception."""

            name = "failing"
            priority = 500
            events = [HookEvent.REQUEST_STARTED]

            async def __call__(self, context: HookContext) -> None:
                raise ValueError("Intentional failure")

        registry = HookRegistry()
        manager = HookManager(registry)
        execution_log: list[str] = []

        # Register hooks
        hook1 = MockHook("before_fail", 400, execution_log)
        failing = FailingHook()
        hook2 = MockHook("after_fail", 600, execution_log)

        registry.register(hook1)
        registry.register(failing)
        registry.register(hook2)

        # Emit event - should not raise
        await manager.emit(HookEvent.REQUEST_STARTED, {"test": "data"})

        # Verify other hooks still executed
        assert execution_log == ["before_fail", "after_fail"]
