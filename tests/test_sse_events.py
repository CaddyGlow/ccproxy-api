"""
Unit tests for SSE event manager functionality.

Tests the SSE event manager's connection handling, event broadcasting,
and error handling capabilities.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from typing import Any

import pytest

from ccproxy.observability.sse_events import (
    SSEEventManager,
    cleanup_sse_manager,
    emit_sse_event,
    get_sse_manager,
)


class TestSSEEventManager:
    """Test SSE event manager functionality."""

    @pytest.fixture
    def sse_manager(self) -> SSEEventManager:
        """Create SSE manager for testing."""
        return SSEEventManager(max_queue_size=10)

    async def test_connection_initialization(
        self, sse_manager: SSEEventManager
    ) -> None:
        """Test SSE connection initialization."""
        connection_id = "test-connection"
        events = []

        async def collect_events() -> None:
            async for event in sse_manager.add_connection(connection_id):
                events.append(event)
                # Stop after connection event
                if len(events) >= 1:
                    break

        # Start connection in background
        task = asyncio.create_task(collect_events())

        # Wait briefly for connection to establish
        await asyncio.sleep(0.1)

        # Cancel connection
        task.cancel()

        with contextlib.suppress(asyncio.CancelledError):
            await task

        # Check connection event was sent
        assert len(events) == 1
        event_data = json.loads(events[0].replace("data: ", "").strip())
        assert event_data["type"] == "connection"
        assert event_data["connection_id"] == connection_id
        assert "timestamp" in event_data

    async def test_event_broadcasting(self, sse_manager: SSEEventManager) -> None:
        """Test event broadcasting to multiple connections."""
        connection_ids = ["conn1", "conn2"]
        all_events: dict[str, list[str]] = {conn_id: [] for conn_id in connection_ids}

        async def collect_events(connection_id: str) -> None:
            async for event in sse_manager.add_connection(connection_id):
                all_events[connection_id].append(event)
                # Stop after receiving test event
                if len(all_events[connection_id]) >= 2:  # connection + test event
                    break

        # Start connections
        tasks = [
            asyncio.create_task(collect_events(conn_id)) for conn_id in connection_ids
        ]

        # Wait for connections to establish
        await asyncio.sleep(0.1)

        # Broadcast test event
        test_event = {
            "request_id": "test-123",
            "method": "POST",
            "path": "/test",
        }
        await sse_manager.emit_event("request_start", test_event)

        # Wait for event propagation
        await asyncio.sleep(0.1)

        # Cancel connections
        for task in tasks:
            task.cancel()

        # Wait for cleanup
        await asyncio.gather(*tasks, return_exceptions=True)

        # Check both connections received the event
        for conn_id in connection_ids:
            assert len(all_events[conn_id]) >= 2

            # Check connection event
            connection_event = json.loads(
                all_events[conn_id][0].replace("data: ", "").strip()
            )
            assert connection_event["type"] == "connection"

            # Check test event
            test_event_data = json.loads(
                all_events[conn_id][1].replace("data: ", "").strip()
            )
            assert test_event_data["type"] == "request_start"
            assert test_event_data["data"]["request_id"] == "test-123"
            assert test_event_data["data"]["method"] == "POST"
            assert test_event_data["data"]["path"] == "/test"

    async def test_queue_overflow_handling(self, sse_manager: SSEEventManager) -> None:
        """Test queue overflow handling with bounded queues."""
        connection_id = "overflow-test"
        events = []

        async def slow_consumer() -> None:
            async for event in sse_manager.add_connection(connection_id):
                events.append(event)
                # Simulate slow consumer
                await asyncio.sleep(0.01)
                if len(events) >= 15:  # Stop after collecting some events
                    break

        # Start slow consumer
        task = asyncio.create_task(slow_consumer())

        # Wait for connection to establish
        await asyncio.sleep(0.1)

        # Flood with events (more than queue size)
        for i in range(15):
            await sse_manager.emit_event("request_start", {"request_id": f"req-{i}"})

        # Wait for processing
        await asyncio.sleep(0.2)

        # Cancel connection
        task.cancel()

        with contextlib.suppress(asyncio.CancelledError):
            await task

        # Check that overflow event was sent
        overflow_found = False
        for event in events:
            if "overflow" in event:
                event_data = json.loads(event.replace("data: ", "").strip())
                if event_data.get("type") == "overflow":
                    overflow_found = True
                    break

        assert overflow_found, "Overflow event should have been sent"

    async def test_connection_cleanup(self, sse_manager: SSEEventManager) -> None:
        """Test connection cleanup on disconnect."""
        connection_id = "cleanup-test"

        # Check initial connection count
        initial_count = await sse_manager.get_connection_count()
        assert initial_count == 0

        async def persistent_connection() -> None:
            async for _event in sse_manager.add_connection(connection_id):
                # Keep connection alive
                await asyncio.sleep(0.01)

        # Start connection
        task = asyncio.create_task(persistent_connection())
        await asyncio.sleep(0.1)  # Let connection establish

        # Check connection was added
        active_count = await sse_manager.get_connection_count()
        assert active_count == 1

        # Cancel connection
        task.cancel()

        with contextlib.suppress(asyncio.CancelledError):
            await task

        # Wait for cleanup
        await asyncio.sleep(0.1)

        # Check connection was removed
        final_count = await sse_manager.get_connection_count()
        assert final_count == 0

    async def test_disconnect_all(self, sse_manager: SSEEventManager) -> None:
        """Test disconnecting all connections."""
        connection_ids = ["disc1", "disc2", "disc3"]
        tasks = []

        async def persistent_connection(connection_id: str) -> None:
            async for _event in sse_manager.add_connection(connection_id):
                # Keep connection alive
                await asyncio.sleep(0.01)

        # Start multiple connections
        for conn_id in connection_ids:
            task = asyncio.create_task(persistent_connection(conn_id))
            tasks.append(task)

        # Wait for connections to establish
        await asyncio.sleep(0.1)

        # Check all connections are active
        active_count = await sse_manager.get_connection_count()
        assert active_count == len(connection_ids)

        # Disconnect all
        await sse_manager.disconnect_all()

        # Wait for cleanup
        await asyncio.sleep(0.1)

        # Check all connections are gone
        final_count = await sse_manager.get_connection_count()
        assert final_count == 0

        # Cancel remaining tasks
        for task in tasks:
            task.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)

    async def test_json_serialization(self, sse_manager: SSEEventManager) -> None:
        """Test JSON serialization of events."""
        connection_id = "json-test"
        events = []

        async def collect_events() -> None:
            async for event in sse_manager.add_connection(connection_id):
                events.append(event)
                if len(events) >= 2:  # connection + test event
                    break

        # Start connection
        task = asyncio.create_task(collect_events())
        await asyncio.sleep(0.1)

        # Send event with datetime (should be serialized)
        from datetime import datetime

        test_event = {
            "request_id": "datetime-test",
            "timestamp": datetime.now(),
            "data": {"nested": "value"},
        }
        await sse_manager.emit_event("test_event", test_event)

        # Wait for event
        await asyncio.sleep(0.1)

        # Cancel connection
        task.cancel()

        with contextlib.suppress(asyncio.CancelledError):
            await task

        # Check event was properly serialized
        assert len(events) >= 2
        test_event_raw = events[1]
        assert test_event_raw.startswith("data: ")

        # Should be valid JSON
        event_data = json.loads(test_event_raw.replace("data: ", "").strip())
        assert event_data["type"] == "test_event"
        assert event_data["data"]["request_id"] == "datetime-test"
        assert isinstance(event_data["data"]["timestamp"], str)  # datetime serialized

    async def test_connection_info(self, sse_manager: SSEEventManager) -> None:
        """Test connection info retrieval."""
        # Check initial state
        info = await sse_manager.get_connection_info()
        assert info["active_connections"] == 0
        assert info["max_queue_size"] == 10
        assert info["connection_ids"] == []

        connection_id = "info-test"

        async def test_connection() -> None:
            async for _event in sse_manager.add_connection(connection_id):
                # Keep connection alive briefly
                await asyncio.sleep(0.1)
                break

        # Start connection
        task = asyncio.create_task(test_connection())
        await asyncio.sleep(0.05)  # Let connection establish

        # Check connection info
        info = await sse_manager.get_connection_info()
        assert info["active_connections"] == 1
        assert connection_id in info["connection_ids"]

        # Cancel connection
        task.cancel()

        with contextlib.suppress(asyncio.CancelledError):
            await task

        # Wait for cleanup
        await asyncio.sleep(0.1)

        # Check final state
        info = await sse_manager.get_connection_info()
        assert info["active_connections"] == 0


class TestSSEGlobalFunctions:
    """Test global SSE functions."""

    async def test_get_sse_manager(self) -> None:
        """Test global SSE manager creation."""
        manager1 = get_sse_manager()
        manager2 = get_sse_manager()

        # Should return same instance
        assert manager1 is manager2

        # Should be functional
        count = await manager1.get_connection_count()
        assert count == 0

    async def test_emit_sse_event(self) -> None:
        """Test global emit_sse_event function."""
        manager = get_sse_manager()
        events = []

        async def collect_events() -> None:
            async for event in manager.add_connection("global-test"):
                events.append(event)
                if len(events) >= 2:  # connection + test event
                    break

        # Start connection
        task = asyncio.create_task(collect_events())
        await asyncio.sleep(0.1)

        # Use global emit function
        await emit_sse_event("request_complete", {"request_id": "global-123"})

        # Wait for event
        await asyncio.sleep(0.1)

        # Cancel connection
        task.cancel()

        with contextlib.suppress(asyncio.CancelledError):
            await task

        # Check event was received
        assert len(events) >= 2
        test_event = json.loads(events[1].replace("data: ", "").strip())
        assert test_event["type"] == "request_complete"
        assert test_event["data"]["request_id"] == "global-123"

    async def test_cleanup_sse_manager(self) -> None:
        """Test global SSE manager cleanup."""
        manager = get_sse_manager()

        # Create connection
        async def test_connection() -> None:
            async for _event in manager.add_connection("cleanup-test"):
                await asyncio.sleep(0.1)

        task = asyncio.create_task(test_connection())
        await asyncio.sleep(0.05)

        # Check connection exists
        count = await manager.get_connection_count()
        assert count == 1

        # Cleanup manager
        await cleanup_sse_manager()

        # Check connections are cleaned up
        new_manager = get_sse_manager()
        count = await new_manager.get_connection_count()
        assert count == 0

        # Cancel remaining task
        task.cancel()

        with contextlib.suppress(asyncio.CancelledError):
            await task


class TestSSEErrorHandling:
    """Test SSE error handling scenarios."""

    @pytest.fixture
    def sse_manager(self) -> SSEEventManager:
        """Create SSE manager for testing."""
        return SSEEventManager(max_queue_size=10)

    async def test_emit_event_with_no_connections(self) -> None:
        """Test emitting events when no connections exist."""
        manager = SSEEventManager()

        # Should not raise exception
        await manager.emit_event("test_event", {"data": "test"})

        # Connection count should still be 0
        count = await manager.get_connection_count()
        assert count == 0

    async def test_emit_sse_event_error_handling(self) -> None:
        """Test error handling in emit_sse_event function."""
        # This should not raise an exception even if something goes wrong
        await emit_sse_event("test_event", {"data": "test"})

        # Function should handle errors gracefully
        assert True  # If we get here, no exception was raised

    async def test_connection_with_invalid_json(
        self, sse_manager: SSEEventManager
    ) -> None:
        """Test handling of events that can't be JSON serialized."""
        connection_id = "invalid-json-test"
        events = []

        async def collect_events() -> None:
            async for event in sse_manager.add_connection(connection_id):
                events.append(event)
                if len(events) >= 3:  # connection + test event + error event
                    break

        # Start connection
        task = asyncio.create_task(collect_events())
        await asyncio.sleep(0.1)

        # Create non-serializable object
        class NonSerializable:
            def __str__(self) -> str:
                return "non-serializable"

        # Try to emit event with non-serializable data
        await sse_manager.emit_event("test_event", {"data": NonSerializable()})

        # Wait for event processing
        await asyncio.sleep(0.1)

        # Cancel connection
        task.cancel()

        with contextlib.suppress(asyncio.CancelledError):
            await task

        # Should have received connection event and error event
        assert len(events) >= 2

        # Check if error event was sent
        error_found = False
        for event in events[1:]:  # Skip connection event
            if "error" in event:
                event_data = json.loads(event.replace("data: ", "").strip())
                if event_data.get("type") == "error":
                    error_found = True
                    break

        assert error_found, (
            "Error event should have been sent for non-serializable data"
        )
