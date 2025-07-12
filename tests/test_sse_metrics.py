"""Tests for SSE metrics streaming functionality."""

import asyncio
import json
from datetime import datetime

import pytest

from ccproxy.metrics.collector import MetricsCollector
from ccproxy.metrics.exporters.sse import SSEMetricsExporter
from ccproxy.metrics.models import MetricType, RequestMetric
from ccproxy.metrics.storage.memory import InMemoryMetricsStorage


@pytest.mark.asyncio
@pytest.mark.unit
class TestSSEMetricsExporter:
    """Test suite for SSE metrics exporter."""

    async def test_sse_exporter_initialization(self) -> None:
        """Test SSE exporter can be initialized properly."""
        storage = InMemoryMetricsStorage()
        exporter = SSEMetricsExporter(storage=storage)

        assert exporter.storage == storage
        assert exporter.max_connections == 100
        assert not exporter._is_running
        assert len(exporter._connections) == 0

    async def test_sse_exporter_start_stop(self) -> None:
        """Test SSE exporter start and stop lifecycle."""
        storage = InMemoryMetricsStorage()
        await storage.initialize()

        exporter = SSEMetricsExporter(storage=storage)

        # Start exporter
        await exporter.start()
        assert exporter._is_running
        assert exporter._cleanup_task is not None

        # Stop exporter
        await exporter.stop()
        assert not exporter._is_running
        assert len(exporter._connections) == 0  # type: ignore[unreachable]

    async def test_sse_connection_creation(self) -> None:
        """Test creating SSE connections with filters."""
        storage = InMemoryMetricsStorage()
        await storage.initialize()

        exporter = SSEMetricsExporter(storage=storage)
        await exporter.start()

        try:
            # Create connection with filters
            async with exporter.create_connection(
                metric_types=[MetricType.REQUEST],
                user_id="test-user",
                session_id="test-session",
                subscription_types=["live"],
            ) as (connection_id, event_stream):
                # Verify connection was created
                assert connection_id in exporter._connections
                connection = exporter._connections[connection_id]

                assert connection.metric_types == {MetricType.REQUEST}
                assert connection.user_id == "test-user"
                assert connection.session_id == "test-session"
                assert connection.subscription_types == {"live"}

                # Connection should be automatically cleaned up when exiting context
                pass

            # Verify connection was cleaned up
            assert connection_id not in exporter._connections

        finally:
            await exporter.stop()

    async def test_metric_broadcasting(self) -> None:
        """Test broadcasting metrics to SSE connections."""
        storage = InMemoryMetricsStorage()
        await storage.initialize()

        exporter = SSEMetricsExporter(storage=storage)
        await exporter.start()

        try:
            # Create a test metric
            metric = RequestMetric(
                request_id="test-request",
                user_id="test-user",
                session_id="test-session",
                method="POST",
                path="/v1/messages",
                endpoint="/v1/messages",
                api_version="v1",
            )

            received_events = []

            async def collect_events() -> None:
                """Collect events from the stream."""
                async with exporter.create_connection(
                    metric_types=[MetricType.REQUEST],
                    user_id="test-user",
                    subscription_types=["live"],
                ) as (connection_id, event_stream):
                    # Collect events for a short time
                    try:
                        async for event in event_stream:
                            received_events.append(event)
                            # Break after receiving one metric event
                            if (
                                "metric" in event and len(received_events) >= 2
                            ):  # connected + metric
                                break
                    except TimeoutError:
                        pass

            # Start collecting events
            collect_task = asyncio.create_task(collect_events())

            # Give the connection time to be established
            await asyncio.sleep(0.1)

            # Broadcast the metric
            broadcast_count = await exporter.broadcast_metric(metric)
            assert broadcast_count == 1

            # Wait for events to be collected
            await asyncio.wait_for(collect_task, timeout=2.0)

            # Verify we received events
            assert len(received_events) >= 2

            # Check that we got a connected event
            connected_event = received_events[0]
            assert "event: connected" in connected_event

            # Check that we got a metric event
            metric_event = None
            for event in received_events:
                if "event: metric" in event:
                    metric_event = event
                    break

            assert metric_event is not None
            assert "test-request" in metric_event
            assert "test-user" in metric_event

        finally:
            await exporter.stop()

    async def test_metric_filtering(self) -> None:
        """Test that metrics are filtered correctly based on connection filters."""
        storage = InMemoryMetricsStorage()
        await storage.initialize()

        exporter = SSEMetricsExporter(storage=storage)
        await exporter.start()

        try:
            # Create metrics for different users
            metric1 = RequestMetric(
                request_id="request-1",
                user_id="user-1",
                session_id="session-1",
                method="POST",
                path="/v1/messages",
                endpoint="/v1/messages",
                api_version="v1",
            )

            metric2 = RequestMetric(
                request_id="request-2",
                user_id="user-2",
                session_id="session-2",
                method="GET",
                path="/v1/models",
                endpoint="/v1/models",
                api_version="v1",
            )

            # Broadcast metrics to all connections
            await exporter.broadcast_metric(metric1)
            await exporter.broadcast_metric(metric2)

            # Test connection that should only receive metric1
            async with exporter.create_connection(
                user_id="user-1",
                subscription_types=["live"],
            ) as (connection_id, event_stream):
                connection = exporter._connections[connection_id]

                # Test filtering logic
                assert connection.matches_filter(metric1)
                assert not connection.matches_filter(metric2)

        finally:
            await exporter.stop()

    async def test_connections_info(self) -> None:
        """Test getting information about active connections."""
        storage = InMemoryMetricsStorage()
        await storage.initialize()

        exporter = SSEMetricsExporter(storage=storage)
        await exporter.start()

        try:
            # Initially no connections
            info = await exporter.get_connections_info()
            assert info["total_connections"] == 0
            assert len(info["connections"]) == 0

            # Create a connection
            async with exporter.create_connection(
                metric_types=[MetricType.REQUEST],
                user_id="test-user",
            ) as (connection_id, event_stream):
                # Now should have one connection
                info = await exporter.get_connections_info()
                assert info["total_connections"] == 1
                assert len(info["connections"]) == 1

                conn_info = info["connections"][0]
                assert conn_info["id"] == connection_id
                assert conn_info["is_active"]
                assert conn_info["filters"]["user_id"] == "test-user"
                assert MetricType.REQUEST.value in conn_info["filters"]["metric_types"]

            # After exiting context, should be back to 0
            await asyncio.sleep(0.1)  # Give cleanup time
            info = await exporter.get_connections_info()
            assert info["total_connections"] == 0

        finally:
            await exporter.stop()

    async def test_health_check(self) -> None:
        """Test SSE exporter health check."""
        storage = InMemoryMetricsStorage()
        await storage.initialize()

        exporter = SSEMetricsExporter(storage=storage)

        # Should be unhealthy when not running
        assert not await exporter.health_check()

        # Should be healthy when running
        await exporter.start()
        assert await exporter.health_check()

        await exporter.stop()


@pytest.mark.asyncio
@pytest.mark.unit
class TestMetricsCollectorSSEIntegration:
    """Test integration between MetricsCollector and SSE exporter."""

    async def test_collector_with_sse_broadcasting(self) -> None:
        """Test that collector broadcasts metrics through SSE exporter."""
        storage = InMemoryMetricsStorage()
        await storage.initialize()

        # Create SSE exporter
        sse_exporter = SSEMetricsExporter(storage=storage)
        await sse_exporter.start()

        # Create collector with SSE exporter
        collector = MetricsCollector(
            storage=storage,
            sse_exporter=sse_exporter,
            enable_auto_flush=False,  # Disable for testing
        )
        await collector.start()

        try:
            received_events = []

            async def collect_events() -> None:
                """Collect events from SSE stream."""
                async with sse_exporter.create_connection(
                    subscription_types=["live"],
                ) as (connection_id, event_stream):
                    async for event in event_stream:
                        received_events.append(event)
                        if len(received_events) >= 2:  # connected + metric
                            break

            # Start collecting events
            collect_task = asyncio.create_task(collect_events())
            await asyncio.sleep(0.1)  # Let connection establish

            # Collect a metric through the collector
            await collector.collect_request_start(
                request_id="test-request",
                method="POST",
                path="/v1/messages",
                endpoint="/v1/messages",
                api_version="v1",
                user_id="test-user",
            )

            # Wait for events
            await asyncio.wait_for(collect_task, timeout=2.0)

            # Verify we received the broadcasted metric
            assert len(received_events) >= 2

            # Find the metric event
            metric_event = None
            for event in received_events:
                if "event: metric" in event and "test-request" in event:
                    metric_event = event
                    break

            assert metric_event is not None
            assert "request" in metric_event  # metric_type is lowercase in JSON

        finally:
            await collector.stop()
            await sse_exporter.stop()

    async def test_sse_events_exclude_none_values(self) -> None:
        """Test that SSE events exclude None values from JSON data."""
        storage = InMemoryMetricsStorage()
        await storage.initialize()

        exporter = SSEMetricsExporter(storage=storage)
        await exporter.start()

        try:
            # Create a metric with some None values
            metric = RequestMetric(
                request_id="test-request-clean",
                method="POST",
                path="/v1/messages",
                endpoint="/v1/messages",
                api_version="v1",
                user_id="test-user",
                # These will be None and should be excluded from SSE event
                session_id=None,
                client_ip=None,
                user_agent=None,
                content_length=None,
                content_type=None,
                model=None,
                provider=None,
                max_tokens=None,
                temperature=None,
                streaming=False,  # This should be included
            )

            received_events = []

            async def collect_events() -> None:
                """Collect events from the stream."""
                async with exporter.create_connection(
                    metric_types=[MetricType.REQUEST],
                    subscription_types=["live"],
                ) as (connection_id, event_stream):
                    async for event in event_stream:
                        received_events.append(event)
                        if len(received_events) >= 2:  # connected + metric
                            break

            # Start collecting events
            collect_task = asyncio.create_task(collect_events())
            await asyncio.sleep(0.1)

            # Broadcast the metric
            await exporter.broadcast_metric(metric)
            await asyncio.wait_for(collect_task, timeout=2.0)

            # Find the metric event
            metric_event = None
            for event in received_events:
                if "event: metric" in event:
                    metric_event = event
                    break

            assert metric_event is not None

            # Extract JSON data from the SSE event
            lines = metric_event.strip().split("\n")
            data_line = None
            for line in lines:
                if line.startswith("data: "):
                    data_line = line[6:]  # Remove 'data: ' prefix
                    break

            assert data_line is not None
            metric_data = json.loads(data_line)

            # Verify required fields are present
            assert metric_data["request_id"] == "test-request-clean"
            assert metric_data["method"] == "POST"
            assert metric_data["path"] == "/v1/messages"
            assert metric_data["user_id"] == "test-user"
            assert metric_data["streaming"] is False

            # Verify None values are excluded from the JSON
            assert "session_id" not in metric_data
            assert "client_ip" not in metric_data
            assert "user_agent" not in metric_data
            assert "content_length" not in metric_data
            assert "content_type" not in metric_data
            assert "model" not in metric_data
            assert "provider" not in metric_data
            assert "max_tokens" not in metric_data
            assert "temperature" not in metric_data

        finally:
            await exporter.stop()

    async def test_sse_connection_event_excludes_none_filters(self) -> None:
        """Test that SSE connection events exclude None values in filters."""
        storage = InMemoryMetricsStorage()
        await storage.initialize()

        exporter = SSEMetricsExporter(storage=storage)
        await exporter.start()

        try:
            received_events = []

            async def collect_events() -> None:
                """Collect the connection event."""
                async with exporter.create_connection(
                    metric_types=[MetricType.REQUEST],
                    user_id="test-user",
                    # session_id is not provided (None)
                    subscription_types=["live"],
                ) as (connection_id, event_stream):
                    async for event in event_stream:
                        received_events.append(event)
                        if len(received_events) >= 1:  # Just the connected event
                            break

            collect_task = asyncio.create_task(collect_events())
            await asyncio.wait_for(collect_task, timeout=1.0)

            # Verify we got the connected event
            assert len(received_events) >= 1
            connected_event = received_events[0]
            assert "event: connected" in connected_event

            # Extract JSON data from the connected event
            lines = connected_event.strip().split("\n")
            data_line = None
            for line in lines:
                if line.startswith("data: "):
                    data_line = line[6:]
                    break

            assert data_line is not None
            connection_data = json.loads(data_line)

            # Verify filters object excludes None values
            filters = connection_data["filters"]
            assert "metric_types" in filters
            assert "user_id" in filters
            assert "subscription_types" in filters
            assert filters["user_id"] == "test-user"

            # session_id should not be present since it was None
            assert "session_id" not in filters

        finally:
            await exporter.stop()

    async def test_sse_connections_info_excludes_none(self) -> None:
        """Test that connections info excludes None values."""
        storage = InMemoryMetricsStorage()
        await storage.initialize()

        exporter = SSEMetricsExporter(storage=storage)
        await exporter.start()

        try:
            # Create a connection with partial filters
            async with exporter.create_connection(
                metric_types=[MetricType.REQUEST],
                user_id="test-user",
                # session_id is None and should be excluded
                subscription_types=["live"],
            ) as (connection_id, event_stream):
                # Get connections info
                info = await exporter.get_connections_info()

                assert info["total_connections"] == 1
                conn_info = info["connections"][0]

                # Verify filters exclude None values
                filters = conn_info["filters"]
                assert "metric_types" in filters
                assert "user_id" in filters
                assert "subscription_types" in filters

                # session_id should not be present
                assert "session_id" not in filters

        finally:
            await exporter.stop()
