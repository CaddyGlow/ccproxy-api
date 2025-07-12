"""Tests for metrics collection and storage functionality.

This module tests the complete metrics pipeline from collection to retrieval,
including middleware, collector, and storage components.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, cast
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from fastapi import Request, Response
from fastapi.testclient import TestClient

from ccproxy.metrics.collector import MetricsCollector
from ccproxy.metrics.exporters.json_api import JsonApiExporter
from ccproxy.metrics.middleware import MetricsMiddleware
from ccproxy.metrics.models import (
    CostMetric,
    ErrorMetric,
    LatencyMetric,
    MetricRecord,
    MetricType,
    RequestMetric,
    ResponseMetric,
    UsageMetric,
)
from ccproxy.metrics.storage.memory import InMemoryMetricsStorage


class TestInMemoryMetricsStorage:
    """Test the in-memory metrics storage implementation."""

    @pytest.mark.unit
    async def test_storage_initialization(
        self, metrics_storage: InMemoryMetricsStorage
    ) -> None:
        """Test storage initialization and cleanup."""
        # Storage should be initialized
        await metrics_storage.initialize()

        # Check initial state
        health = await metrics_storage.health_check()
        assert health["status"] == "healthy"
        assert health["total_metrics"] == 0

        # Close storage
        await metrics_storage.close()

    @pytest.mark.unit
    async def test_store_single_metric(
        self, metrics_storage: InMemoryMetricsStorage
    ) -> None:
        """Test storing a single metric record."""
        await metrics_storage.initialize()

        # Create a test metric
        metric = RequestMetric(
            request_id="test-req-123",
            user_id="test-user",
            session_id="test-session",
            method="POST",
            path="/v1/messages",
            endpoint="/v1/messages",
            api_version="v1",
            model="claude-3-5-sonnet-20241022",
            provider="anthropic",
        )

        # Store the metric
        success = await metrics_storage.store_metric(metric)
        assert success is True

        # Verify storage
        stored_metric = await metrics_storage.get_metric(metric.id)
        assert stored_metric is not None
        assert stored_metric.request_id == "test-req-123"
        assert stored_metric.user_id == "test-user"
        assert cast(RequestMetric, stored_metric).model == "claude-3-5-sonnet-20241022"

        await metrics_storage.close()

    @pytest.mark.unit
    async def test_store_multiple_metrics(
        self, metrics_storage: InMemoryMetricsStorage
    ) -> None:
        """Test storing multiple metrics at once."""
        await metrics_storage.initialize()

        # Create test metrics
        metrics: list[MetricRecord] = [
            RequestMetric(
                request_id=f"test-req-{i}",
                user_id=f"user-{i % 3}",  # 3 different users
                method="POST",
                path="/v1/messages",
                endpoint="/v1/messages",
                api_version="v1",
            )
            for i in range(5)
        ]

        # Store multiple metrics
        stored_count = await metrics_storage.store_metrics(metrics)
        assert stored_count == 5

        # Verify all were stored
        all_metrics = await metrics_storage.get_metrics()
        assert len(all_metrics) == 5

        # Check user filtering
        user_0_metrics = await metrics_storage.get_metrics(user_id="user-0")
        assert len(user_0_metrics) == 2  # users 0 and 3

        await metrics_storage.close()

    @pytest.mark.unit
    async def test_metrics_filtering(
        self, metrics_storage: InMemoryMetricsStorage
    ) -> None:
        """Test filtering metrics by various criteria."""
        await metrics_storage.initialize()

        now = datetime.utcnow()
        past = now - timedelta(hours=1)
        future = now + timedelta(hours=1)

        # Create metrics with different timestamps and types
        metrics = [
            RequestMetric(
                request_id="req-1",
                user_id="user-1",
                session_id="session-1",
                method="POST",
                path="/v1/messages",
                endpoint="/v1/messages",
                api_version="v1",
                timestamp=past,
            ),
            ResponseMetric(
                request_id="req-1",
                user_id="user-1",
                session_id="session-1",
                status_code=200,
                response_time_ms=100.0,
                timestamp=now,
            ),
            ErrorMetric(
                request_id="req-2",
                user_id="user-2",
                error_type="ValidationError",
                error_message="Invalid input",
                timestamp=future,
            ),
        ]

        await metrics_storage.store_metrics(metrics)

        # Test time filtering
        past_metrics = await metrics_storage.get_metrics(
            start_time=past - timedelta(minutes=30),
            end_time=past + timedelta(minutes=30),
        )
        assert len(past_metrics) == 1
        assert isinstance(past_metrics[0], RequestMetric)

        # Test type filtering
        response_metrics = await metrics_storage.get_metrics(
            metric_type=MetricType.RESPONSE
        )
        assert len(response_metrics) == 1
        assert isinstance(response_metrics[0], ResponseMetric)

        # Test user filtering
        user_1_metrics = await metrics_storage.get_metrics(user_id="user-1")
        assert len(user_1_metrics) == 2  # Request and Response

        # Test request filtering
        req_1_metrics = await metrics_storage.get_metrics(request_id="req-1")
        assert len(req_1_metrics) == 2  # Request and Response

        await metrics_storage.close()

    @pytest.mark.unit
    async def test_metrics_counting(
        self, metrics_storage: InMemoryMetricsStorage
    ) -> None:
        """Test counting metrics with filters."""
        await metrics_storage.initialize()

        # Create test metrics
        metrics: list[MetricRecord] = [
            RequestMetric(
                request_id=f"req-{i}",
                user_id="test-user",
                method="POST",
                path="/v1/messages",
                endpoint="/v1/messages",
                api_version="v1",
            )
            for i in range(10)
        ]

        await metrics_storage.store_metrics(metrics)

        # Test total count
        total_count = await metrics_storage.count_metrics()
        assert total_count == 10

        # Test filtered count
        user_count = await metrics_storage.count_metrics(user_id="test-user")
        assert user_count == 10

        # Test type filtering
        request_count = await metrics_storage.count_metrics(
            metric_type=MetricType.REQUEST
        )
        assert request_count == 10

        response_count = await metrics_storage.count_metrics(
            metric_type=MetricType.RESPONSE
        )
        assert response_count == 0

        await metrics_storage.close()

    @pytest.mark.unit
    async def test_metrics_deletion(
        self, metrics_storage: InMemoryMetricsStorage
    ) -> None:
        """Test deleting metrics with filters."""
        await metrics_storage.initialize()

        # Create test metrics with different users
        metrics: list[MetricRecord] = [
            RequestMetric(
                request_id=f"req-{i}",
                user_id=f"user-{i % 2}",  # user-0 and user-1
                method="POST",
                path="/v1/messages",
                endpoint="/v1/messages",
                api_version="v1",
            )
            for i in range(10)
        ]

        await metrics_storage.store_metrics(metrics)

        # Delete metrics for user-0
        deleted_count = await metrics_storage.delete_metrics(user_id="user-0")
        assert deleted_count == 5

        # Verify deletion
        remaining_count = await metrics_storage.count_metrics()
        assert remaining_count == 5

        # Verify only user-1 metrics remain
        user_1_count = await metrics_storage.count_metrics(user_id="user-1")
        assert user_1_count == 5

        user_0_count = await metrics_storage.count_metrics(user_id="user-0")
        assert user_0_count == 0

        await metrics_storage.close()

    @pytest.mark.unit
    async def test_auto_cleanup(self) -> None:
        """Test automatic cleanup of old metrics."""
        # Create storage with small max_metrics
        storage = InMemoryMetricsStorage(max_metrics=5, auto_cleanup=True)
        await storage.initialize()

        # Create more metrics than the limit
        metrics: list[MetricRecord] = [
            RequestMetric(
                request_id=f"req-{i}",
                method="POST",
                path="/v1/messages",
                endpoint="/v1/messages",
                api_version="v1",
                timestamp=datetime.utcnow()
                + timedelta(seconds=i),  # Different timestamps
            )
            for i in range(10)
        ]

        # Store all metrics (should trigger cleanup)
        await storage.store_metrics(metrics)

        # Should only have max_metrics number of metrics
        total_count = await storage.count_metrics()
        assert total_count == 5

        # Should keep the most recent ones
        all_metrics = await storage.get_metrics(order_by="timestamp", order_desc=True)
        assert len(all_metrics) == 5

        # The most recent metric should be req-9
        assert all_metrics[0].request_id == "req-9"

        await storage.close()


class TestMetricsCollector:
    """Test the metrics collector functionality."""

    @pytest.mark.unit
    async def test_collector_lifecycle(
        self, metrics_storage: InMemoryMetricsStorage
    ) -> None:
        """Test collector start and stop lifecycle."""
        collector = MetricsCollector(
            storage=metrics_storage,
            buffer_size=10,
            flush_interval=1.0,
            enable_auto_flush=False,  # Disable for test control
        )

        # Start collector
        await collector.start()
        assert collector._is_running is True

        # Stop collector
        await collector.stop()
        assert collector._is_running is False

    @pytest.mark.unit
    async def test_request_metric_collection(
        self, metrics_storage: InMemoryMetricsStorage
    ) -> None:
        """Test collecting request metrics."""
        collector = MetricsCollector(storage=metrics_storage, enable_auto_flush=False)
        await collector.start()

        # Collect request start
        request_metric = await collector.collect_request_start(
            request_id="test-req-123",
            method="POST",
            path="/v1/messages",
            endpoint="/v1/messages",
            api_version="v1",
            user_id="test-user",
            session_id="test-session",
            model="claude-3-5-sonnet-20241022",
            provider="anthropic",
        )

        # Verify metric creation
        assert request_metric.request_id == "test-req-123"
        assert request_metric.method == "POST"
        assert request_metric.model == "claude-3-5-sonnet-20241022"

        # Verify it's in the collector's tracking
        assert "test-req-123" in collector._active_requests

        # Flush to storage and verify
        flushed_count = await collector.flush()
        assert flushed_count == 1

        stored_metrics = await metrics_storage.get_metrics()
        assert len(stored_metrics) == 1
        assert isinstance(stored_metrics[0], RequestMetric)

        await collector.stop()

    @pytest.mark.unit
    async def test_response_metric_collection(
        self, metrics_storage: InMemoryMetricsStorage
    ) -> None:
        """Test collecting response metrics."""
        collector = MetricsCollector(storage=metrics_storage, enable_auto_flush=False)
        await collector.start()

        # First collect request start
        await collector.collect_request_start(
            request_id="test-req-123",
            method="POST",
            path="/v1/messages",
            endpoint="/v1/messages",
            api_version="v1",
            user_id="test-user",
        )

        # Small delay to ensure different timestamps
        await asyncio.sleep(0.01)

        # Collect response
        response_metric = await collector.collect_response(
            request_id="test-req-123",
            status_code=200,
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=10,
            cache_write_tokens=5,
        )

        # Verify response metric
        assert response_metric.request_id == "test-req-123"
        assert response_metric.status_code == 200
        assert response_metric.input_tokens == 100
        assert response_metric.output_tokens == 50
        assert (
            response_metric.response_time_ms > 0
        )  # Should have calculated response time
        assert response_metric.user_id == "test-user"  # Should inherit from request

        # Flush and verify storage
        await collector.flush()
        stored_metrics = await metrics_storage.get_metrics()
        assert len(stored_metrics) == 2  # Request + Response

        response_metrics = await metrics_storage.get_metrics(
            metric_type=MetricType.RESPONSE
        )
        assert len(response_metrics) == 1

        await collector.stop()

    @pytest.mark.unit
    async def test_error_metric_collection(
        self, metrics_storage: InMemoryMetricsStorage
    ) -> None:
        """Test collecting error metrics."""
        collector = MetricsCollector(storage=metrics_storage, enable_auto_flush=False)
        await collector.start()

        # Start a request for context
        await collector.collect_request_start(
            request_id="test-req-123",
            method="POST",
            path="/v1/messages",
            endpoint="/v1/messages",
            api_version="v1",
            user_id="test-user",
        )

        # Collect error
        error_metric = await collector.collect_error(
            request_id="test-req-123",
            error_type="ValidationError",
            error_code="invalid_model",
            error_message="The specified model is not supported",
            stack_trace="Traceback...",
        )

        # Verify error metric
        assert error_metric.request_id == "test-req-123"
        assert error_metric.error_type == "ValidationError"
        assert error_metric.error_code == "invalid_model"
        assert error_metric.user_id == "test-user"  # Should inherit from request

        # Flush and verify storage
        await collector.flush()
        error_metrics = await metrics_storage.get_metrics(metric_type=MetricType.ERROR)
        assert len(error_metrics) == 1

        await collector.stop()

    @pytest.mark.unit
    async def test_latency_metric_collection(
        self, metrics_storage: InMemoryMetricsStorage
    ) -> None:
        """Test collecting latency metrics."""
        collector = MetricsCollector(storage=metrics_storage, enable_auto_flush=False)
        await collector.start()

        # Start a request for context
        await collector.collect_request_start(
            request_id="test-req-123",
            method="POST",
            path="/v1/messages",
            endpoint="/v1/messages",
            api_version="v1",
            user_id="test-user",
        )

        # Collect latency metrics
        latency_metric = await collector.collect_latency(
            request_id="test-req-123",
            total_latency_ms=250.0,
            request_processing_ms=200.0,
            model_latency_ms=180.0,
            network_latency_ms=50.0,
        )

        # Verify latency metric
        assert latency_metric.request_id == "test-req-123"
        assert latency_metric.total_latency_ms == 250.0
        assert latency_metric.user_id == "test-user"

        # Flush and verify storage
        await collector.flush()
        latency_metrics = await metrics_storage.get_metrics(
            metric_type=MetricType.LATENCY
        )
        assert len(latency_metrics) == 1

        await collector.stop()

    @pytest.mark.unit
    async def test_usage_metric_collection(
        self, metrics_storage: InMemoryMetricsStorage
    ) -> None:
        """Test collecting usage metrics."""
        collector = MetricsCollector(storage=metrics_storage, enable_auto_flush=False)
        await collector.start()

        # Collect usage metrics
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(hours=1)

        usage_metric = await collector.collect_usage(
            window_start=start_time,
            window_end=end_time,
            aggregation_level="hourly",
            request_count=100,
            token_count=5000,
        )

        # Verify usage metric
        assert usage_metric.window_start == start_time
        assert usage_metric.window_end == end_time
        assert usage_metric.request_count == 100
        assert usage_metric.aggregation_level == "hourly"

        # Flush and verify storage
        await collector.flush()
        usage_metrics = await metrics_storage.get_metrics(metric_type=MetricType.USAGE)
        assert len(usage_metrics) == 1

        await collector.stop()

    @pytest.mark.unit
    async def test_buffer_auto_flush(
        self, metrics_storage: InMemoryMetricsStorage
    ) -> None:
        """Test automatic buffer flushing when buffer is full."""
        collector = MetricsCollector(
            storage=metrics_storage,
            buffer_size=3,  # Small buffer for testing
            enable_auto_flush=False,
        )
        await collector.start()

        # Add metrics to fill the buffer
        for i in range(5):  # More than buffer size
            await collector.collect_request_start(
                request_id=f"req-{i}",
                method="POST",
                path="/v1/messages",
                endpoint="/v1/messages",
                api_version="v1",
            )

        # Give auto-flush tasks time to complete
        await asyncio.sleep(0.1)

        # Should have flushed to storage
        stored_metrics = await metrics_storage.get_metrics()
        assert len(stored_metrics) >= 3  # At least one flush occurred

        await collector.stop()

    @pytest.mark.unit
    async def test_request_context_manager(
        self, metrics_storage: InMemoryMetricsStorage
    ) -> None:
        """Test the request context manager for complete lifecycle tracking."""
        collector = MetricsCollector(storage=metrics_storage, enable_auto_flush=False)
        await collector.start()

        request_id = "context-test-123"

        # Use context manager
        async with collector.request_context(
            request_id=request_id,
            method="POST",
            path="/v1/messages",
            endpoint="/v1/messages",
            api_version="v1",
            user_id="test-user",
        ) as request_metric:
            # Verify the request was tracked
            assert request_metric.request_id == request_id
            assert request_id in collector._active_requests

            # Collect response within context
            await collector.collect_response(
                request_id=request_id,
                status_code=200,
                input_tokens=50,
                output_tokens=25,
            )

        # After context, request should be finished
        assert request_id not in collector._active_requests

        await collector.flush()
        stored_metrics = await metrics_storage.get_metrics()
        assert len(stored_metrics) == 2  # Request + Response

        await collector.stop()

    @pytest.mark.unit
    async def test_metrics_summary_generation(
        self, metrics_storage: InMemoryMetricsStorage
    ) -> None:
        """Test generating metrics summary from collected data."""
        collector = MetricsCollector(storage=metrics_storage, enable_auto_flush=False)
        await collector.start()

        # Collect various metrics
        for i in range(3):
            request_id = f"req-{i}"

            # Request
            await collector.collect_request_start(
                request_id=request_id,
                method="POST",
                path="/v1/messages",
                endpoint="/v1/messages",
                api_version="v1",
                user_id=f"user-{i % 2}",  # 2 different users
                model="claude-3-5-sonnet-20241022",
            )

            # Response (mix of success and failure)
            status_code = 200 if i < 2 else 400
            await collector.collect_response(
                request_id=request_id,
                status_code=status_code,
                input_tokens=100,
                output_tokens=50 if status_code == 200 else 0,
            )

            # Error for failed request
            if status_code == 400:
                await collector.collect_error(
                    request_id=request_id,
                    error_type="ValidationError",
                    error_message="Invalid request",
                )

        # Flush all metrics
        await collector.flush()

        # Get summary
        summary = await collector.get_summary()

        # Verify summary calculations
        assert summary.total_requests == 3
        assert summary.successful_requests == 2
        assert summary.failed_requests == 1
        assert abs(summary.error_rate - (1 / 3)) < 0.01  # Approximately 0.33
        assert summary.total_input_tokens == 300  # 3 * 100
        assert summary.total_output_tokens == 100  # 2 * 50 (only successful requests)
        assert summary.unique_users == 2
        assert "claude-3-5-sonnet-20241022" in summary.model_usage
        assert summary.model_usage["claude-3-5-sonnet-20241022"] == 3

        await collector.stop()


class TestMetricsMiddleware:
    """Test the metrics middleware integration."""

    @pytest.mark.unit
    async def test_middleware_request_collection(
        self, metrics_storage: InMemoryMetricsStorage
    ) -> None:
        """Test that middleware collects request metrics."""
        collector = MetricsCollector(storage=metrics_storage, enable_auto_flush=False)
        await collector.start()

        # Create mock middleware
        middleware = MetricsMiddleware(
            app=Mock(),
            collector=collector,
            excluded_paths=["/health"],
        )

        # Create mock request
        request = Mock(spec=Request)
        request.method = "POST"
        request.url.path = "/v1/messages"
        request.query_params = {}
        request.headers = {
            "content-type": "application/json",
            "user-agent": "test-client",
        }
        request.cookies = {}
        request.state = Mock()
        # Ensure session_id is None instead of a Mock object
        request.state.session_id = None
        # Mock the client to provide proper IP address
        request.client = Mock()
        request.client.host = "127.0.0.1"

        # Mock the route handler
        async def mock_handler(req: Request) -> Response:
            return Response(content="OK", status_code=200)

        # Process request through middleware
        response = await middleware.dispatch(request, mock_handler)

        # Verify response
        assert response.status_code == 200

        # Flush collector and check metrics
        await collector.flush()
        stored_metrics = await metrics_storage.get_metrics()

        # Should have request, response, and latency metrics
        assert len(stored_metrics) >= 2

        request_metrics = await metrics_storage.get_metrics(
            metric_type=MetricType.REQUEST
        )
        response_metrics = await metrics_storage.get_metrics(
            metric_type=MetricType.RESPONSE
        )

        assert len(request_metrics) == 1
        assert len(response_metrics) == 1

        # Verify request details
        req_metric = cast(RequestMetric, request_metrics[0])
        assert req_metric.method == "POST"
        assert req_metric.path == "/v1/messages"

        # Verify response details
        resp_metric = cast(ResponseMetric, response_metrics[0])
        assert resp_metric.status_code == 200
        assert resp_metric.response_time_ms > 0

        await collector.stop()

    @pytest.mark.unit
    async def test_middleware_excludes_paths(
        self, metrics_storage: InMemoryMetricsStorage
    ) -> None:
        """Test that middleware excludes specified paths."""
        collector = MetricsCollector(storage=metrics_storage, enable_auto_flush=False)
        await collector.start()

        middleware = MetricsMiddleware(
            app=Mock(),
            collector=collector,
            excluded_paths=["/health", "/metrics"],
        )

        # Create mock request for excluded path
        request = Mock(spec=Request)
        request.url.path = "/health"

        async def mock_handler(req: Request) -> Response:
            return Response(content="OK", status_code=200)

        # Process request
        response = await middleware.dispatch(request, mock_handler)
        assert response.status_code == 200

        # Should not have collected any metrics
        await collector.flush()
        stored_metrics = await metrics_storage.get_metrics()
        assert len(stored_metrics) == 0

        await collector.stop()

    @pytest.mark.unit
    async def test_middleware_error_handling(
        self, metrics_storage: InMemoryMetricsStorage
    ) -> None:
        """Test middleware handling of errors in route handlers."""
        collector = MetricsCollector(storage=metrics_storage, enable_auto_flush=False)
        await collector.start()

        middleware = MetricsMiddleware(
            app=Mock(),
            collector=collector,
        )

        # Create mock request
        request = Mock(spec=Request)
        request.method = "POST"
        request.url.path = "/v1/messages"
        request.query_params = {}
        request.headers = {}
        request.cookies = {}
        request.state = Mock()
        # Ensure session_id is None instead of a Mock object
        request.state.session_id = None
        # Mock the client to provide proper IP address
        request.client = Mock()
        request.client.host = "127.0.0.1"

        # Mock handler that raises an exception
        async def failing_handler(req: Request) -> Response:
            raise ValueError("Test error")

        # Process request (should handle the exception)
        response = await middleware.dispatch(request, failing_handler)

        # Should return error response
        assert response.status_code == 500

        # Flush and check metrics
        await collector.flush()

        # Should have request, response, and error metrics
        error_metrics = await metrics_storage.get_metrics(metric_type=MetricType.ERROR)
        response_metrics = await metrics_storage.get_metrics(
            metric_type=MetricType.RESPONSE
        )

        assert len(error_metrics) == 1
        assert len(response_metrics) == 1

        # Verify error details
        error_metric = cast(ErrorMetric, error_metrics[0])
        assert error_metric.error_type == "ValueError"
        assert "Test error" in (error_metric.error_message or "")

        # Verify response shows error
        resp_metric = cast(ResponseMetric, response_metrics[0])
        assert resp_metric.status_code == 500

        await collector.stop()


class TestConcurrentMetrics:
    """Test metrics collection under concurrent load."""

    @pytest.mark.unit
    async def test_concurrent_request_collection(
        self, metrics_storage: InMemoryMetricsStorage
    ) -> None:
        """Test collecting metrics from concurrent requests."""
        collector = MetricsCollector(storage=metrics_storage, enable_auto_flush=False)
        await collector.start()

        async def simulate_request(request_id: str, user_id: str) -> None:
            """Simulate a complete request lifecycle."""
            # Request start
            await collector.collect_request_start(
                request_id=request_id,
                method="POST",
                path="/v1/messages",
                endpoint="/v1/messages",
                api_version="v1",
                user_id=user_id,
            )

            # Small delay to simulate processing
            await asyncio.sleep(0.01)

            # Response
            await collector.collect_response(
                request_id=request_id,
                status_code=200,
                input_tokens=100,
                output_tokens=50,
            )

            # Finish request
            await collector.finish_request(request_id)

        # Run multiple concurrent requests
        tasks = [simulate_request(f"req-{i}", f"user-{i % 3}") for i in range(10)]

        await asyncio.gather(*tasks)

        # Flush and verify all metrics were collected
        await collector.flush()
        stored_metrics = await metrics_storage.get_metrics()

        # Should have 10 requests + 10 responses = 20 metrics
        assert len(stored_metrics) == 20

        request_metrics = await metrics_storage.get_metrics(
            metric_type=MetricType.REQUEST
        )
        response_metrics = await metrics_storage.get_metrics(
            metric_type=MetricType.RESPONSE
        )

        assert len(request_metrics) == 10
        assert len(response_metrics) == 10

        # Verify no active requests remain
        assert len(collector._active_requests) == 0

        await collector.stop()

    @pytest.mark.unit
    async def test_concurrent_storage_operations(
        self, metrics_storage: InMemoryMetricsStorage
    ) -> None:
        """Test concurrent storage operations."""
        await metrics_storage.initialize()

        async def store_batch(batch_id: int) -> None:
            """Store a batch of metrics."""
            metrics: list[MetricRecord] = [
                RequestMetric(
                    request_id=f"batch-{batch_id}-req-{i}",
                    user_id=f"user-{batch_id}",
                    method="POST",
                    path="/v1/messages",
                    endpoint="/v1/messages",
                    api_version="v1",
                )
                for i in range(5)
            ]
            await metrics_storage.store_metrics(metrics)

        # Run concurrent storage operations
        tasks = [store_batch(i) for i in range(5)]
        await asyncio.gather(*tasks)

        # Verify all metrics were stored
        total_count = await metrics_storage.count_metrics()
        assert total_count == 25  # 5 batches * 5 metrics each

        # Verify user distribution
        for i in range(5):
            user_count = await metrics_storage.count_metrics(user_id=f"user-{i}")
            assert user_count == 5

        await metrics_storage.close()

    @pytest.mark.unit
    async def test_high_throughput_metrics(
        self, metrics_storage: InMemoryMetricsStorage
    ) -> None:
        """Test metrics collection under high throughput."""
        collector = MetricsCollector(
            storage=metrics_storage,
            buffer_size=50,
            enable_auto_flush=True,
            flush_interval=0.1,  # Fast flush for testing
        )
        await collector.start()

        # Generate high volume of metrics quickly
        start_time = time.time()

        for i in range(100):
            await collector.collect_request_start(
                request_id=f"high-throughput-{i}",
                method="POST",
                path="/v1/messages",
                endpoint="/v1/messages",
                api_version="v1",
                user_id=f"user-{i % 10}",
            )

        end_time = time.time()
        collection_time = end_time - start_time

        # Should be able to collect 100 metrics quickly
        assert collection_time < 1.0  # Less than 1 second

        # Wait for auto-flush to complete
        await asyncio.sleep(0.5)

        # Verify metrics were stored
        total_count = await metrics_storage.count_metrics()
        assert total_count == 100

        # Verify collector statistics
        stats = collector.get_stats()
        assert stats["total_metrics_collected"] == 100
        assert stats["metrics_by_type"][MetricType.REQUEST] == 100

        await collector.stop()


class TestJsonApiExporter:
    """Test JSON API exporter functionality including clean serialization."""

    @pytest.mark.unit
    async def test_metric_to_dict_excludes_none_values(
        self, metrics_storage: InMemoryMetricsStorage
    ) -> None:
        """Test that metric serialization excludes None values."""
        exporter = JsonApiExporter(storage=metrics_storage)

        # Create a metric with some None values
        metric = RequestMetric(
            request_id="test-request-123",
            method="POST",
            path="/v1/messages",
            endpoint="/v1/messages",
            api_version="v1",
            user_id="test-user",
            # These should be None and excluded
            session_id=None,
            client_ip=None,
            user_agent=None,
            content_length=None,
            content_type=None,
            model=None,
            provider=None,
            max_tokens=None,
            temperature=None,
            streaming=False,  # This should be included as it's not None
        )

        # Convert to dictionary
        metric_dict = exporter._metric_to_dict(metric)

        # Verify required fields are present
        assert "id" in metric_dict
        assert "timestamp" in metric_dict
        assert "metric_type" in metric_dict
        assert "request_id" in metric_dict
        assert metric_dict["request_id"] == "test-request-123"
        assert metric_dict["method"] == "POST"
        assert metric_dict["path"] == "/v1/messages"
        assert metric_dict["user_id"] == "test-user"
        assert metric_dict["streaming"] is False

        # Verify None values are excluded
        assert "session_id" not in metric_dict
        assert "client_ip" not in metric_dict
        assert "user_agent" not in metric_dict
        assert "content_length" not in metric_dict
        assert "content_type" not in metric_dict
        assert "model" not in metric_dict
        assert "provider" not in metric_dict
        assert "max_tokens" not in metric_dict
        assert "temperature" not in metric_dict

    @pytest.mark.unit
    async def test_get_metrics_excludes_none_in_filters(
        self, metrics_storage: InMemoryMetricsStorage
    ) -> None:
        """Test that get_metrics response excludes None values in filters."""
        await metrics_storage.initialize()
        exporter = JsonApiExporter(storage=metrics_storage)

        # Call get_metrics with some None parameters
        result = await exporter.get_metrics(
            metric_type=None,  # This should be excluded
            user_id=None,  # This should be excluded
            session_id=None,  # This should be excluded
            limit=10,  # This should be included
        )

        # Verify filters object doesn't contain None values
        filters = result["metadata"]["filters"]
        assert "limit" not in filters  # limit is not in filters

        # These should not be present since they were None
        assert "metric_type" not in filters
        assert "user_id" not in filters
        assert "session_id" not in filters
        assert "request_id" not in filters
        assert "custom_filters" not in filters

        # Pagination should exclude None values too
        pagination = result["metadata"]["pagination"]
        assert "has_next" in pagination
        assert "has_previous" in pagination
        # next_offset and previous_offset should not be present if None
        if "next_offset" in pagination:
            assert pagination["next_offset"] is not None
        if "previous_offset" in pagination:
            assert pagination["previous_offset"] is not None

    @pytest.mark.unit
    async def test_response_metric_serialization_excludes_none(
        self, metrics_storage: InMemoryMetricsStorage
    ) -> None:
        """Test that ResponseMetric serialization excludes None values."""
        exporter = JsonApiExporter(storage=metrics_storage)

        # Create response metric with minimal data
        metric = ResponseMetric(
            request_id="test-response-123",
            status_code=200,
            response_time_ms=150.5,
            user_id="test-user",
            # These will be None and should be excluded
            content_length=None,
            content_type=None,
            input_tokens=None,
            output_tokens=None,
            cache_read_tokens=None,
            cache_write_tokens=None,
            first_token_time_ms=None,
            stream_completion_time_ms=None,
            completion_reason=None,
            streaming=False,
            safety_filtered=False,
        )

        metric_dict = exporter._metric_to_dict(metric)

        # Required fields should be present
        assert metric_dict["request_id"] == "test-response-123"
        assert metric_dict["status_code"] == 200
        assert metric_dict["response_time_ms"] == 150.5
        assert metric_dict["user_id"] == "test-user"
        assert metric_dict["streaming"] is False
        assert metric_dict["safety_filtered"] is False

        # None values should be excluded
        assert "content_length" not in metric_dict
        assert "content_type" not in metric_dict
        assert "input_tokens" not in metric_dict
        assert "output_tokens" not in metric_dict
        assert "cache_read_tokens" not in metric_dict
        assert "cache_write_tokens" not in metric_dict
        assert "first_token_time_ms" not in metric_dict
        assert "stream_completion_time_ms" not in metric_dict
        assert "completion_reason" not in metric_dict

    @pytest.mark.unit
    async def test_cost_metric_with_partial_data(
        self, metrics_storage: InMemoryMetricsStorage
    ) -> None:
        """Test CostMetric serialization with partial data."""
        exporter = JsonApiExporter(storage=metrics_storage)

        # Create cost metric with only some values (None values for optional fields)
        metric = CostMetric(
            request_id="test-cost-123",
            input_cost=0.001,
            output_cost=0.002,
            total_cost=0.003,
            model="claude-3-haiku-20240307",
            currency="USD",
            input_tokens=100,
            output_tokens=50,
            # These have defaults and will be present
            cache_read_cost=0.0,  # Has default value
            cache_write_cost=0.0,  # Has default value
            cache_read_tokens=0,  # Has default value
            cache_write_tokens=0,  # Has default value
            # These are None and should be excluded
            sdk_total_cost=None,
            sdk_input_cost=None,
            sdk_output_cost=None,
            sdk_cache_read_cost=None,
            sdk_cache_write_cost=None,
            cost_difference=None,
            cost_accuracy_percentage=None,
            pricing_tier=None,
        )

        metric_dict = exporter._metric_to_dict(metric)

        # Present values should be included
        assert metric_dict["input_cost"] == 0.001
        assert metric_dict["output_cost"] == 0.002
        assert metric_dict["total_cost"] == 0.003
        assert metric_dict["model"] == "claude-3-haiku-20240307"
        assert metric_dict["currency"] == "USD"
        assert metric_dict["input_tokens"] == 100
        assert metric_dict["output_tokens"] == 50
        assert metric_dict["cache_read_cost"] == 0.0
        assert metric_dict["cache_write_cost"] == 0.0
        assert metric_dict["cache_read_tokens"] == 0
        assert metric_dict["cache_write_tokens"] == 0

        # None values should be excluded
        assert "sdk_total_cost" not in metric_dict
        assert "sdk_input_cost" not in metric_dict
        assert "sdk_output_cost" not in metric_dict
        assert "sdk_cache_read_cost" not in metric_dict
        assert "sdk_cache_write_cost" not in metric_dict
        assert "cost_difference" not in metric_dict
        assert "cost_accuracy_percentage" not in metric_dict
        assert "pricing_tier" not in metric_dict


class TestMetricsEndToEnd:
    """Test complete end-to-end metrics pipeline."""

    @pytest.mark.unit
    async def test_complete_request_lifecycle_metrics(
        self, metrics_storage: InMemoryMetricsStorage
    ) -> None:
        """Test metrics collection for a complete request lifecycle."""
        collector = MetricsCollector(storage=metrics_storage, enable_auto_flush=False)
        await collector.start()

        request_id = "e2e-test-123"
        user_id = "test-user"
        model = "claude-3-5-sonnet-20241022"

        # 1. Request start
        request_metric = await collector.collect_request_start(
            request_id=request_id,
            method="POST",
            path="/v1/messages",
            endpoint="/v1/messages",
            api_version="v1",
            user_id=user_id,
            model=model,
            provider="anthropic",
            streaming=False,
            max_tokens=1000,
            temperature=0.7,
        )

        # 2. Simulate processing delay
        await asyncio.sleep(0.05)

        # 3. Response with token usage
        response_metric = await collector.collect_response(
            request_id=request_id,
            status_code=200,
            input_tokens=150,
            output_tokens=75,
            cache_read_tokens=20,
            cache_write_tokens=10,
        )

        # 4. Latency tracking
        latency_metric = await collector.collect_latency(
            request_id=request_id,
            total_latency_ms=50.0,
            request_processing_ms=45.0,
            model_latency_ms=40.0,
        )

        # 5. Finish request
        await collector.finish_request(request_id)

        # Flush all metrics
        await collector.flush()

        # Verify complete lifecycle was captured
        all_metrics = await metrics_storage.get_metrics(request_id=request_id)
        assert (
            len(all_metrics) >= 4
        )  # Request, Response, Cost (auto-generated), Latency

        # Verify each metric type
        request_metrics = [m for m in all_metrics if isinstance(m, RequestMetric)]
        response_metrics = [m for m in all_metrics if isinstance(m, ResponseMetric)]
        cost_metrics = [m for m in all_metrics if isinstance(m, CostMetric)]
        latency_metrics = [m for m in all_metrics if isinstance(m, LatencyMetric)]

        assert len(request_metrics) == 1
        assert len(response_metrics) == 1
        assert len(cost_metrics) == 1  # Auto-generated from token usage
        assert len(latency_metrics) == 1

        # Verify data consistency across metrics
        req = request_metrics[0]
        resp = response_metrics[0]
        cost = cost_metrics[0]
        lat = latency_metrics[0]

        # All should have same correlation IDs
        assert req.request_id == resp.request_id == cost.request_id == lat.request_id
        assert req.user_id == resp.user_id == cost.user_id == lat.user_id

        # Verify specific values
        assert req.model == model
        assert req.max_tokens == 1000
        assert req.temperature == 0.7
        assert resp.status_code == 200
        assert resp.input_tokens == 150
        assert resp.output_tokens == 75
        assert cost.total_cost > 0  # Should have calculated cost
        assert lat.total_latency_ms == 50.0

        # Generate summary for the complete lifecycle
        summary = await collector.get_summary()
        assert summary.total_requests == 1
        assert summary.successful_requests == 1
        assert summary.failed_requests == 0
        assert summary.total_input_tokens == 150
        assert summary.total_output_tokens == 75
        assert summary.total_cost > 0
        assert summary.unique_users == 1

        await collector.stop()

    @pytest.mark.unit
    async def test_error_scenario_metrics(
        self, metrics_storage: InMemoryMetricsStorage
    ) -> None:
        """Test metrics collection for error scenarios."""
        collector = MetricsCollector(storage=metrics_storage, enable_auto_flush=False)
        await collector.start()

        request_id = "error-test-123"

        # 1. Request start
        await collector.collect_request_start(
            request_id=request_id,
            method="POST",
            path="/v1/messages",
            endpoint="/v1/messages",
            api_version="v1",
            user_id="test-user",
            model="invalid-model",
        )

        # 2. Error during processing
        await collector.collect_error(
            request_id=request_id,
            error_type="ValidationError",
            error_code="invalid_model",
            error_message="The specified model is not supported",
            endpoint="/v1/messages",
            method="POST",
            status_code=400,
        )

        # 3. Error response
        await collector.collect_response(
            request_id=request_id,
            status_code=400,
            input_tokens=0,
            output_tokens=0,
        )

        # 4. Finish request
        await collector.finish_request(request_id)

        # Flush and analyze
        await collector.flush()

        # Verify error tracking
        error_metrics = await metrics_storage.get_metrics(metric_type=MetricType.ERROR)
        assert len(error_metrics) == 1

        error = cast(ErrorMetric, error_metrics[0])
        assert error.error_type == "ValidationError"
        assert error.error_code == "invalid_model"
        assert error.status_code == 400

        # Generate summary
        summary = await collector.get_summary()
        assert summary.total_requests == 1
        assert summary.successful_requests == 0
        assert summary.failed_requests == 1
        assert summary.error_rate == 1.0
        assert "ValidationError" in summary.error_types
        assert summary.error_types["ValidationError"] == 1

        await collector.stop()
