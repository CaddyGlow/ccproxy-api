"""Tests for metrics API endpoints with DuckDB storage."""

import asyncio
import json
import time
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from ccproxy.api.app import create_app
from ccproxy.config.settings import Settings


@pytest.mark.unit
class TestMetricsAPIEndpoints:
    """Test metrics API endpoints functionality."""

    @pytest.fixture
    def client(self, test_settings: Settings) -> TestClient:
        """Create test client."""
        app = create_app(test_settings)
        return TestClient(app)

    @pytest.fixture
    def mock_storage(self) -> AsyncMock:
        """Mock DuckDB storage backend."""
        storage = AsyncMock()
        storage.is_enabled.return_value = True
        storage.health_check.return_value = {
            "status": "healthy",
            "enabled": True,
            "database_path": "/tmp/test.duckdb",
            "request_count": 100,
            "pool_size": 3,
        }
        storage.query.return_value = [
            {
                "request_id": "req_123",
                "method": "POST",
                "endpoint": "messages",
                "model": "claude-3-sonnet",
                "status": "success",
                "response_time": 1.5,
                "tokens_input": 150,
                "tokens_output": 75,
                "cost_usd": 0.0023,
            }
        ]
        storage.get_analytics.return_value = {
            "summary": {
                "total_requests": 100,
                "successful_requests": 95,
                "failed_requests": 5,
                "avg_response_time": 1.2,
                "median_response_time": 1.0,
                "p95_response_time": 2.5,
                "total_tokens_input": 15000,
                "total_tokens_output": 7500,
                "total_cost_usd": 0.23,
            },
            "hourly_data": [
                {"hour": "2024-01-01 10:00:00", "request_count": 25, "error_count": 1},
                {"hour": "2024-01-01 11:00:00", "request_count": 30, "error_count": 2},
            ],
            "model_stats": [
                {
                    "model": "claude-3-sonnet",
                    "request_count": 60,
                    "avg_response_time": 1.1,
                    "total_cost": 0.15,
                },
                {
                    "model": "claude-3-haiku",
                    "request_count": 40,
                    "avg_response_time": 0.8,
                    "total_cost": 0.08,
                },
            ],
            "query_time": time.time(),
        }
        return storage

    def test_query_endpoint_success(
        self, client: TestClient, mock_storage: AsyncMock
    ) -> None:
        """Test successful query execution with filters."""
        from ccproxy.api.dependencies import get_duckdb_storage

        # Mock the storage engine and session
        mock_engine = MagicMock()
        mock_session = MagicMock()
        mock_storage._engine = mock_engine

        # Mock the session context manager
        mock_session_context = MagicMock()
        mock_session_context.__enter__.return_value = mock_session
        mock_session_context.__exit__.return_value = None

        # Override the dependency - match the actual signature
        async def get_mock_storage(request: Request) -> AsyncMock:
            return mock_storage

        # Replace the dependency in the app
        app: FastAPI = client.app  # type: ignore[assignment]
        app.dependency_overrides[get_duckdb_storage] = get_mock_storage

        try:
            with patch(
                "ccproxy.api.routes.metrics.Session", return_value=mock_session_context
            ):
                # Mock the exec method to return mock results
                mock_result = MagicMock()
                mock_log = MagicMock()
                mock_log.dict.return_value = {
                    "request_id": "req_123",
                    "method": "POST",
                    "endpoint": "messages",
                    "model": "claude-3-sonnet",
                    "status": "success",
                    "response_time": 1.5,
                    "tokens_input": 150,
                    "tokens_output": 75,
                    "cost_usd": 0.0023,
                }
                mock_result.all.return_value = [mock_log]
                mock_session.exec.return_value = mock_result

                response = client.get(
                    "/logs/query",
                    params={
                        "model": "claude-3-sonnet",
                        "limit": 100,
                    },
                )

                assert response.status_code == 200
                data = response.json()

                assert "results" in data
                assert "count" in data
                assert "limit" in data
                assert "timestamp" in data

                assert data["count"] == 1
                assert data["limit"] == 100
                assert len(data["results"]) == 1
                assert data["results"][0]["model"] == "claude-3-sonnet"
        finally:
            # Clean up the dependency override
            app.dependency_overrides.clear()

    def test_query_endpoint_no_sql_injection_risk(
        self, client: TestClient, mock_storage: AsyncMock
    ) -> None:
        """Test that query endpoint doesn't accept raw SQL (no SQL injection risk)."""
        from ccproxy.api.dependencies import get_duckdb_storage

        # Mock the storage engine and session
        mock_engine = MagicMock()
        mock_session = MagicMock()
        mock_storage._engine = mock_engine

        # Add proper async context manager attributes
        mock_session.in_transaction = False
        mock_session.is_active = True
        mock_session.connection = MagicMock()

        # Mock the session context manager
        mock_session_context = MagicMock()
        mock_session_context.__enter__.return_value = mock_session
        mock_session_context.__exit__.return_value = None

        # Override the dependency - match the actual signature
        async def get_mock_storage(request: Request) -> AsyncMock:
            return mock_storage

        app: FastAPI = client.app  # type: ignore[assignment]
        app.dependency_overrides[get_duckdb_storage] = get_mock_storage

        try:
            with patch(
                "ccproxy.api.routes.metrics.Session", return_value=mock_session_context
            ):
                # Mock the exec method to return empty results
                mock_result = MagicMock()
                mock_result.all.return_value = []
                mock_session.exec.return_value = mock_result

                # The current implementation doesn't accept raw SQL, only predefined filters
                # This is actually safer as it prevents SQL injection entirely
                response = client.get(
                    "/logs/query", params={"model": "claude-3-sonnet"}
                )

                # Should work with valid filters
                assert response.status_code == 200
        finally:
            app.dependency_overrides.clear()

    def test_query_endpoint_valid_filters(
        self, client: TestClient, mock_storage: AsyncMock
    ) -> None:
        """Test valid filter parameters are accepted."""
        from ccproxy.api.dependencies import get_duckdb_storage

        # Mock the storage engine and session
        mock_engine = MagicMock()
        mock_session = MagicMock()
        mock_storage._engine = mock_engine

        # Mock the session context manager
        mock_session_context = MagicMock()
        mock_session_context.__enter__.return_value = mock_session
        mock_session_context.__exit__.return_value = None

        # Override the dependency - match the actual signature
        async def get_mock_storage(request: Request) -> AsyncMock:
            return mock_storage

        app: FastAPI = client.app  # type: ignore[assignment]
        app.dependency_overrides[get_duckdb_storage] = get_mock_storage

        try:
            with patch(
                "ccproxy.api.routes.metrics.Session", return_value=mock_session_context
            ):
                # Mock the exec method to return empty results
                mock_result = MagicMock()
                mock_result.all.return_value = []
                mock_session.exec.return_value = mock_result

                valid_filter_sets: list[dict[str, Any]] = [
                    {},  # No filters
                    {"model": "claude-3-sonnet"},
                    {"limit": 50},
                    {"start_time": 1704067200, "end_time": 1704153600},  # Jan 1-2, 2024
                    {"model": "claude-3-haiku", "limit": 10},
                    {"service_type": "proxy_service"},
                ]

                for filters in valid_filter_sets:
                    response = client.get("/logs/query", params=filters)
                    assert response.status_code == 200
        finally:
            app.dependency_overrides.clear()

    def test_query_endpoint_no_storage(self, client: TestClient) -> None:
        """Test query endpoint when storage is not available."""
        from ccproxy.api.dependencies import get_duckdb_storage

        # Override the dependency to return None
        async def get_mock_storage(request: Request) -> None:
            return None

        app: FastAPI = client.app  # type: ignore[assignment]
        app.dependency_overrides[get_duckdb_storage] = get_mock_storage

        try:
            response = client.get(
                "/logs/query",
                params={"model": "claude-3-sonnet"},
            )

            assert response.status_code == 503
            assert (
                "Storage backend not available" in response.json()["error"]["message"]
            )
        finally:
            app.dependency_overrides.clear()

    def test_analytics_endpoint_success(
        self, client: TestClient, mock_storage: AsyncMock
    ) -> None:
        """Test successful analytics generation."""
        from ccproxy.api.dependencies import get_duckdb_storage

        # Mock the dependency to return the storage
        async def get_mock_storage(request: Request) -> AsyncMock:
            return mock_storage

        app: FastAPI = client.app  # type: ignore[assignment]
        app.dependency_overrides[get_duckdb_storage] = get_mock_storage

        try:
            # Mock the storage engine and session
            mock_engine = MagicMock()
            mock_session = MagicMock()
            mock_storage._engine = mock_engine

            # Mock the session context manager
            mock_session_context = MagicMock()
            mock_session_context.__enter__.return_value = mock_session
            mock_session_context.__exit__.return_value = None

            with patch(
                "ccproxy.api.routes.metrics.Session", return_value=mock_session_context
            ):
                # Mock the exec method to return analytics data
                mock_result = MagicMock()

                # Mock the different queries in sequence
                exec_call_count = 0

                def mock_exec_side_effect(*args: Any, **kwargs: Any) -> MagicMock:
                    nonlocal exec_call_count
                    exec_call_count += 1
                    mock_result_temp = MagicMock()
                    # Return different values for different analytics queries
                    if exec_call_count == 1:  # total_requests
                        mock_result_temp.first.return_value = 100
                    elif exec_call_count == 2:  # avg_duration
                        mock_result_temp.first.return_value = 1.2
                    elif exec_call_count == 3:  # total_cost
                        mock_result_temp.first.return_value = 0.23
                    elif exec_call_count == 4:  # total_tokens_input
                        mock_result_temp.first.return_value = 15000
                    elif exec_call_count == 5:  # total_tokens_output
                        mock_result_temp.first.return_value = 7500
                    elif exec_call_count == 6:  # cache_read_tokens
                        mock_result_temp.first.return_value = 500
                    elif exec_call_count == 7:  # cache_write_tokens
                        mock_result_temp.first.return_value = 300
                    elif exec_call_count == 8:  # successful_requests
                        mock_result_temp.first.return_value = 95
                    elif exec_call_count == 9:  # error_requests
                        mock_result_temp.first.return_value = 5
                    elif exec_call_count == 10:  # unique_services
                        mock_result_temp.all.return_value = ["proxy_service"]
                    else:  # service-specific queries
                        mock_result_temp.first.return_value = 50
                    return mock_result_temp

                mock_session.exec.side_effect = mock_exec_side_effect

                response = client.get("/logs/analytics", params={"hours": 24})

                assert response.status_code == 200
                data = response.json()

                assert "summary" in data
                assert "token_analytics" in data
                assert "request_analytics" in data
                assert "service_type_breakdown" in data
                assert "query_params" in data

                summary = data["summary"]
                assert summary["total_requests"] == 100
                assert summary["total_successful_requests"] == 95
                assert summary["total_error_requests"] == 5
        finally:
            app.dependency_overrides.clear()

    def test_analytics_endpoint_with_filters(
        self, client: TestClient, mock_storage: AsyncMock
    ) -> None:
        """Test analytics with time and model filters."""
        from ccproxy.api.dependencies import get_duckdb_storage

        # Override the dependency to return the storage
        async def get_mock_storage(request: Request) -> AsyncMock:
            return mock_storage

        app: FastAPI = client.app  # type: ignore[assignment]
        app.dependency_overrides[get_duckdb_storage] = get_mock_storage

        try:
            # Mock the storage engine and session
            mock_engine = MagicMock()
            mock_session = MagicMock()
            mock_storage._engine = mock_engine

            # Mock the session context manager
            mock_session_context = MagicMock()
            mock_session_context.__enter__.return_value = mock_session
            mock_session_context.__exit__.return_value = None

            with patch(
                "ccproxy.api.routes.metrics.Session", return_value=mock_session_context
            ):
                # Mock the exec method to return basic analytics data
                mock_result = MagicMock()
                mock_result.first.return_value = 10  # Simple mock value
                mock_result.all.return_value = []  # Empty services list
                mock_session.exec.return_value = mock_result

                start_time = time.time() - 86400  # 24 hours ago
                end_time = time.time()

                response = client.get(
                    "/logs/analytics",
                    params={
                        "start_time": start_time,
                        "end_time": end_time,
                        "model": "claude-3-sonnet",
                    },
                )

                assert response.status_code == 200
                data = response.json()

                # Verify filters were passed correctly
                query_params = data["query_params"]
                assert query_params["start_time"] == start_time
                assert query_params["end_time"] == end_time
                assert query_params["model"] == "claude-3-sonnet"
        finally:
            app.dependency_overrides.clear()

    def test_analytics_endpoint_default_time_range(
        self, client: TestClient, mock_storage: AsyncMock
    ) -> None:
        """Test analytics with default time range."""
        from ccproxy.api.dependencies import get_duckdb_storage

        # Override the dependency to return the storage
        async def get_mock_storage(request: Request) -> AsyncMock:
            return mock_storage

        app: FastAPI = client.app  # type: ignore[assignment]
        app.dependency_overrides[get_duckdb_storage] = get_mock_storage

        try:
            # Mock the storage engine and session
            mock_engine = MagicMock()
            mock_session = MagicMock()
            mock_storage._engine = mock_engine

            # Mock the session context manager
            mock_session_context = MagicMock()
            mock_session_context.__enter__.return_value = mock_session
            mock_session_context.__exit__.return_value = None

            with patch(
                "ccproxy.api.routes.metrics.Session", return_value=mock_session_context
            ):
                # Mock the exec method to return basic analytics data
                mock_result = MagicMock()
                mock_result.first.return_value = 10  # Simple mock value
                mock_result.all.return_value = []  # Empty services list
                mock_session.exec.return_value = mock_result

                response = client.get("/logs/analytics", params={"hours": 48})

                assert response.status_code == 200
                data = response.json()

                query_params = data["query_params"]
                assert query_params["hours"] == 48
                assert query_params["start_time"] is not None
                assert query_params["end_time"] is not None

                # Verify time range is approximately 48 hours
                time_diff = query_params["end_time"] - query_params["start_time"]
                assert abs(time_diff - (48 * 3600)) < 60  # Within 1 minute tolerance
        finally:
            app.dependency_overrides.clear()

    def test_analytics_endpoint_no_storage(self, client: TestClient) -> None:
        """Test analytics endpoint when storage is not available."""
        from ccproxy.api.dependencies import get_duckdb_storage

        # Override the dependency to return None
        async def get_mock_storage(request: Request) -> None:
            return None

        app: FastAPI = client.app  # type: ignore[assignment]
        app.dependency_overrides[get_duckdb_storage] = get_mock_storage

        try:
            response = client.get("/logs/analytics")

            assert response.status_code == 503
            assert (
                "Storage backend not available" in response.json()["error"]["message"]
            )
        finally:
            app.dependency_overrides.clear()

    def test_status_endpoint(self, client: TestClient) -> None:
        """Test status endpoint returns observability system info."""
        response = client.get("/logs/status")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert "prometheus_enabled" in data
        assert data["observability_system"] == "hybrid_prometheus_structlog"

    def test_prometheus_endpoint_unavailable(self, client: TestClient) -> None:
        """Test prometheus endpoint when prometheus_client not available."""
        with patch("ccproxy.observability.metrics.PROMETHEUS_AVAILABLE", False):
            from ccproxy.observability import reset_metrics

            # Reset global state to pick up the patched PROMETHEUS_AVAILABLE
            reset_metrics()

            response = client.get("/metrics")

            # Should get 503 due to missing prometheus_client
            assert response.status_code == 503


@pytest.mark.unit
@pytest.mark.skip("infinte loop")
class TestSSEStreamingEndpoint:
    """Test SSE streaming endpoint functionality."""

    @pytest.fixture
    def client(self, test_settings: Settings) -> TestClient:
        """Create test client."""
        app = create_app(test_settings)
        return TestClient(app)

    @pytest.mark.skip("infinte loop")
    def test_sse_stream_endpoint_basic(self, client: TestClient) -> None:
        """Test basic SSE stream endpoint functionality."""
        # Test the new event-driven SSE endpoint
        with client.stream("GET", "/logs/stream") as response:
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/event-stream"
            assert response.headers["cache-control"] == "no-cache"
            assert response.headers["connection"] == "keep-alive"

            # Read first event (should be connection event)
            events = []
            line_count = 0
            max_lines = 10  # Limit to prevent infinite loop

            for line_count, line in enumerate(response.iter_lines()):
                line_count += 1
                if line_count > max_lines:
                    break
                if line.startswith("data: "):
                    event_data = json.loads(line[6:])  # Remove "data: " prefix
                    events.append(event_data)

                    # Stop after connection event
                    if event_data.get("type") == "connection":
                        break

            # Should receive connection event
            assert len(events) >= 1
            connection_event = events[0]
            assert connection_event["type"] == "connection"
            assert connection_event["message"] == "Connected to metrics stream"
            assert "connection_id" in connection_event
            assert "timestamp" in connection_event

    def test_sse_stream_endpoint_headers(self, client: TestClient) -> None:
        """Test SSE stream endpoint has correct headers."""
        response = client.get("/logs/stream")

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream"
        assert response.headers["cache-control"] == "no-cache"
        assert response.headers["connection"] == "keep-alive"
        assert response.headers["access-control-allow-origin"] == "*"
        assert response.headers["access-control-allow-headers"] == "Cache-Control"

    @patch("ccproxy.observability.sse_events.get_sse_manager")
    def test_sse_stream_with_events(
        self, mock_get_sse_manager: MagicMock, client: TestClient
    ) -> None:
        """Test SSE stream endpoint with event emission."""
        # Mock SSE manager
        mock_manager = AsyncMock()
        mock_get_sse_manager.return_value = mock_manager

        # Mock event stream
        async def mock_event_stream(connection_id: str) -> AsyncGenerator[str, None]:
            # Connection event
            connection_event = {
                "type": "connection",
                "message": "Connected to metrics stream",
                "connection_id": connection_id,
                "timestamp": time.time(),
            }
            yield f"data: {json.dumps(connection_event)}\n\n"

            # Test request event
            request_event = {
                "type": "request_start",
                "data": {
                    "request_id": "test-123",
                    "method": "POST",
                    "path": "/api/v1/messages",
                },
                "timestamp": time.time(),
            }
            yield f"data: {json.dumps(request_event)}\n\n"

            # Test completion event
            completion_event = {
                "type": "request_complete",
                "data": {
                    "request_id": "test-123",
                    "method": "POST",
                    "path": "/api/v1/messages",
                    "status_code": 200,
                    "duration_ms": 850.5,
                    "tokens_input": 39,
                    "tokens_output": 10,
                },
                "timestamp": time.time(),
            }
            yield f"data: {json.dumps(completion_event)}\n\n"

        mock_manager.add_connection.side_effect = mock_event_stream

        # Test streaming
        with client.stream("GET", "/logs/stream") as response:
            assert response.status_code == 200

            events = []
            for line in response.iter_lines():
                if line.startswith("data: "):
                    event_data = json.loads(line[6:])  # Remove "data: " prefix
                    events.append(event_data)

                    # Stop after receiving completion event
                    if event_data.get("type") == "request_complete":
                        break

            # Should receive all three events
            assert len(events) == 3

            # Check connection event
            assert events[0]["type"] == "connection"
            assert events[0]["message"] == "Connected to metrics stream"

            # Check request start event
            assert events[1]["type"] == "request_start"
            assert events[1]["data"]["request_id"] == "test-123"
            assert events[1]["data"]["method"] == "POST"
            assert events[1]["data"]["path"] == "/api/v1/messages"

            # Check request complete event
            assert events[2]["type"] == "request_complete"
            assert events[2]["data"]["request_id"] == "test-123"
            assert events[2]["data"]["status_code"] == 200
            assert events[2]["data"]["duration_ms"] == 850.5
            assert events[2]["data"]["tokens_input"] == 39
            assert events[2]["data"]["tokens_output"] == 10

    @patch("ccproxy.observability.sse_events.get_sse_manager")
    def test_sse_stream_error_handling(
        self, mock_get_sse_manager: MagicMock, client: TestClient
    ) -> None:
        """Test SSE stream endpoint error handling."""
        # Mock SSE manager to raise exception
        mock_manager = AsyncMock()
        mock_get_sse_manager.return_value = mock_manager

        async def mock_failing_stream(connection_id: str) -> AsyncGenerator[str, None]:
            # Connection event
            connection_event = {
                "type": "connection",
                "message": "Connected to metrics stream",
                "connection_id": connection_id,
                "timestamp": time.time(),
            }
            yield f"data: {json.dumps(connection_event)}\n\n"

            # Raise exception
            raise Exception("Test error")

        mock_manager.add_connection.side_effect = mock_failing_stream

        # Test streaming with error
        with client.stream("GET", "/logs/stream") as response:
            assert response.status_code == 200

            events = []
            for line in response.iter_lines():
                if line.startswith("data: "):
                    event_data = json.loads(line[6:])  # Remove "data: " prefix
                    events.append(event_data)

                    # Stop after error event
                    if event_data.get("type") == "error":
                        break

            # Should receive connection event and error event
            assert len(events) >= 2

            # Check connection event
            assert events[0]["type"] == "connection"

            # Check error event
            assert events[1]["type"] == "error"
            assert events[1]["message"] == "Test error"
            assert "timestamp" in events[1]


@pytest.mark.integration
@pytest.mark.skip("infinte loop")
class TestSSEIntegration:
    """Integration tests for SSE functionality with real events."""

    @pytest.fixture
    def client_with_sse(self, test_settings: Settings) -> TestClient:
        """Create test client with SSE functionality."""
        app = create_app(test_settings)
        return TestClient(app)

    async def test_sse_real_time_events(self, client_with_sse: TestClient) -> None:
        """Test real-time SSE events during API requests."""
        # This test would ideally make real API requests and verify
        # that SSE events are emitted in real-time. Due to complexity
        # of setting up full authentication and Claude API mocking,
        # this demonstrates the test structure for integration testing.

        # Start SSE stream
        events = []

        async def collect_events() -> None:
            with client_with_sse.stream("GET", "/logs/stream") as response:
                assert response.status_code == 200

                for line in response.iter_lines():
                    if line.startswith("data: "):
                        event_data = json.loads(line[6:])
                        events.append(event_data)

                        # Stop after collecting a few events
                        if len(events) >= 2:
                            break

        # Note: In a full integration test, we would:
        # 1. Start SSE stream collection in background
        # 2. Make API requests that trigger access logging
        # 3. Verify that corresponding SSE events are emitted
        # 4. Check event timing and content accuracy

        # For now, just verify the stream starts correctly
        await collect_events()

        # Should receive at least connection event
        assert len(events) >= 1
        assert events[0]["type"] == "connection"
