"""Tests for metrics API endpoints with DuckDB storage."""

import asyncio
import json
import time
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
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

    @patch("ccproxy.api.routes.metrics.get_storage_backend")
    def test_query_endpoint_success(
        self, mock_get_storage: MagicMock, client: TestClient, mock_storage: AsyncMock
    ) -> None:
        """Test successful SQL query execution."""
        mock_get_storage.return_value = mock_storage

        response = client.get(
            "/metrics/query",
            params={
                "sql": "SELECT * FROM requests WHERE model = 'claude-3-sonnet'",
                "limit": 100,
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert "query" in data
        assert "results" in data
        assert "count" in data
        assert "limit" in data
        assert "timestamp" in data

        assert data["count"] == 1
        assert data["limit"] == 100
        assert len(data["results"]) == 1
        assert data["results"][0]["model"] == "claude-3-sonnet"

    @patch("ccproxy.api.routes.metrics.get_storage_backend")
    def test_query_endpoint_sql_injection_protection(
        self, mock_get_storage: MagicMock, client: TestClient, mock_storage: AsyncMock
    ) -> None:
        """Test SQL injection protection."""
        mock_get_storage.return_value = mock_storage

        # Test malicious SQL
        malicious_queries = [
            "DROP TABLE requests",
            "SELECT * FROM users; DROP TABLE requests;",
            "INSERT INTO requests VALUES (1,2,3)",
            "UPDATE requests SET status = 'hacked'",
            "DELETE FROM requests",
        ]

        for sql in malicious_queries:
            response = client.get("/metrics/query", params={"sql": sql})
            assert response.status_code == 400
            assert "Invalid SQL query" in response.json()["error"]["message"]

    @patch("ccproxy.api.routes.metrics.get_storage_backend")
    def test_query_endpoint_valid_queries(
        self, mock_get_storage: MagicMock, client: TestClient, mock_storage: AsyncMock
    ) -> None:
        """Test valid SQL queries are accepted."""
        mock_get_storage.return_value = mock_storage

        valid_queries = [
            "SELECT * FROM requests",
            "SELECT COUNT(*) FROM requests WHERE timestamp > '2024-01-01'",
            "SELECT model, AVG(response_time) FROM requests GROUP BY model",
            "SELECT * FROM operations WHERE status = 'error' ORDER BY timestamp DESC",
            "SELECT * FROM requests WHERE model = 'claude-3-sonnet' LIMIT 10",
        ]

        for sql in valid_queries:
            response = client.get("/metrics/query", params={"sql": sql})
            assert response.status_code == 200

    @patch("ccproxy.api.routes.metrics.get_storage_backend")
    def test_query_endpoint_no_storage(
        self, mock_get_storage: MagicMock, client: TestClient
    ) -> None:
        """Test query endpoint when storage is not available."""
        mock_get_storage.return_value = None

        response = client.get(
            "/metrics/query",
            params={"sql": "SELECT * FROM requests"},
        )

        assert response.status_code == 503
        assert "Storage backend not available" in response.json()["error"]["message"]

    @patch("ccproxy.api.routes.metrics.get_storage_backend")
    def test_analytics_endpoint_success(
        self, mock_get_storage: MagicMock, client: TestClient, mock_storage: AsyncMock
    ) -> None:
        """Test successful analytics generation."""
        mock_get_storage.return_value = mock_storage

        response = client.get("/metrics/analytics", params={"hours": 24})

        assert response.status_code == 200
        data = response.json()

        assert "summary" in data
        assert "hourly_data" in data
        assert "model_stats" in data
        assert "query_params" in data

        summary = data["summary"]
        assert summary["total_requests"] == 100
        assert summary["successful_requests"] == 95
        assert summary["failed_requests"] == 5

        assert len(data["hourly_data"]) == 2
        assert len(data["model_stats"]) == 2

    @patch("ccproxy.api.routes.metrics.get_storage_backend")
    def test_analytics_endpoint_with_filters(
        self, mock_get_storage: MagicMock, client: TestClient, mock_storage: AsyncMock
    ) -> None:
        """Test analytics with time and model filters."""
        mock_get_storage.return_value = mock_storage

        start_time = time.time() - 86400  # 24 hours ago
        end_time = time.time()

        response = client.get(
            "/metrics/analytics",
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

        # Verify storage method was called with correct parameters
        mock_storage.get_analytics.assert_called_once_with(
            start_time=start_time,
            end_time=end_time,
            model="claude-3-sonnet",
            service_type=None,
        )

    @patch("ccproxy.api.routes.metrics.get_storage_backend")
    def test_analytics_endpoint_default_time_range(
        self, mock_get_storage: MagicMock, client: TestClient, mock_storage: AsyncMock
    ) -> None:
        """Test analytics with default time range."""
        mock_get_storage.return_value = mock_storage

        response = client.get("/metrics/analytics", params={"hours": 48})

        assert response.status_code == 200
        data = response.json()

        query_params = data["query_params"]
        assert query_params["hours"] == 48
        assert query_params["start_time"] is not None
        assert query_params["end_time"] is not None

        # Verify time range is approximately 48 hours
        time_diff = query_params["end_time"] - query_params["start_time"]
        assert abs(time_diff - (48 * 3600)) < 60  # Within 1 minute tolerance

    @patch("ccproxy.api.routes.metrics.get_storage_backend")
    def test_analytics_endpoint_no_storage(
        self, mock_get_storage: MagicMock, client: TestClient
    ) -> None:
        """Test analytics endpoint when storage is not available."""
        mock_get_storage.return_value = None

        response = client.get("/metrics/analytics")

        assert response.status_code == 503
        assert "Storage backend not available" in response.json()["error"]["message"]

    @patch("ccproxy.api.routes.metrics.get_storage_backend")
    def test_health_endpoint_healthy_storage(
        self, mock_get_storage: MagicMock, client: TestClient, mock_storage: AsyncMock
    ) -> None:
        """Test health endpoint with healthy storage."""
        mock_get_storage.return_value = mock_storage

        response = client.get("/metrics/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert data["enabled"] is True
        assert data["storage_backend"] == "duckdb"
        assert data["database_path"] == "/tmp/test.duckdb"
        assert data["request_count"] == 100

    @patch("ccproxy.api.routes.metrics.get_storage_backend")
    def test_health_endpoint_no_storage(
        self, mock_get_storage: MagicMock, client: TestClient
    ) -> None:
        """Test health endpoint when storage is not available."""
        mock_get_storage.return_value = None

        response = client.get("/metrics/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "unavailable"
        assert data["storage_backend"] == "none"
        assert "No storage backend available" in data["message"]

    @patch("ccproxy.api.routes.metrics.get_storage_backend")
    def test_health_endpoint_storage_error(
        self, mock_get_storage: MagicMock, client: TestClient
    ) -> None:
        """Test health endpoint when storage health check fails."""
        mock_storage = AsyncMock()
        mock_storage.health_check.side_effect = Exception("Connection failed")
        mock_get_storage.return_value = mock_storage

        response = client.get("/metrics/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "error"
        assert data["storage_backend"] == "duckdb"
        assert "Connection failed" in data["error"]

    def test_status_endpoint(self, client: TestClient) -> None:
        """Test status endpoint returns observability system info."""
        response = client.get("/metrics/status")

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

            response = client.get("/metrics/prometheus")

            # Should get 503 due to missing prometheus_client
            assert response.status_code == 503


@pytest.mark.integration
class TestMetricsAPIIntegration:
    """Integration tests for metrics API with actual DuckDB storage."""

    @pytest.fixture
    def client_with_duckdb(self, test_settings: Settings) -> TestClient:
        """Create test client with DuckDB storage enabled."""
        app = create_app(test_settings)
        return TestClient(app)

    def test_full_metrics_pipeline_integration(
        self, client_with_duckdb: TestClient
    ) -> None:
        """Test full metrics pipeline from storage to API."""
        # Note: This test requires actual DuckDB installation
        # It tests the integration but may be skipped if DuckDB not available

        # Test health endpoint
        response = client_with_duckdb.get("/metrics/health")
        assert response.status_code == 200

        # If DuckDB is available, storage should be healthy
        data = response.json()
        if data["status"] == "healthy":
            # Test analytics endpoint
            analytics_response = client_with_duckdb.get(
                "/metrics/analytics", params={"hours": 1}
            )
            assert analytics_response.status_code == 200

            analytics_data = analytics_response.json()
            assert "summary" in analytics_data
            assert "hourly_data" in analytics_data
            assert "model_stats" in analytics_data


@pytest.mark.unit
class TestSSEStreamingEndpoint:
    """Test SSE streaming endpoint functionality."""

    @pytest.fixture
    def client(self, test_settings: Settings) -> TestClient:
        """Create test client."""
        app = create_app(test_settings)
        return TestClient(app)

    def test_sse_stream_endpoint_basic(self, client: TestClient) -> None:
        """Test basic SSE stream endpoint functionality."""
        # Test the new event-driven SSE endpoint
        with client.stream("GET", "/metrics/stream") as response:
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/event-stream"
            assert response.headers["cache-control"] == "no-cache"
            assert response.headers["connection"] == "keep-alive"

            # Read first event (should be connection event)
            events = []
            for line in response.iter_lines():
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
        response = client.get("/metrics/stream")

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
        with client.stream("GET", "/metrics/stream") as response:
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
        with client.stream("GET", "/metrics/stream") as response:
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
            with client_with_sse.stream("GET", "/metrics/stream") as response:
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
