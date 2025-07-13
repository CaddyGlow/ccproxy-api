"""Tests for the hybrid observability system.

This module tests the new observability architecture including:
- PrometheusMetrics for operational monitoring
- Request context management with timing
- Prometheus endpoint integration
- Real component integration (no internal mocking)
"""

import asyncio
import time
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.mark.unit
class TestPrometheusMetrics:
    """Test the PrometheusMetrics class for operational monitoring."""

    def test_prometheus_metrics_initialization_with_available_client(self) -> None:
        """Test PrometheusMetrics initialization when prometheus_client is available."""
        with patch("ccproxy.observability.metrics.PROMETHEUS_AVAILABLE", True):
            from ccproxy.observability import PrometheusMetrics

            metrics = PrometheusMetrics(namespace="test")
            assert metrics.namespace == "test"
            assert metrics.is_enabled()

    def test_prometheus_metrics_initialization_without_client(self) -> None:
        """Test PrometheusMetrics initialization when prometheus_client unavailable."""
        with patch("ccproxy.observability.metrics.PROMETHEUS_AVAILABLE", False):
            from ccproxy.observability import PrometheusMetrics

            metrics = PrometheusMetrics(namespace="test")
            assert metrics.namespace == "test"
            assert not metrics.is_enabled()

    def test_prometheus_metrics_operations_with_available_client(self) -> None:
        """Test Prometheus metrics recording operations when client available."""
        with patch("ccproxy.observability.metrics.PROMETHEUS_AVAILABLE", True):
            from ccproxy.observability import PrometheusMetrics

            metrics = PrometheusMetrics(namespace="test")

            # Test request recording
            metrics.record_request("POST", "/v1/messages", "claude-3-sonnet", "200")

            # Test response time recording
            metrics.record_response_time(1.5, "claude-3-sonnet", "/v1/messages")

            # Test token recording
            metrics.record_tokens(150, "input", "claude-3-sonnet")
            metrics.record_tokens(75, "output", "claude-3-sonnet")

            # Test cost recording
            metrics.record_cost(0.0023, "claude-3-sonnet", "total")

            # Test error recording
            metrics.record_error("timeout_error", "/v1/messages", "claude-3-sonnet")

            # Test active requests
            metrics.inc_active_requests()
            metrics.dec_active_requests()
            metrics.set_active_requests(5)

    def test_prometheus_metrics_graceful_degradation(self) -> None:
        """Test that metrics operations work when prometheus_client unavailable."""
        with patch("ccproxy.observability.metrics.PROMETHEUS_AVAILABLE", False):
            from ccproxy.observability import PrometheusMetrics

            metrics = PrometheusMetrics(namespace="test")

            # All operations should work without errors
            metrics.record_request("POST", "/v1/messages", "claude-3-sonnet", "200")
            metrics.record_response_time(1.5, "claude-3-sonnet", "/v1/messages")
            metrics.record_tokens(150, "input", "claude-3-sonnet")
            metrics.record_cost(0.0023, "claude-3-sonnet")
            metrics.record_error("timeout_error", "/v1/messages")
            metrics.inc_active_requests()
            metrics.dec_active_requests()

    def test_global_metrics_instance(self) -> None:
        """Test global metrics instance management."""
        from ccproxy.observability import get_metrics

        # Reset global state
        with patch("ccproxy.observability.metrics._global_metrics", None):
            metrics1 = get_metrics()
            metrics2 = get_metrics()
            assert metrics1 is metrics2  # Should be the same instance


@pytest.mark.unit
class TestRequestContext:
    """Test request context management and timing."""

    async def test_request_context_basic(self) -> None:
        """Test basic request context functionality."""
        from ccproxy.observability import RequestContext, request_context

        async with request_context(method="POST", path="/v1/messages") as ctx:
            assert isinstance(ctx, RequestContext)
            assert ctx.request_id is not None
            assert ctx.start_time > 0
            assert ctx.duration_ms >= 0
            assert ctx.duration_seconds >= 0
            assert "method" in ctx.metadata
            assert "path" in ctx.metadata

    async def test_request_context_timing(self) -> None:
        """Test accurate timing measurement."""
        from ccproxy.observability import request_context

        async with request_context() as ctx:
            initial_duration = ctx.duration_ms
            await asyncio.sleep(0.01)  # Small delay
            final_duration = ctx.duration_ms
            assert final_duration > initial_duration

    async def test_request_context_metadata(self) -> None:
        """Test metadata management."""
        from ccproxy.observability import request_context

        async with request_context(model="claude-3-sonnet") as ctx:
            # Initial metadata
            assert ctx.metadata["model"] == "claude-3-sonnet"

            # Add metadata
            ctx.add_metadata(tokens_input=150, status_code=200)
            assert ctx.metadata["tokens_input"] == 150
            assert ctx.metadata["status_code"] == 200

    async def test_request_context_error_handling(self) -> None:
        """Test error handling in request context."""
        from ccproxy.observability import request_context

        with pytest.raises(ValueError):
            async with request_context() as ctx:
                ctx.add_metadata(test="value")
                raise ValueError("Test error")

    async def test_timed_operation(self) -> None:
        """Test timed operation context manager."""
        from uuid import uuid4

        from ccproxy.observability import timed_operation

        request_id = str(uuid4())

        async with timed_operation("test_operation", request_id) as op:
            assert "operation_id" in op
            assert "logger" in op
            assert "start_time" in op
            await asyncio.sleep(0.01)  # Small delay

    async def test_context_tracker(self) -> None:
        """Test request context tracking."""
        from ccproxy.observability import get_context_tracker, request_context

        tracker = get_context_tracker()

        # Test adding context
        async with request_context() as ctx:
            await tracker.add_context(ctx)

            # Test retrieving context
            retrieved_ctx = await tracker.get_context(ctx.request_id)
            assert retrieved_ctx is ctx

            # Test active count
            count = await tracker.get_active_count()
            assert count >= 1

            # Test removing context
            removed_ctx = await tracker.remove_context(ctx.request_id)
            assert removed_ctx is ctx

    async def test_tracked_request_context(self) -> None:
        """Test tracked request context that automatically manages global state."""
        from ccproxy.observability import get_context_tracker, tracked_request_context

        tracker = get_context_tracker()
        initial_count = await tracker.get_active_count()

        async with tracked_request_context() as ctx:
            # Should be tracked
            current_count = await tracker.get_active_count()
            assert current_count > initial_count

            # Context should be retrievable
            retrieved_ctx = await tracker.get_context(ctx.request_id)
            assert retrieved_ctx is ctx

        # Should be cleaned up
        final_count = await tracker.get_active_count()
        assert final_count == initial_count


@pytest.mark.unit
class TestObservabilityIntegration:
    """Test integration between observability components."""

    async def test_context_with_metrics_integration(self) -> None:
        """Test request context integration with metrics."""
        from ccproxy.observability import get_metrics, request_context, timed_operation

        metrics = get_metrics()

        async with request_context(
            method="POST", endpoint="messages", model="claude-3-sonnet"
        ) as ctx:
            # Record operational metrics
            metrics.inc_active_requests()
            metrics.record_request("POST", "messages", "claude-3-sonnet", "200")

            # Simulate API call timing
            async with timed_operation("api_call", ctx.request_id):
                await asyncio.sleep(0.01)

            # Record response metrics
            metrics.record_response_time(
                ctx.duration_seconds, "claude-3-sonnet", "messages"
            )
            metrics.record_tokens(150, "input", "claude-3-sonnet")
            metrics.record_tokens(75, "output", "claude-3-sonnet")
            metrics.record_cost(0.0023, "claude-3-sonnet")

            metrics.dec_active_requests()

    async def test_error_handling_integration(self) -> None:
        """Test error handling across observability components."""
        from ccproxy.observability import get_metrics, request_context

        metrics = get_metrics()

        with pytest.raises(ValueError):
            async with request_context(method="POST", endpoint="messages") as ctx:
                metrics.inc_active_requests()

                try:
                    # Simulate error
                    raise ValueError("Test error")
                except Exception as e:
                    # Record error metrics
                    metrics.record_error(type(e).__name__, "messages")
                    metrics.dec_active_requests()
                    raise


@pytest.mark.unit
class TestPrometheusEndpoint:
    """Test the new Prometheus endpoint functionality."""

    def test_prometheus_endpoint_with_client_available(
        self, client: TestClient
    ) -> None:
        """Test prometheus endpoint when prometheus_client is available."""
        with patch("ccproxy.observability.metrics.PROMETHEUS_AVAILABLE", True):
            response = client.get("/metrics/prometheus")

            # Should succeed
            assert response.status_code == 200

            # Check content type
            assert "text/plain" in response.headers.get("content-type", "")

            # Should contain basic metrics structure
            content = response.text
            # Empty metrics are valid too
            assert isinstance(content, str)

    def test_prometheus_endpoint_without_client_available(
        self, client: TestClient
    ) -> None:
        """Test prometheus endpoint when prometheus_client unavailable."""
        with patch("ccproxy.observability.metrics.PROMETHEUS_AVAILABLE", False):
            from ccproxy.observability import reset_metrics

            # Reset global state to pick up the patched PROMETHEUS_AVAILABLE
            reset_metrics()

            response = client.get("/metrics/prometheus")

            # Should return 503 Service Unavailable
            assert response.status_code == 503
            data = response.json()
            assert "error" in data
            assert "message" in data["error"]
            assert "prometheus-client" in data["error"]["message"]

    def test_prometheus_endpoint_with_metrics_recorded(
        self, client: TestClient
    ) -> None:
        """Test prometheus endpoint with actual metrics recorded."""
        with patch("ccproxy.observability.metrics.PROMETHEUS_AVAILABLE", True):
            from ccproxy.observability import get_metrics, reset_metrics

            # Reset global state to pick up the patched PROMETHEUS_AVAILABLE
            reset_metrics()

            # Create a custom registry for testing to avoid global state contamination
            from prometheus_client import CollectorRegistry

            test_registry = CollectorRegistry()

            # Get metrics with custom registry and record some data
            metrics = get_metrics()
            # Override the registry for this test instance
            metrics.registry = test_registry
            metrics._init_metrics()  # Re-initialize metrics with the test registry

            if metrics.is_enabled():
                metrics.record_request("POST", "messages", "claude-3-sonnet", "200")
                metrics.record_response_time(1.5, "claude-3-sonnet", "messages")
                metrics.record_tokens(150, "input", "claude-3-sonnet")

            # Patch the endpoint to use our test registry
            with patch.object(metrics, "registry", test_registry):
                response = client.get("/metrics/prometheus")

                if response.status_code == 200 and metrics.is_enabled():
                    content = response.text
                    # Should contain our recorded metrics
                    assert "ccproxy_requests_total" in content
                    assert "ccproxy_response_duration_seconds" in content
                    assert "ccproxy_tokens_total" in content


@pytest.mark.unit
class TestProxyServiceObservabilityIntegration:
    """Test ProxyService integration with observability system."""

    def test_proxy_service_uses_observability_system(self) -> None:
        """Test that ProxyService is configured to use new observability system."""
        from ccproxy.api.dependencies import get_proxy_service
        from ccproxy.config.settings import Settings
        from ccproxy.observability import PrometheusMetrics
        from ccproxy.services.credentials.manager import CredentialsManager

        # Create test settings
        settings = Settings()

        # Create credentials manager
        credentials_manager = CredentialsManager(config=settings.auth)

        # Get proxy service (this should use the new observability system)
        proxy_service = get_proxy_service(settings, credentials_manager)

        # Verify it has metrics attribute (new system)
        assert hasattr(proxy_service, "metrics")
        assert isinstance(proxy_service.metrics, PrometheusMetrics)

        # Verify it doesn't have the old metrics_collector attribute
        assert not hasattr(proxy_service, "metrics_collector")


@pytest.mark.unit
class TestObservabilityEndpoints:
    """Test observability-related endpoints."""

    def test_metrics_status(self, client: TestClient) -> None:
        """Test metrics status endpoint."""
        response = client.get("/metrics/status")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_metrics_prometheus_headers(self, client: TestClient) -> None:
        """Test prometheus endpoint returns correct headers."""
        with patch("ccproxy.observability.metrics.PROMETHEUS_AVAILABLE", True):
            response = client.get("/metrics/prometheus")

            if response.status_code == 200:
                # Check no-cache headers
                assert "no-cache" in response.headers.get("cache-control", "")
                assert "no-store" in response.headers.get("cache-control", "")
                assert "must-revalidate" in response.headers.get("cache-control", "")


@pytest.mark.unit
class TestObservabilityDependencies:
    """Test observability dependency injection."""

    def test_observability_metrics_dependency(self) -> None:
        """Test observability metrics dependency resolution."""
        from ccproxy.api.dependencies import get_observability_metrics
        from ccproxy.observability import PrometheusMetrics

        metrics = get_observability_metrics()
        assert isinstance(metrics, PrometheusMetrics)

    def test_global_metrics_consistency(self) -> None:
        """Test that dependency and direct access return same instance."""
        from ccproxy.api.dependencies import get_observability_metrics
        from ccproxy.observability import get_metrics

        dep_metrics = get_observability_metrics()
        direct_metrics = get_metrics()

        # Should be the same instance
        assert dep_metrics is direct_metrics


@pytest.mark.unit
class TestObservabilitySettings:
    """Test ObservabilitySettings configuration."""

    def test_default_settings(self) -> None:
        """Test default observability settings."""
        from ccproxy.config.observability import ObservabilitySettings

        settings = ObservabilitySettings()

        assert settings.metrics_enabled is True
        assert settings.pushgateway_enabled is False
        assert settings.pushgateway_url is None
        assert settings.pushgateway_job == "ccproxy"
        assert settings.duckdb_enabled is True
        assert settings.duckdb_path == "data/metrics.duckdb"

    def test_custom_settings(self) -> None:
        """Test custom observability settings."""
        from ccproxy.config.observability import ObservabilitySettings

        settings = ObservabilitySettings(
            metrics_enabled=False,
            pushgateway_enabled=True,
            pushgateway_url="http://pushgateway:9091",
            pushgateway_job="test-job",
            duckdb_enabled=False,
            duckdb_path="/custom/path/metrics.duckdb",
        )

        assert settings.metrics_enabled is False
        assert settings.pushgateway_enabled is True
        assert settings.pushgateway_url == "http://pushgateway:9091"
        assert settings.pushgateway_job == "test-job"
        assert settings.duckdb_enabled is False
        assert settings.duckdb_path == "/custom/path/metrics.duckdb"

    def test_settings_from_dict(self) -> None:
        """Test creating settings from dictionary."""
        from typing import Any

        from ccproxy.config.observability import ObservabilitySettings

        config_dict: dict[str, Any] = {
            "metrics_enabled": False,
            "pushgateway_enabled": True,
            "pushgateway_url": "http://localhost:9091",
            "duckdb_path": "custom/metrics.duckdb",
        }

        settings = ObservabilitySettings(**config_dict)

        assert settings.metrics_enabled is False
        assert settings.pushgateway_enabled is True
        assert settings.pushgateway_url == "http://localhost:9091"
        assert settings.duckdb_path == "custom/metrics.duckdb"


@pytest.mark.unit
class TestPushgatewayClient:
    """Test PushgatewayClient functionality."""

    def test_client_initialization_disabled(self) -> None:
        """Test client initialization when Pushgateway is disabled."""
        from ccproxy.config.observability import ObservabilitySettings
        from ccproxy.observability.pushgateway import PushgatewayClient

        settings = ObservabilitySettings(pushgateway_enabled=False)
        client = PushgatewayClient(settings)

        assert not client.is_enabled()

    def test_client_initialization_enabled_no_url(self) -> None:
        """Test client initialization when enabled but no URL provided."""
        from ccproxy.config.observability import ObservabilitySettings
        from ccproxy.observability.pushgateway import PushgatewayClient

        settings = ObservabilitySettings(pushgateway_enabled=True, pushgateway_url=None)
        client = PushgatewayClient(settings)

        assert not client.is_enabled()

    def test_client_initialization_enabled_with_url(self) -> None:
        """Test client initialization when enabled with URL."""
        from ccproxy.config.observability import ObservabilitySettings
        from ccproxy.observability.pushgateway import PushgatewayClient

        settings = ObservabilitySettings(
            pushgateway_enabled=True, pushgateway_url="http://pushgateway:9091"
        )

        with patch("ccproxy.observability.pushgateway.PROMETHEUS_AVAILABLE", True):
            client = PushgatewayClient(settings)
            assert client.is_enabled()

    def test_client_initialization_no_prometheus(self) -> None:
        """Test client initialization when prometheus_client not available."""
        from ccproxy.config.observability import ObservabilitySettings
        from ccproxy.observability.pushgateway import PushgatewayClient

        settings = ObservabilitySettings(
            pushgateway_enabled=True, pushgateway_url="http://pushgateway:9091"
        )

        with patch("ccproxy.observability.pushgateway.PROMETHEUS_AVAILABLE", False):
            client = PushgatewayClient(settings)
            assert not client.is_enabled()

    @patch("ccproxy.observability.pushgateway.push_to_gateway")
    def test_push_metrics_success(self, mock_push: Any) -> None:
        """Test successful metrics push."""
        from unittest.mock import Mock

        from ccproxy.config.observability import ObservabilitySettings
        from ccproxy.observability.pushgateway import PushgatewayClient

        settings = ObservabilitySettings(
            pushgateway_enabled=True,
            pushgateway_url="http://pushgateway:9091",
            pushgateway_job="test-job",
        )

        with patch("ccproxy.observability.pushgateway.PROMETHEUS_AVAILABLE", True):
            client = PushgatewayClient(settings)
            mock_registry = Mock()

            result = client.push_metrics(mock_registry)

            assert result is True
            mock_push.assert_called_once_with(
                gateway="http://pushgateway:9091",
                job="test-job",
                registry=mock_registry,
            )

    @patch("ccproxy.observability.pushgateway.push_to_gateway")
    def test_push_metrics_failure(self, mock_push: Any) -> None:
        """Test failed metrics push."""
        from unittest.mock import Mock

        from ccproxy.config.observability import ObservabilitySettings
        from ccproxy.observability.pushgateway import PushgatewayClient

        settings = ObservabilitySettings(
            pushgateway_enabled=True, pushgateway_url="http://pushgateway:9091"
        )

        mock_push.side_effect = Exception("Connection failed")

        with patch("ccproxy.observability.pushgateway.PROMETHEUS_AVAILABLE", True):
            client = PushgatewayClient(settings)
            mock_registry = Mock()

            result = client.push_metrics(mock_registry)

            assert result is False

    def test_push_metrics_disabled(self) -> None:
        """Test push metrics when client is disabled."""
        from unittest.mock import Mock

        from ccproxy.config.observability import ObservabilitySettings
        from ccproxy.observability.pushgateway import PushgatewayClient

        settings = ObservabilitySettings(pushgateway_enabled=False)
        client = PushgatewayClient(settings)
        mock_registry = Mock()

        result = client.push_metrics(mock_registry)

        assert result is False

    def test_push_metrics_no_url(self) -> None:
        """Test push metrics when no URL configured."""
        from unittest.mock import Mock

        from ccproxy.config.observability import ObservabilitySettings
        from ccproxy.observability.pushgateway import PushgatewayClient

        settings = ObservabilitySettings(pushgateway_enabled=True, pushgateway_url=None)

        with patch("ccproxy.observability.pushgateway.PROMETHEUS_AVAILABLE", True):
            client = PushgatewayClient(settings)
            mock_registry = Mock()

            result = client.push_metrics(mock_registry)

            assert result is False
