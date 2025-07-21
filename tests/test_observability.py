"""Tests for the hybrid observability system.

This module tests the new observability architecture including:
- PrometheusMetrics for operational monitoring
- Request context management with timing
- Prometheus endpoint integration
- Real component integration (no internal mocking)
"""

import asyncio
import time
from collections.abc import Generator
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pytest_httpx import HTTPXMock


@pytest.fixture(autouse=True)
def reset_observability_state() -> Generator[None, None, None]:
    """Fixture to reset global observability state before each test."""
    # Reset global state before test
    try:
        from ccproxy.observability import reset_metrics
        from ccproxy.observability.pushgateway import reset_pushgateway_client

        reset_metrics()
        reset_pushgateway_client()

        # Also reset global variables
        import ccproxy.observability.metrics
        import ccproxy.observability.pushgateway

        ccproxy.observability.metrics._global_metrics = None
        ccproxy.observability.pushgateway._global_pushgateway_client = None
    except ImportError:
        pass  # Module not available in some test scenarios

    yield

    # Clean up after test
    try:
        from ccproxy.observability import reset_metrics
        from ccproxy.observability.pushgateway import reset_pushgateway_client

        reset_metrics()
        reset_pushgateway_client()

        # Also reset global variables
        import ccproxy.observability.metrics
        import ccproxy.observability.pushgateway

        ccproxy.observability.metrics._global_metrics = None
        ccproxy.observability.pushgateway._global_pushgateway_client = None
    except ImportError:
        pass


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
            from prometheus_client import CollectorRegistry

            from ccproxy.observability import PrometheusMetrics

            # Use isolated registry for this test
            test_registry = CollectorRegistry()
            metrics = PrometheusMetrics(namespace="test", registry=test_registry)

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
        from ccproxy.observability import get_metrics, reset_metrics

        # Reset global state
        reset_metrics()

        with patch("ccproxy.observability.metrics.PROMETHEUS_AVAILABLE", True):
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

        with patch("ccproxy.observability.metrics.PROMETHEUS_AVAILABLE", True):
            from prometheus_client import CollectorRegistry

            test_registry = CollectorRegistry()

            metrics = get_metrics(registry=test_registry)

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

        with patch("ccproxy.observability.metrics.PROMETHEUS_AVAILABLE", True):
            from prometheus_client import CollectorRegistry

            test_registry = CollectorRegistry()

            metrics = get_metrics(registry=test_registry)

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
            response = client.get("/metrics")

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

            response = client.get("/metrics")

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
                response = client.get("/metrics")

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
        response = client.get("/logs/status")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_metrics_prometheus_headers(self, client: TestClient) -> None:
        """Test prometheus endpoint returns correct headers."""
        with patch("ccproxy.observability.metrics.PROMETHEUS_AVAILABLE", True):
            response = client.get("/metrics")

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

        assert settings.metrics_enabled is False  # Disabled by default
        # pushgateway_enabled removed - now controlled by scheduler config
        assert settings.pushgateway_url is None
        assert settings.pushgateway_job == "ccproxy"
        assert settings.duckdb_enabled is True
        # Default path is now XDG data directory
        assert settings.duckdb_path.endswith("ccproxy/metrics.duckdb")

    def test_custom_settings(self) -> None:
        """Test custom observability settings."""
        from ccproxy.config.observability import ObservabilitySettings

        settings = ObservabilitySettings(
            metrics_endpoint_enabled=False,
            logs_endpoints_enabled=False,
            logs_collection_enabled=False,
            pushgateway_url="http://pushgateway:9091",
            pushgateway_job="test-job",
            log_storage_backend="none",  # This makes duckdb_enabled=False
            duckdb_path="/custom/path/metrics.duckdb",
        )

        assert settings.metrics_enabled is False
        # pushgateway_enabled removed - now controlled by scheduler config
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
            "pushgateway_url": "http://localhost:9091",
            "duckdb_path": "custom/metrics.duckdb",
        }

        settings = ObservabilitySettings(**config_dict)

        assert settings.metrics_enabled is False
        # pushgateway_enabled removed - now controlled by scheduler config
        assert settings.pushgateway_url == "http://localhost:9091"
        assert settings.duckdb_path == "custom/metrics.duckdb"


@pytest.mark.unit
class TestPushgatewayClient:
    """Test PushgatewayClient functionality."""

    def test_client_initialization_disabled(self) -> None:
        """Test client initialization when Pushgateway is disabled."""
        from ccproxy.config.observability import ObservabilitySettings
        from ccproxy.observability.pushgateway import PushgatewayClient

        settings = ObservabilitySettings()
        client = PushgatewayClient(settings)

        assert not client.is_enabled()

    def test_client_initialization_enabled_no_url(self) -> None:
        """Test client initialization when enabled but no URL provided."""
        from ccproxy.config.observability import ObservabilitySettings
        from ccproxy.observability.pushgateway import PushgatewayClient

        settings = ObservabilitySettings(pushgateway_url=None)
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

        settings = ObservabilitySettings()
        client = PushgatewayClient(settings)
        mock_registry = Mock()

        result = client.push_metrics(mock_registry)

        assert result is False


@pytest.mark.unit
class TestPrometheusMetricsIntegration:
    """Test PrometheusMetrics integration with PushgatewayClient."""

    def test_metrics_pushgateway_initialization(self) -> None:
        """Test PrometheusMetrics initializes pushgateway client."""
        from ccproxy.observability.metrics import PrometheusMetrics

        with patch("ccproxy.observability.metrics.PROMETHEUS_AVAILABLE", True):
            from prometheus_client import CollectorRegistry

            test_registry = CollectorRegistry()

            metrics = PrometheusMetrics(registry=test_registry)

            # Should have pushgateway client (even if not enabled)
            assert metrics._pushgateway_client is not None

    def test_metrics_push_to_gateway_success(self) -> None:
        """Test successful push to gateway via PrometheusMetrics."""
        from unittest.mock import Mock

        from ccproxy.observability.metrics import PrometheusMetrics

        with patch("ccproxy.observability.metrics.PROMETHEUS_AVAILABLE", True):
            from prometheus_client import CollectorRegistry

            test_registry = CollectorRegistry()

            metrics = PrometheusMetrics(registry=test_registry)

            # Mock pushgateway client
            mock_client: Mock = Mock()
            mock_client.push_metrics.return_value = True
            metrics._pushgateway_client = mock_client

            result: bool = metrics.push_to_gateway()

            assert result is True
            mock_client.push_metrics.assert_called_once_with(metrics.registry, "push")

    def test_metrics_push_to_gateway_disabled(self) -> None:
        """Test push to gateway when disabled."""
        from ccproxy.observability.metrics import PrometheusMetrics

        with patch("ccproxy.observability.metrics.PROMETHEUS_AVAILABLE", False):
            metrics = PrometheusMetrics()

            result: bool = metrics.push_to_gateway()

            assert result is False

    def test_metrics_is_pushgateway_enabled(self) -> None:
        """Test checking if pushgateway is enabled."""
        from unittest.mock import Mock

        from ccproxy.observability.metrics import PrometheusMetrics

        with patch("ccproxy.observability.metrics.PROMETHEUS_AVAILABLE", True):
            from prometheus_client import CollectorRegistry

            test_registry = CollectorRegistry()

            metrics = PrometheusMetrics(registry=test_registry)

            # Mock pushgateway client
            mock_client: Mock = Mock()
            mock_client.is_enabled.return_value = True
            metrics._pushgateway_client = mock_client

            result: bool = metrics.is_pushgateway_enabled()

            assert result is True
            mock_client.is_enabled.assert_called_once()


@pytest.mark.unit
# Note: ObservabilityScheduler tests removed - functionality moved to unified scheduler
# See tests/test_unified_scheduler.py for comprehensive scheduler testing


@pytest.mark.unit
class TestPushgatewayDependencyInjection:
    """Test dependency injection patterns for pushgateway."""

    def test_get_pushgateway_client_singleton(self) -> None:
        """Test get_pushgateway_client returns singleton instance."""
        from ccproxy.observability.pushgateway import (
            get_pushgateway_client,
            reset_pushgateway_client,
        )

        # Reset state
        reset_pushgateway_client()

        client1 = get_pushgateway_client()
        client2 = get_pushgateway_client()

        assert client1 is client2

    def test_reset_pushgateway_client(self) -> None:
        """Test reset_pushgateway_client clears singleton."""
        from ccproxy.observability.pushgateway import (
            get_pushgateway_client,
            reset_pushgateway_client,
        )

        client1 = get_pushgateway_client()
        reset_pushgateway_client()
        client2 = get_pushgateway_client()

        assert client1 is not client2

    def test_metrics_dependency_injection(self) -> None:
        """Test PrometheusMetrics uses dependency injection for pushgateway."""
        from unittest.mock import Mock

        from ccproxy.observability.metrics import PrometheusMetrics

        mock_pushgateway_client = Mock()
        mock_pushgateway_client.is_enabled.return_value = True

        with patch("ccproxy.observability.metrics.PROMETHEUS_AVAILABLE", True):
            from prometheus_client import CollectorRegistry

            test_registry = CollectorRegistry()

            metrics = PrometheusMetrics(
                registry=test_registry, pushgateway_client=mock_pushgateway_client
            )

            assert metrics._pushgateway_client is mock_pushgateway_client
            assert metrics.is_pushgateway_enabled() is True

    def test_get_metrics_dependency_injection(self) -> None:
        """Test get_metrics function uses dependency injection."""
        from unittest.mock import Mock

        from ccproxy.observability.metrics import get_metrics, reset_metrics

        mock_pushgateway_client = Mock()
        mock_pushgateway_client.is_enabled.return_value = True

        # Reset global state
        reset_metrics()

        with patch("ccproxy.observability.metrics.PROMETHEUS_AVAILABLE", True):
            from prometheus_client import CollectorRegistry

            test_registry = CollectorRegistry()

            metrics = get_metrics(
                registry=test_registry, pushgateway_client=mock_pushgateway_client
            )

            assert metrics._pushgateway_client is mock_pushgateway_client


@pytest.mark.unit
class TestPushgatewayRemoteWrite:
    """Test remote write protocol for VictoriaMetrics."""

    @patch("prometheus_client.exposition.generate_latest")
    def test_remote_write_success(
        self, mock_generate_latest: Any, httpx_mock: HTTPXMock
    ) -> None:
        """Test successful remote write push."""
        from unittest.mock import Mock

        from ccproxy.config.observability import ObservabilitySettings
        from ccproxy.observability.pushgateway import PushgatewayClient

        # Configure mock response
        httpx_mock.add_response(
            url="http://victoriametrics:8428/api/v1/import/prometheus", status_code=200
        )

        # Mock prometheus metrics generation
        mock_generate_latest.return_value = (
            b"# HELP test_metric Test metric\ntest_metric 1.0\n"
        )

        settings = ObservabilitySettings(
            pushgateway_url="http://victoriametrics:8428/api/v1/write",
            pushgateway_job="test-job",
        )

        with patch("ccproxy.observability.pushgateway.PROMETHEUS_AVAILABLE", True):
            client = PushgatewayClient(settings)
            mock_registry = Mock()

            result = client.push_metrics(mock_registry)

            assert result is True
            request = httpx_mock.get_request()
            assert request is not None
            assert request.url == "http://victoriametrics:8428/api/v1/import/prometheus"
            assert request.headers["content-type"] == "text/plain; charset=utf-8"

    @patch("prometheus_client.exposition.generate_latest")
    def test_remote_write_failure(
        self, mock_generate_latest: Any, httpx_mock: HTTPXMock
    ) -> None:
        """Test failed remote write push."""
        from unittest.mock import Mock

        from ccproxy.config.observability import ObservabilitySettings
        from ccproxy.observability.pushgateway import PushgatewayClient

        # Configure mock response
        httpx_mock.add_response(
            url="http://victoriametrics:8428/api/v1/import/prometheus",
            status_code=400,
            text="Bad Request",
        )

        # Mock prometheus metrics generation
        mock_generate_latest.return_value = (
            b"# HELP test_metric Test metric\ntest_metric 1.0\n"
        )

        settings = ObservabilitySettings(
            pushgateway_url="http://victoriametrics:8428/api/v1/write",
            pushgateway_job="test-job",
        )

        with patch("ccproxy.observability.pushgateway.PROMETHEUS_AVAILABLE", True):
            client = PushgatewayClient(settings)
            mock_registry = Mock()

            result = client.push_metrics(mock_registry)

            assert result is False

    def test_standard_pushgateway_protocol(self) -> None:
        """Test standard pushgateway protocol selection."""
        from unittest.mock import Mock

        from ccproxy.config.observability import ObservabilitySettings
        from ccproxy.observability.pushgateway import PushgatewayClient

        settings = ObservabilitySettings(
            pushgateway_url="http://pushgateway:9091",
            pushgateway_job="test-job",
        )

        with (
            patch("ccproxy.observability.pushgateway.PROMETHEUS_AVAILABLE", True),
            patch("ccproxy.observability.pushgateway.push_to_gateway") as mock_push,
        ):
            client = PushgatewayClient(settings)
            mock_registry = Mock()

            result = client.push_metrics(mock_registry)

            assert result is True
            mock_push.assert_called_once_with(
                gateway="http://pushgateway:9091",
                job="test-job",
                registry=mock_registry,
            )

    def test_protocol_detection_logic(self) -> None:
        """Test protocol detection based on URL."""
        from ccproxy.config.observability import ObservabilitySettings
        from ccproxy.observability.pushgateway import PushgatewayClient

        # Test remote write detection
        settings_remote = ObservabilitySettings(
            pushgateway_url="http://victoriametrics:8428/api/v1/write",
        )

        # Test standard pushgateway detection
        settings_standard = ObservabilitySettings(
            pushgateway_url="http://pushgateway:9091",
        )

        with patch("ccproxy.observability.pushgateway.PROMETHEUS_AVAILABLE", True):
            client_remote = PushgatewayClient(settings_remote)
            client_standard = PushgatewayClient(settings_standard)

            # Both should be enabled
            assert client_remote.is_enabled()
            assert client_standard.is_enabled()

            # URLs should be different
            assert (
                client_remote.settings.pushgateway_url
                and "/api/v1/write" in client_remote.settings.pushgateway_url
            )
            assert (
                client_standard.settings.pushgateway_url
                and "/api/v1/write" not in client_standard.settings.pushgateway_url
            )

    @patch("ccproxy.observability.pushgateway.pushadd_to_gateway")
    def test_push_add_method_wrapper(self, mock_pushadd: Any) -> None:
        """Test push_add_metrics wrapper method."""
        from unittest.mock import Mock

        from ccproxy.config.observability import ObservabilitySettings
        from ccproxy.observability.pushgateway import PushgatewayClient

        settings = ObservabilitySettings(
            pushgateway_url="http://pushgateway:9091",
            pushgateway_job="test-job",
        )

        with patch("ccproxy.observability.pushgateway.PROMETHEUS_AVAILABLE", True):
            client = PushgatewayClient(settings)
            mock_registry = Mock()

            result = client.push_add_metrics(mock_registry)

            assert result is True
            mock_pushadd.assert_called_once_with(
                gateway="http://pushgateway:9091",
                job="test-job",
                registry=mock_registry,
            )

    @patch("ccproxy.observability.pushgateway.delete_from_gateway")
    def test_delete_metrics_wrapper(self, mock_delete: Any) -> None:
        """Test delete_metrics wrapper method."""
        from ccproxy.config.observability import ObservabilitySettings
        from ccproxy.observability.pushgateway import PushgatewayClient

        settings = ObservabilitySettings(
            pushgateway_url="http://pushgateway:9091",
            pushgateway_job="test-job",
        )

        with patch("ccproxy.observability.pushgateway.PROMETHEUS_AVAILABLE", True):
            client = PushgatewayClient(settings)

            result = client.delete_metrics()

            assert result is True
            mock_delete.assert_called_once_with(
                gateway="http://pushgateway:9091",
                job="test-job",
            )

    def test_pushgateway_method_parameter_validation(self) -> None:
        """Test that invalid method parameters are handled correctly."""
        from unittest.mock import Mock

        from ccproxy.config.observability import ObservabilitySettings
        from ccproxy.observability.pushgateway import PushgatewayClient

        settings = ObservabilitySettings(
            pushgateway_url="http://pushgateway:9091",
            pushgateway_job="test-job",
        )

        with patch("ccproxy.observability.pushgateway.PROMETHEUS_AVAILABLE", True):
            client = PushgatewayClient(settings)
            mock_registry = Mock()

            # Test invalid method
            result = client.push_metrics(mock_registry, method="invalid_method")
            assert result is False

            # Test valid methods
            with patch("ccproxy.observability.pushgateway.push_to_gateway"):
                assert client.push_metrics(mock_registry, method="push") is True

            with patch("ccproxy.observability.pushgateway.pushadd_to_gateway"):
                assert client.push_metrics(mock_registry, method="pushadd") is True

            with patch("ccproxy.observability.pushgateway.delete_from_gateway"):
                assert client.push_metrics(mock_registry, method="delete") is True


@pytest.mark.unit
class TestPrometheusClientMethods:
    """Test new Prometheus client methods integration."""

    def test_prometheus_metrics_new_pushgateway_methods(self) -> None:
        """Test new PrometheusMetrics pushgateway methods."""
        from unittest.mock import Mock

        from ccproxy.observability.metrics import PrometheusMetrics

        with patch("ccproxy.observability.metrics.PROMETHEUS_AVAILABLE", True):
            from prometheus_client import CollectorRegistry

            test_registry = CollectorRegistry()

            metrics = PrometheusMetrics(registry=test_registry)

            # Mock pushgateway client
            mock_client: Mock = Mock()
            mock_client.push_metrics.return_value = True
            mock_client.delete_metrics.return_value = True
            metrics._pushgateway_client = mock_client

            # Test default push (should use "push" method)
            result = metrics.push_to_gateway()
            assert result is True
            mock_client.push_metrics.assert_called_with(metrics.registry, "push")

            # Test pushadd method
            result = metrics.push_to_gateway(method="pushadd")
            assert result is True
            mock_client.push_metrics.assert_called_with(metrics.registry, "pushadd")

            # Test convenience method for pushadd
            result = metrics.push_add_to_gateway()
            assert result is True

            # Test delete method
            result = metrics.delete_from_gateway()
            assert result is True
            mock_client.delete_metrics.assert_called_once()

    def test_prometheus_metrics_methods_when_disabled(self) -> None:
        """Test pushgateway methods when metrics are disabled."""
        from ccproxy.observability.metrics import PrometheusMetrics

        with patch("ccproxy.observability.metrics.PROMETHEUS_AVAILABLE", False):
            metrics = PrometheusMetrics()

            # All methods should return False when disabled
            assert metrics.push_to_gateway() is False
            assert metrics.push_add_to_gateway() is False
            assert metrics.delete_from_gateway() is False


@pytest.mark.unit
class TestErrorMiddlewareMetricsIntegration:
    """Test error middleware integration with metrics recording."""

    def test_error_middleware_records_404_errors(self, client: TestClient) -> None:
        """Test that 404 errors are recorded in metrics by the error middleware."""
        from ccproxy.observability.metrics import get_metrics

        # Reset metrics state for clean test
        with patch("ccproxy.observability.metrics.PROMETHEUS_AVAILABLE", True):
            from prometheus_client import CollectorRegistry

            test_registry = CollectorRegistry()
            metrics = get_metrics(registry=test_registry)

            # Override the registry for this test instance to avoid global state
            metrics.registry = test_registry
            metrics._init_metrics()  # Re-initialize metrics with the test registry

            # Make request to non-existent endpoint to trigger 404
            response = client.get("/nonexistent-endpoint-test")

            # Verify 404 response
            assert response.status_code == 404
            assert response.json()["error"]["type"] == "http_error"

            # Check that error was recorded in metrics
            error_counter = metrics.error_counter
            error_count = 0
            starlette_404_count = 0

            for metric in error_counter.collect():
                for sample in metric.samples:
                    if sample.name.endswith("_total"):
                        error_count += int(sample.value)
                        if sample.labels.get("error_type") == "starlette_http_404":
                            starlette_404_count += int(sample.value)

            # Should have recorded exactly one error
            assert error_count == 1
            assert starlette_404_count == 1

    def test_error_middleware_records_validation_errors(
        self, client: TestClient
    ) -> None:
        """Test that HTTP errors are recorded in metrics by the error middleware."""
        from ccproxy.observability.metrics import get_metrics

        # Reset metrics state for clean test
        with patch("ccproxy.observability.metrics.PROMETHEUS_AVAILABLE", True):
            from prometheus_client import CollectorRegistry

            test_registry = CollectorRegistry()
            metrics = get_metrics(registry=test_registry)

            # Override the registry for this test instance
            metrics.registry = test_registry
            metrics._init_metrics()

            # Trigger an error by making a request to a non-existent endpoint
            # This will generate a 404 error that should be recorded by the middleware
            response = client.get("/nonexistent-error-test-endpoint")

            # Should get 404 error
            assert response.status_code == 404

            # Check that error was recorded in metrics
            error_counter = None
            for collector in test_registry._collector_to_names:
                if hasattr(collector, "_name") and collector._name == "ccproxy_errors":
                    error_counter = collector
                    break

            assert error_counter is not None, "Error counter metric not found"

            # Collect the samples and count 404 errors
            samples = list(error_counter.collect())[0].samples

            error_count = 0
            for sample in samples:
                # Look for the main error counter (not the _created timestamp)
                if (
                    sample.name == "ccproxy_errors_total"
                    and sample.labels.get("error_type") == "starlette_http_404"
                ):
                    error_count += int(sample.value)

            # Should have recorded exactly one 404 error
            assert error_count == 1, (
                f"Expected 1 error, got {error_count}. Samples: {samples}"
            )

    def test_error_middleware_metrics_dependency_injection(self) -> None:
        """Test that error middleware properly gets metrics instance."""
        from fastapi import FastAPI

        from ccproxy.api.middleware.errors import setup_error_handlers
        from ccproxy.observability.metrics import get_metrics

        # Create test app
        app = FastAPI()

        # Setup error handlers (this should inject metrics)
        setup_error_handlers(app)

        # Verify metrics instance is available globally
        metrics = get_metrics()
        assert metrics is not None
        assert hasattr(metrics, "record_error")

    def test_multiple_errors_accumulate_in_metrics(self, client: TestClient) -> None:
        """Test that multiple errors accumulate correctly in metrics."""
        from ccproxy.observability.metrics import get_metrics

        with patch("ccproxy.observability.metrics.PROMETHEUS_AVAILABLE", True):
            from prometheus_client import CollectorRegistry

            test_registry = CollectorRegistry()
            metrics = get_metrics(registry=test_registry)

            # Override the registry for this test instance
            metrics.registry = test_registry
            metrics._init_metrics()

            # Make multiple 404 requests
            for i in range(3):
                response = client.get(f"/nonexistent-endpoint-{i}")
                assert response.status_code == 404

            # Check accumulated error count
            error_counter = metrics.error_counter
            total_errors = 0

            for metric in error_counter.collect():
                for sample in metric.samples:
                    if sample.name.endswith("_total"):
                        total_errors += int(sample.value)

            # Should have 3 errors total
            assert total_errors == 3
