"""Tests for pushgateway error handling improvements."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import Mock, patch

import pytest
from prometheus_client import CollectorRegistry

from ccproxy.config.observability import ObservabilitySettings
from ccproxy.observability.pushgateway import CircuitBreaker, PushgatewayClient


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    def test_circuit_breaker_initial_state(self) -> None:
        """Test circuit breaker starts in closed state."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=2.0)

        assert cb.can_execute() is True
        assert cb.state == "CLOSED"
        assert cb.failure_count == 0

    def test_circuit_breaker_opens_after_failures(self) -> None:
        """Test circuit breaker opens after failure threshold."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=2.0)

        # Record failures below threshold
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "CLOSED"
        assert cb.can_execute() is True

        # Third failure should open circuit
        cb.record_failure()
        assert cb.state == "OPEN"
        assert cb.can_execute() is False
        assert cb.failure_count == 3

    def test_circuit_breaker_recovery_after_timeout(self) -> None:
        """Test circuit breaker recovers after timeout."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)

        # Open circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "OPEN"

        # Wait for recovery timeout
        time.sleep(0.2)

        # Should be half-open now
        assert cb.can_execute() is True

        # Success should close it
        cb.record_success()
        assert cb.state == "CLOSED"
        assert cb.failure_count == 0

    def test_circuit_breaker_success_resets_failures(self) -> None:
        """Test success resets failure count."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=2.0)

        # Record some failures
        cb.record_failure()
        cb.record_failure()
        assert cb.failure_count == 2

        # Success should reset
        cb.record_success()
        assert cb.failure_count == 0
        assert cb.state == "CLOSED"


class TestPushgatewayClient:
    """Test PushgatewayClient with circuit breaker integration."""

    @pytest.fixture
    def settings(self) -> ObservabilitySettings:
        """Create test settings."""
        return ObservabilitySettings(
            pushgateway_url="http://localhost:9091",
            pushgateway_job="test-job",
        )

    @pytest.fixture
    def client(self, settings: ObservabilitySettings) -> PushgatewayClient:
        """Create PushgatewayClient instance."""
        return PushgatewayClient(settings)

    @pytest.fixture
    def mock_registry(self) -> CollectorRegistry:
        """Create mock registry."""
        return CollectorRegistry()

    def test_push_metrics_disabled_when_not_enabled(
        self, settings: ObservabilitySettings
    ) -> None:
        """Test push_metrics returns False when disabled."""
        settings.pushgateway_url = None
        client = PushgatewayClient(settings)
        mock_registry = CollectorRegistry()

        result = client.push_metrics(mock_registry)
        assert result is False

    def test_push_metrics_disabled_when_no_url(
        self, settings: ObservabilitySettings
    ) -> None:
        """Test push_metrics returns False when no URL configured."""
        settings.pushgateway_url = ""
        client = PushgatewayClient(settings)
        mock_registry = CollectorRegistry()

        result = client.push_metrics(mock_registry)
        assert result is False

    def test_circuit_breaker_blocks_after_failures(
        self, client: PushgatewayClient, mock_registry: CollectorRegistry
    ) -> None:
        """Test circuit breaker blocks requests after failures."""
        # Mock the push_to_gateway to raise exceptions
        with patch("ccproxy.observability.pushgateway.push_to_gateway") as mock_push:
            mock_push.side_effect = ConnectionError("Connection refused")

            # Make multiple requests to trigger circuit breaker
            failures = 0
            for _ in range(7):  # More than failure threshold (5)
                success = client.push_metrics(mock_registry)
                if not success:
                    failures += 1

            # Should have failed all attempts
            assert failures == 7

            # Circuit breaker should be open now
            assert client._circuit_breaker.state == "OPEN"

            # Next request should be blocked by circuit breaker
            success = client.push_metrics(mock_registry)
            assert success is False

    def test_circuit_breaker_records_success(
        self, client: PushgatewayClient, mock_registry: CollectorRegistry
    ) -> None:
        """Test circuit breaker records success."""
        with patch("ccproxy.observability.pushgateway.push_to_gateway") as mock_push:
            mock_push.return_value = None  # Success

            # Make successful request
            success = client.push_metrics(mock_registry)
            assert success is True

            # Circuit breaker should remain closed
            assert client._circuit_breaker.state == "CLOSED"
            assert client._circuit_breaker.failure_count == 0

    def test_push_standard_handles_connection_errors(
        self, client: PushgatewayClient, mock_registry: CollectorRegistry
    ) -> None:
        """Test _push_standard handles connection errors gracefully."""
        with patch("ccproxy.observability.pushgateway.push_to_gateway") as mock_push:
            mock_push.side_effect = ConnectionError("Connection refused")

            success = client._push_standard(mock_registry, "push")
            assert success is False

    def test_push_standard_handles_timeout_errors(
        self, client: PushgatewayClient, mock_registry: CollectorRegistry
    ) -> None:
        """Test _push_standard handles timeout errors gracefully."""
        with patch("ccproxy.observability.pushgateway.push_to_gateway") as mock_push:
            mock_push.side_effect = TimeoutError("Request timeout")

            success = client._push_standard(mock_registry, "push")
            assert success is False

    def test_push_standard_invalid_method(
        self, client: PushgatewayClient, mock_registry: CollectorRegistry
    ) -> None:
        """Test _push_standard handles invalid methods."""
        success = client._push_standard(mock_registry, "invalid")
        assert success is False

    def test_delete_metrics_with_circuit_breaker(
        self, client: PushgatewayClient
    ) -> None:
        """Test delete_metrics uses circuit breaker."""
        with patch(
            "ccproxy.observability.pushgateway.delete_from_gateway"
        ) as mock_delete:
            mock_delete.side_effect = ConnectionError("Connection refused")

            # Multiple failures should trigger circuit breaker
            for _ in range(6):
                success = client.delete_metrics()
                assert success is False

            # Circuit breaker should be open
            assert client._circuit_breaker.state == "OPEN"

    def test_delete_metrics_remote_write_not_supported(
        self, settings: ObservabilitySettings
    ) -> None:
        """Test delete_metrics not supported for remote write URLs."""
        settings.pushgateway_url = "http://localhost:8428/api/v1/write"
        client = PushgatewayClient(settings)

        success = client.delete_metrics()
        assert success is False

    def test_is_enabled_returns_correct_state(self, client: PushgatewayClient) -> None:
        """Test is_enabled returns correct state."""
        assert client.is_enabled() is True

        # Disable and test
        client._enabled = False
        assert client.is_enabled() is False


class TestIntegration:
    """Integration tests for error handling components."""

    @pytest.fixture
    def settings(self) -> ObservabilitySettings:
        """Create test settings with failing pushgateway."""
        return ObservabilitySettings(
            pushgateway_url="http://localhost:9999",  # Non-existent service
            pushgateway_job="test-job",
        )

    async def test_scheduler_with_failing_pushgateway(
        self, settings: ObservabilitySettings
    ) -> None:
        """Test scheduler behavior with failing pushgateway."""
        from ccproxy.config.scheduler import SchedulerSettings
        from ccproxy.scheduler import PushgatewayTask, Scheduler
        from ccproxy.scheduler.registry import register_task

        # Create scheduler settings that enable pushgateway with fast interval
        scheduler_settings = SchedulerSettings(
            pushgateway_enabled=True,
            pushgateway_interval_seconds=1.0,  # Fast interval for testing (min 1.0)
        )

        scheduler = Scheduler(
            max_concurrent_tasks=5,
            graceful_shutdown_timeout=1.0,
        )

        # Register the task type
        register_task("pushgateway", PushgatewayTask)

        # Mock the metrics to simulate failures
        with patch("ccproxy.observability.metrics.get_metrics") as mock_get_metrics:
            mock_metrics = Mock()
            mock_metrics.is_pushgateway_enabled.return_value = True
            mock_metrics.push_to_gateway.return_value = False  # Always fail
            mock_get_metrics.return_value = mock_metrics

            # Add pushgateway task that will fail using task registry
            await scheduler.add_task(
                task_name="test_pushgateway",
                task_type="pushgateway",
                interval_seconds=1.0,
                enabled=True,
            )
            await scheduler.start()

            # Check status while scheduler is running
            status = scheduler.get_scheduler_status()
            assert len(status["task_names"]) > 0  # At least one task was added
            assert status["running"] is True

            # Wait for task to run and potentially fail
            await asyncio.sleep(1.5)

            await scheduler.stop()

            # Verify scheduler is now stopped
            final_status = scheduler.get_scheduler_status()
            assert final_status["running"] is False

    def test_circuit_breaker_and_scheduler_integration(
        self, settings: ObservabilitySettings
    ) -> None:
        """Test circuit breaker integration with scheduler."""
        from ccproxy.scheduler import PushgatewayTask

        client = PushgatewayClient(settings)

        # Create a pushgateway task to simulate scheduler behavior
        task = PushgatewayTask(
            name="test_pushgateway_circuit",
            interval_seconds=1.0,
            enabled=True,
        )

        # Mock registry
        mock_registry = CollectorRegistry()

        # Simulate multiple failures
        with patch("ccproxy.observability.pushgateway.push_to_gateway") as mock_push:
            mock_push.side_effect = ConnectionError("Connection refused")

            # Multiple failures should trigger circuit breaker
            for _ in range(6):
                success = client.push_metrics(mock_registry)
                if not success:
                    # Manually increment task failure counter to simulate scheduler behavior
                    task._consecutive_failures += 1

            # Circuit breaker should be open
            assert client._circuit_breaker.state == "OPEN"

            # Task should have recorded failures
            assert task.consecutive_failures > 0

            # Next push should be blocked by circuit breaker
            success = client.push_metrics(mock_registry)
            assert success is False
