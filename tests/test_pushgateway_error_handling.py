"""Tests for pushgateway error handling improvements."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Generator
from typing import Any
from unittest.mock import Mock, patch

import pytest
from prometheus_client import CollectorRegistry

from ccproxy.config.observability import ObservabilitySettings
from ccproxy.observability.pushgateway import CircuitBreaker, PushgatewayClient
from ccproxy.observability.scheduler import ObservabilityScheduler


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
            pushgateway_enabled=True,
            pushgateway_url="http://localhost:9091",
            pushgateway_job="test-job",
            pushgateway_batch_interval=1.0,
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
        settings.pushgateway_enabled = False
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


class TestObservabilityScheduler:
    """Test ObservabilityScheduler with exponential backoff."""

    @pytest.fixture
    def settings(self) -> ObservabilitySettings:
        """Create test settings."""
        return ObservabilitySettings(
            pushgateway_enabled=True,
            pushgateway_url="http://localhost:9091",
            pushgateway_job="test-job",
            pushgateway_batch_interval=1.0,
        )

    @pytest.fixture
    def scheduler(self, settings: ObservabilitySettings) -> ObservabilityScheduler:
        """Create scheduler instance."""
        return ObservabilityScheduler(settings)

    def test_calculate_backoff_delay_no_failures(
        self, scheduler: ObservabilityScheduler
    ) -> None:
        """Test backoff delay with no failures."""
        scheduler._consecutive_failures = 0
        delay = scheduler._calculate_backoff_delay()
        assert delay == scheduler._pushgateway_interval

    def test_calculate_backoff_delay_exponential_growth(
        self, scheduler: ObservabilityScheduler
    ) -> None:
        """Test backoff delay grows exponentially."""
        delays = []
        for i in range(5):
            scheduler._consecutive_failures = i
            delay = scheduler._calculate_backoff_delay()
            delays.append(delay)

        # Should increase exponentially (with some jitter)
        assert delays[1] > delays[0]
        assert delays[2] > delays[1]
        assert delays[3] > delays[2]

    def test_calculate_backoff_delay_max_cap(
        self, scheduler: ObservabilityScheduler
    ) -> None:
        """Test backoff delay is capped at maximum."""
        scheduler._consecutive_failures = 20  # Very high failure count
        delay = scheduler._calculate_backoff_delay()

        # Should be capped at max_backoff (allowing for jitter)
        assert delay <= scheduler._max_backoff * 1.25  # 25% jitter tolerance

    def test_calculate_backoff_delay_minimum(
        self, scheduler: ObservabilityScheduler
    ) -> None:
        """Test backoff delay has minimum value."""
        scheduler._consecutive_failures = 1
        delay = scheduler._calculate_backoff_delay()
        assert delay >= 1.0

    def test_calculate_backoff_delay_has_jitter(
        self, scheduler: ObservabilityScheduler
    ) -> None:
        """Test backoff delay includes jitter."""
        scheduler._consecutive_failures = 3

        # Calculate multiple delays and verify they're different (jitter)
        delays = []
        for _ in range(10):
            delay = scheduler._calculate_backoff_delay()
            delays.append(delay)

        # Should have variation due to jitter
        assert len(set(delays)) > 1

    async def test_scheduler_start_stop(
        self, scheduler: ObservabilityScheduler
    ) -> None:
        """Test scheduler start and stop lifecycle."""
        assert scheduler._running is False

        await scheduler.start()
        assert scheduler._running is True

        await scheduler.stop()  # type: ignore[unreachable]
        assert scheduler._running is False

    async def test_scheduler_tracks_consecutive_failures(
        self, scheduler: ObservabilityScheduler
    ) -> None:
        """Test scheduler tracks consecutive failures."""
        # Mock metrics instance that always fails
        mock_metrics = Mock()
        mock_metrics.is_pushgateway_enabled.return_value = True
        mock_metrics.push_to_gateway.return_value = False
        scheduler._metrics_instance = mock_metrics

        # Start scheduler briefly
        await scheduler.start()

        # Wait a bit for task to run
        await asyncio.sleep(0.1)

        await scheduler.stop()

        # Should have recorded failures
        assert scheduler._consecutive_failures > 0

    def test_set_pushgateway_interval(self, scheduler: ObservabilityScheduler) -> None:
        """Test setting pushgateway interval."""
        scheduler.set_pushgateway_interval(5.0)
        assert scheduler._pushgateway_interval == 5.0

        # Test minimum value
        scheduler.set_pushgateway_interval(0.5)
        assert scheduler._pushgateway_interval == 1.0  # Should be clamped to minimum


class TestIntegration:
    """Integration tests for error handling components."""

    @pytest.fixture
    def settings(self) -> ObservabilitySettings:
        """Create test settings with failing pushgateway."""
        return ObservabilitySettings(
            pushgateway_enabled=True,
            pushgateway_url="http://localhost:9999",  # Non-existent service
            pushgateway_job="test-job",
            pushgateway_batch_interval=0.1,  # Fast interval for testing
        )

    async def test_scheduler_with_failing_pushgateway(
        self, settings: ObservabilitySettings
    ) -> None:
        """Test scheduler behavior with failing pushgateway."""
        scheduler = ObservabilityScheduler(settings)

        # Mock metrics instance
        mock_metrics = Mock()
        mock_metrics.is_pushgateway_enabled.return_value = True
        mock_metrics.push_to_gateway.return_value = False
        scheduler._metrics_instance = mock_metrics

        # Start scheduler
        await scheduler.start()

        # Wait for a few failures
        await asyncio.sleep(0.3)

        # Should have recorded multiple failures
        assert scheduler._consecutive_failures > 0

        await scheduler.stop()

    def test_circuit_breaker_and_scheduler_integration(
        self, settings: ObservabilitySettings
    ) -> None:
        """Test circuit breaker integration with scheduler."""
        client = PushgatewayClient(settings)
        scheduler = ObservabilityScheduler(settings)

        # Mock registry
        mock_registry = CollectorRegistry()

        # Simulate multiple failures
        with patch("ccproxy.observability.pushgateway.push_to_gateway") as mock_push:
            mock_push.side_effect = ConnectionError("Connection refused")

            # Multiple failures should trigger circuit breaker
            for _ in range(6):
                success = client.push_metrics(mock_registry)
                if not success:
                    scheduler._consecutive_failures += 1

            # Circuit breaker should be open
            assert client._circuit_breaker.state == "OPEN"

            # Scheduler should have recorded failures
            assert scheduler._consecutive_failures > 0

            # Next push should be blocked by circuit breaker
            success = client.push_metrics(mock_registry)
            assert success is False
