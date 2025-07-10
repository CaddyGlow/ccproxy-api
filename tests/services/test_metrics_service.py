"""Tests for MetricsService."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from ccproxy.services.metrics_service import MetricsService


class TestMetricsService:
    """Test cases for MetricsService."""

    @pytest.fixture
    def metrics_service(self):
        """Create a MetricsService instance."""
        return MetricsService(buffer_size=100, export_interval=30)

    @pytest.fixture
    def mock_exporter(self):
        """Create a mock metrics exporter."""
        mock = AsyncMock()
        mock.export_metrics = AsyncMock(return_value=True)
        mock.health_check = AsyncMock(return_value=True)
        return mock

    def test_metrics_service_initialization(self, metrics_service):
        """Test that MetricsService initializes correctly."""
        assert metrics_service.buffer_size == 100
        assert metrics_service.export_interval == 30
        assert len(metrics_service._metrics_buffer) == 0
        assert metrics_service._request_metrics["total_requests"] == 0

    def test_record_request_start(self, metrics_service):
        """Test recording request start."""
        metrics_service.record_request_start(
            request_id="test-123",
            method="POST",
            path="/v1/messages",
            model="claude-3-sonnet",
            user_id="user-456",
        )

        assert len(metrics_service._metrics_buffer) == 1
        assert metrics_service._request_metrics["total_requests"] == 1
        assert metrics_service._request_metrics["requests_by_path"]["/v1/messages"] == 1
        assert (
            metrics_service._request_metrics["requests_by_model"]["claude-3-sonnet"]
            == 1
        )

    def test_record_request_end_success(self, metrics_service):
        """Test recording successful request end."""
        metrics_service.record_request_end(
            request_id="test-123",
            status_code=200,
            response_time=1.5,
            tokens_used=100,
            service_type="sdk",
        )

        assert len(metrics_service._metrics_buffer) == 1
        assert metrics_service._request_metrics["successful_requests"] == 1
        assert metrics_service._request_metrics["failed_requests"] == 0

    def test_record_request_end_error(self, metrics_service):
        """Test recording failed request end."""
        metrics_service.record_request_end(
            request_id="test-123",
            status_code=500,
            response_time=0.1,
            service_type="proxy",
        )

        assert metrics_service._request_metrics["successful_requests"] == 0
        assert metrics_service._request_metrics["failed_requests"] == 1

    def test_record_error(self, metrics_service):
        """Test recording error events."""
        metrics_service.record_error(
            error_type="timeout",
            error_message="Request timeout",
            status_code=504,
            request_id="test-123",
        )

        assert len(metrics_service._metrics_buffer) == 1
        assert metrics_service._error_metrics["total_errors"] == 1
        assert metrics_service._error_metrics["errors_by_type"]["timeout"] == 1
        assert metrics_service._error_metrics["errors_by_status_code"][504] == 1

    def test_record_service_event(self, metrics_service):
        """Test recording service events."""
        metrics_service.record_service_event(
            event_type="startup",
            service_type="sdk",
            metadata={"streaming": True, "tools": False},
        )

        assert len(metrics_service._metrics_buffer) == 1
        assert metrics_service._service_metrics["sdk_requests"] == 1
        assert metrics_service._service_metrics["streaming_requests"] == 1
        assert metrics_service._service_metrics["tool_requests"] == 0

    def test_get_metrics_summary(self, metrics_service):
        """Test getting metrics summary."""
        # Add some test data
        metrics_service.record_request_start("test-1", "POST", "/v1/messages")
        metrics_service.record_request_end("test-1", 200, 1.0)

        summary = metrics_service.get_metrics_summary()

        assert "request_metrics" in summary
        assert "service_metrics" in summary
        assert "error_metrics" in summary
        assert "buffer_size" in summary
        assert "last_export" in summary
        assert summary["buffer_size"] == 2  # start + end events

    def test_get_metrics_by_timeframe(self, metrics_service):
        """Test filtering metrics by timeframe."""
        # Record some metrics
        metrics_service.record_request_start("test-1", "POST", "/v1/messages")

        # Get metrics from the last hour
        now = datetime.now()
        start_time = now - timedelta(hours=1)
        end_time = now + timedelta(minutes=1)

        filtered_metrics = metrics_service.get_metrics_by_timeframe(
            start_time, end_time
        )

        assert len(filtered_metrics) == 1
        assert filtered_metrics[0]["request_id"] == "test-1"

    def test_calculate_request_rate(self, metrics_service):
        """Test calculating request rate."""
        # Add multiple request starts
        for i in range(5):
            metrics_service.record_request_start(f"test-{i}", "POST", "/v1/messages")

        # Calculate rate for last 5 minutes
        rate = metrics_service.calculate_request_rate(window_minutes=5)

        assert rate == 1.0  # 5 requests / 5 minutes = 1 req/min

    def test_calculate_error_rate(self, metrics_service):
        """Test calculating error rate."""
        # Add some successful and failed requests
        for i in range(8):
            metrics_service.record_request_end(f"test-{i}", 200, 1.0)
        for i in range(2):
            metrics_service.record_request_end(f"error-{i}", 500, 0.1)

        error_rate = metrics_service.calculate_error_rate(window_minutes=5)

        assert error_rate == 20.0  # 2 errors out of 10 total = 20%

    @pytest.mark.asyncio
    async def test_export_metrics_success(self, mock_exporter):
        """Test successful metrics export."""
        metrics_service = MetricsService(exporters=[mock_exporter])

        result = await metrics_service.export_metrics()

        assert result is True
        mock_exporter.export_metrics.assert_called_once()

    @pytest.mark.asyncio
    async def test_export_metrics_no_exporters(self, metrics_service):
        """Test export with no exporters configured."""
        result = await metrics_service.export_metrics()
        assert result is True

    @pytest.mark.asyncio
    async def test_export_metrics_failure(self):
        """Test export failure handling."""
        mock_exporter = AsyncMock()
        mock_exporter.export_metrics = AsyncMock(side_effect=Exception("Export failed"))

        metrics_service = MetricsService(exporters=[mock_exporter])

        result = await metrics_service.export_metrics()

        assert result is False

    def test_should_export(self, metrics_service):
        """Test export timing logic."""
        # Should not export immediately after initialization
        assert metrics_service.should_export() is False

        # Simulate time passing
        metrics_service._last_export_time = datetime.now() - timedelta(seconds=60)

        # Should export now
        assert metrics_service.should_export() is True

    def test_clear_buffer(self, metrics_service):
        """Test clearing metrics buffer."""
        # Add some metrics
        metrics_service.record_request_start("test-1", "POST", "/v1/messages")
        assert len(metrics_service._metrics_buffer) == 1

        # Clear buffer
        metrics_service.clear_buffer()
        assert len(metrics_service._metrics_buffer) == 0

    def test_buffer_size_limit(self):
        """Test that buffer respects size limit."""
        metrics_service = MetricsService(buffer_size=5)

        # Add more metrics than buffer size
        for i in range(10):
            metrics_service.record_request_start(f"test-{i}", "POST", "/v1/messages")

        # Buffer should not exceed limit
        assert len(metrics_service._metrics_buffer) == 5
