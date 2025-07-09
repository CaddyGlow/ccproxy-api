"""Tests for synchronous metrics storage functionality."""

from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from ccproxy.metrics.models import (
    ErrorMetrics,
    HTTPMetrics,
    ModelMetrics,
    UserAgentCategory,
)
from ccproxy.metrics.sync_storage import SyncMetricsStorage


class TestSyncMetricsStorage:
    """Test cases for synchronous metrics storage functionality."""

    @pytest.fixture
    def temp_storage(self) -> SyncMetricsStorage:
        """Create a temporary storage instance."""
        with TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_metrics.db"
            storage = SyncMetricsStorage(f"sqlite:///{db_path}")
            storage.initialize()
            yield storage
            storage.close()

    @pytest.fixture
    def sample_http_metrics(self) -> HTTPMetrics:
        """Create sample HTTP metrics."""
        return HTTPMetrics(
            method="POST",
            endpoint="/v1/chat/completions",
            status_code=200,
            api_type="anthropic",
            user_agent_category=UserAgentCategory.PYTHON_SDK,
            duration_seconds=1.5,
            request_size_bytes=1024,
            response_size_bytes=2048,
        )

    @pytest.fixture
    def sample_model_metrics(self) -> ModelMetrics:
        """Create sample model metrics."""
        return ModelMetrics(
            model="claude-3-5-sonnet-20241022",
            api_type="anthropic",
            endpoint="/v1/chat/completions",
            streaming=False,
            input_tokens=100,
            output_tokens=50,
            estimated_cost=0.45,
        )

    def test_initialize_storage(self, temp_storage: SyncMetricsStorage) -> None:
        """Test storage initialization."""
        # If we got here without error, initialization worked
        assert temp_storage is not None

        # Test that we can get stats (which queries all tables)
        stats = temp_storage.get_storage_stats()
        assert stats["metrics_snapshots_count"] == 0
        assert stats["request_logs_count"] == 0
        assert stats["daily_aggregates_count"] == 0

    def test_store_and_retrieve_metrics_snapshot(
        self, temp_storage: SyncMetricsStorage
    ) -> None:
        """Test storing and retrieving metrics snapshots."""
        timestamp = datetime.utcnow()
        labels = {"endpoint": "/v1/chat/completions", "method": "POST"}

        temp_storage.store_metrics_snapshot(
            metric_name="http_requests_total",
            metric_type="counter",
            labels=labels,
            value=42.0,
            timestamp=timestamp,
        )

        # Verify the snapshot was stored
        snapshots = temp_storage.get_metrics_snapshots(
            metric_name="http_requests_total"
        )
        assert len(snapshots) == 1

        snapshot = snapshots[0]
        assert snapshot.metric_name == "http_requests_total"
        assert snapshot.metric_type == "counter"
        assert snapshot.value == 42.0

    def test_store_and_retrieve_request_log(
        self,
        temp_storage: SyncMetricsStorage,
        sample_http_metrics: HTTPMetrics,
        sample_model_metrics: ModelMetrics,
    ) -> None:
        """Test storing and retrieving request logs."""
        temp_storage.store_request_log(
            http_metrics=sample_http_metrics,
            model_metrics=sample_model_metrics,
        )

        # Verify the request log was stored
        logs = temp_storage.get_request_logs()
        assert len(logs) == 1

        log = logs[0]
        assert log.method == sample_http_metrics.method
        assert log.endpoint == sample_http_metrics.endpoint
        assert log.status_code == sample_http_metrics.status_code
        assert log.api_type == sample_http_metrics.api_type
        assert log.model == sample_model_metrics.model
        assert log.input_tokens == sample_model_metrics.input_tokens
        assert log.output_tokens == sample_model_metrics.output_tokens
        assert log.cost_dollars == sample_model_metrics.estimated_cost

    def test_calculate_daily_aggregates(
        self,
        temp_storage: SyncMetricsStorage,
        sample_http_metrics: HTTPMetrics,
        sample_model_metrics: ModelMetrics,
    ) -> None:
        """Test calculating daily aggregates."""
        # Store some request logs
        temp_storage.store_request_log(sample_http_metrics, sample_model_metrics)

        # Create error log with same model to ensure single aggregate
        error_metrics = HTTPMetrics(
            method="POST",
            endpoint="/v1/chat/completions",
            status_code=429,
            api_type="anthropic",
            user_agent_category=UserAgentCategory.PYTHON_SDK,
            duration_seconds=0.1,
        )
        # Use the same model for error request
        error_model_metrics = ModelMetrics(
            model="claude-3-5-sonnet-20241022",
            api_type="anthropic",
            endpoint="/v1/chat/completions",
            streaming=False,
            input_tokens=0,
            output_tokens=0,
            estimated_cost=0.0,
        )
        temp_storage.store_request_log(error_metrics, error_model_metrics)

        # Calculate aggregates for today
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        temp_storage.calculate_daily_aggregates(today)

        # Verify aggregates were calculated
        aggregates = temp_storage.get_daily_aggregates(start_date=today, end_date=today)
        assert len(aggregates) == 1

        agg = aggregates[0]
        assert agg.endpoint == "/v1/chat/completions"
        assert agg.api_type == "anthropic"
        assert agg.total_requests == 2
        assert agg.total_errors == 1
        assert agg.total_input_tokens == sample_model_metrics.input_tokens
        assert agg.total_output_tokens == sample_model_metrics.output_tokens
        assert agg.total_cost_dollars == sample_model_metrics.estimated_cost

    def test_cleanup_old_data(self, temp_storage: SyncMetricsStorage) -> None:
        """Test cleaning up old data."""
        old_time = datetime.utcnow() - timedelta(days=35)
        recent_time = datetime.utcnow() - timedelta(days=10)

        # Store old and recent data
        temp_storage.store_metrics_snapshot(
            "old_metric", "counter", {"test": "old"}, 1.0, old_time
        )
        temp_storage.store_metrics_snapshot(
            "recent_metric", "counter", {"test": "recent"}, 2.0, recent_time
        )

        # Clean up data older than 30 days
        temp_storage.cleanup_old_data(retention_days=30)

        # Verify old data is gone, recent data remains
        snapshots = temp_storage.get_metrics_snapshots()
        assert len(snapshots) == 1
        assert snapshots[0].metric_name == "recent_metric"

    def test_get_storage_stats(self, temp_storage: SyncMetricsStorage) -> None:
        """Test getting storage statistics."""
        # Add some sample data
        temp_storage.store_metrics_snapshot(
            "test_metric", "counter", {"test": "value"}, 1.0
        )

        stats = temp_storage.get_storage_stats()

        assert isinstance(stats, dict)
        assert "metrics_snapshots_count" in stats
        assert "request_logs_count" in stats
        assert "daily_aggregates_count" in stats
        assert "database_url" in stats
        assert stats["metrics_snapshots_count"] == 1
        assert stats["request_logs_count"] == 0
        assert stats["daily_aggregates_count"] == 0
