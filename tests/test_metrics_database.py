"""Tests for metrics database and storage functionality."""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import text

from ccproxy.metrics.database import (
    Base,
    DailyAggregate,
    MetricsSnapshot,
    RequestLog,
    deserialize_labels,
    serialize_labels,
)
from ccproxy.metrics.models import (
    ErrorMetrics,
    HTTPMetrics,
    ModelMetrics,
    UserAgentCategory,
)
from ccproxy.metrics.storage import MetricsStorage


class TestMetricsDatabase:
    """Test cases for metrics database models and utilities."""

    def test_serialize_labels(self) -> None:
        """Test label serialization."""
        labels = {"endpoint": "/v1/chat/completions", "method": "POST", "status": "200"}
        serialized = serialize_labels(labels)

        # Should be valid JSON
        assert isinstance(serialized, str)
        parsed = json.loads(serialized)
        assert parsed == labels

    def test_deserialize_labels(self) -> None:
        """Test label deserialization."""
        labels = {"endpoint": "/v1/chat/completions", "method": "POST", "status": "200"}
        serialized = serialize_labels(labels)
        deserialized = deserialize_labels(serialized)

        assert deserialized == labels

    def test_serialize_labels_sorting(self) -> None:
        """Test that labels are sorted consistently."""
        labels1 = {"b": "2", "a": "1", "c": "3"}
        labels2 = {"c": "3", "a": "1", "b": "2"}

        serialized1 = serialize_labels(labels1)
        serialized2 = serialize_labels(labels2)

        # Should be identical when sorted
        assert serialized1 == serialized2


class TestMetricsStorage:
    """Test cases for metrics storage functionality."""

    @pytest.fixture
    async def storage(self) -> MetricsStorage:
        """Create a test storage instance."""
        # Use in-memory SQLite for testing
        storage = MetricsStorage("sqlite+aiosqlite:///:memory:")
        await storage.initialize()
        return storage

    @pytest.fixture
    async def sample_http_metrics(self) -> HTTPMetrics:
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
    async def sample_model_metrics(self) -> ModelMetrics:
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

    @pytest.fixture
    async def sample_error_metrics(self) -> ErrorMetrics:
        """Create sample error metrics."""
        return ErrorMetrics(
            error_type="rate_limit",
            endpoint="/v1/chat/completions",
            status_code=429,
            api_type="anthropic",
            user_agent_category=UserAgentCategory.PYTHON_SDK,
        )

    @pytest.mark.asyncio
    async def test_initialize_storage(self, storage: MetricsStorage) -> None:
        """Test storage initialization."""
        # Check that tables exist by querying them
        async with storage.get_session() as session:
            # Test each table
            await session.execute(text("SELECT COUNT(*) FROM metrics_snapshots"))
            await session.execute(text("SELECT COUNT(*) FROM request_logs"))
            await session.execute(text("SELECT COUNT(*) FROM daily_aggregates"))

    @pytest.mark.asyncio
    async def test_store_metrics_snapshot(self, storage: MetricsStorage) -> None:
        """Test storing metrics snapshots."""
        timestamp = datetime.utcnow()
        labels = {"endpoint": "/v1/chat/completions", "method": "POST"}

        await storage.store_metrics_snapshot(
            metric_name="http_requests_total",
            metric_type="counter",
            labels=labels,
            value=42.0,
            timestamp=timestamp,
        )

        # Verify the snapshot was stored
        snapshots = await storage.get_metrics_snapshots(
            metric_name="http_requests_total"
        )
        assert len(snapshots) == 1

        snapshot = snapshots[0]
        assert snapshot.metric_name == "http_requests_total"
        assert snapshot.metric_type == "counter"
        assert snapshot.value == 42.0
        assert deserialize_labels(snapshot.labels) == labels

    @pytest.mark.asyncio
    async def test_store_request_log(
        self,
        storage: MetricsStorage,
        sample_http_metrics: HTTPMetrics,
        sample_model_metrics: ModelMetrics,
        sample_error_metrics: ErrorMetrics,
    ) -> None:
        """Test storing request logs."""
        await storage.store_request_log(
            http_metrics=sample_http_metrics,
            model_metrics=sample_model_metrics,
        )

        # Verify the request log was stored
        logs = await storage.get_request_logs()
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
        assert log.duration_ms == sample_http_metrics.duration_seconds * 1000

    @pytest.mark.asyncio
    async def test_store_request_log_with_error(
        self,
        storage: MetricsStorage,
        sample_http_metrics: HTTPMetrics,
        sample_error_metrics: ErrorMetrics,
    ) -> None:
        """Test storing request logs with error metrics."""
        # Update HTTP metrics to match error status
        sample_http_metrics.status_code = 429

        await storage.store_request_log(
            http_metrics=sample_http_metrics,
            error_metrics=sample_error_metrics,
        )

        # Verify the error was captured
        logs = await storage.get_request_logs(status_code=429)
        assert len(logs) == 1

        log = logs[0]
        assert log.error_type == sample_error_metrics.error_type
        assert log.status_code == 429

    @pytest.mark.asyncio
    async def test_get_metrics_snapshots_filtering(
        self, storage: MetricsStorage
    ) -> None:
        """Test filtering metrics snapshots."""
        base_time = datetime.utcnow()

        # Store multiple snapshots
        await storage.store_metrics_snapshot(
            "http_requests_total", "counter", {"endpoint": "/v1/chat"}, 10.0, base_time
        )
        await storage.store_metrics_snapshot(
            "http_requests_total",
            "counter",
            {"endpoint": "/v1/messages"},
            20.0,
            base_time + timedelta(minutes=1),
        )
        await storage.store_metrics_snapshot(
            "http_duration_seconds",
            "histogram",
            {"endpoint": "/v1/chat"},
            1.5,
            base_time + timedelta(minutes=2),
        )

        # Test filtering by metric name
        requests_snapshots = await storage.get_metrics_snapshots(
            metric_name="http_requests_total"
        )
        assert len(requests_snapshots) == 2

        # Test filtering by time range
        time_filtered = await storage.get_metrics_snapshots(
            start_time=base_time + timedelta(minutes=1),
            end_time=base_time + timedelta(minutes=2),
        )
        assert len(time_filtered) == 2

    @pytest.mark.asyncio
    async def test_get_request_logs_filtering(
        self,
        storage: MetricsStorage,
        sample_http_metrics: HTTPMetrics,
        sample_model_metrics: ModelMetrics,
    ) -> None:
        """Test filtering request logs."""
        # Store logs with different attributes
        await storage.store_request_log(sample_http_metrics, sample_model_metrics)

        # Create another log with different endpoint
        other_metrics = HTTPMetrics(
            method="GET",
            endpoint="/v1/models",
            status_code=200,
            api_type="openai",
            user_agent_category=UserAgentCategory.OPENAI_SDK,
            duration_seconds=0.5,
        )
        await storage.store_request_log(other_metrics)

        # Test filtering by endpoint
        chat_logs = await storage.get_request_logs(endpoint="/v1/chat/completions")
        assert len(chat_logs) == 1
        assert chat_logs[0].endpoint == "/v1/chat/completions"

        # Test filtering by API type
        openai_logs = await storage.get_request_logs(api_type="openai")
        assert len(openai_logs) == 1
        assert openai_logs[0].api_type == "openai"

    @pytest.mark.asyncio
    async def test_calculate_daily_aggregates(
        self,
        storage: MetricsStorage,
        sample_http_metrics: HTTPMetrics,
        sample_model_metrics: ModelMetrics,
    ) -> None:
        """Test calculating daily aggregates."""
        # Store some request logs
        await storage.store_request_log(sample_http_metrics, sample_model_metrics)

        # Create error log
        error_metrics = HTTPMetrics(
            method="POST",
            endpoint="/v1/chat/completions",
            status_code=429,
            api_type="anthropic",
            user_agent_category=UserAgentCategory.PYTHON_SDK,
            duration_seconds=0.1,
        )
        await storage.store_request_log(error_metrics)

        # Calculate aggregates for today
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        await storage.calculate_daily_aggregates(today)

        # Verify aggregates were calculated
        aggregates = await storage.get_daily_aggregates(
            start_date=today, end_date=today
        )
        assert len(aggregates) == 1

        agg = aggregates[0]
        assert agg.endpoint == "/v1/chat/completions"
        assert agg.api_type == "anthropic"
        assert agg.total_requests == 2
        assert agg.total_errors == 1
        assert agg.total_input_tokens == sample_model_metrics.input_tokens
        assert agg.total_output_tokens == sample_model_metrics.output_tokens
        assert agg.total_cost_dollars == sample_model_metrics.estimated_cost

    @pytest.mark.asyncio
    async def test_get_daily_aggregates_filtering(
        self, storage: MetricsStorage
    ) -> None:
        """Test filtering daily aggregates."""
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        yesterday = today - timedelta(days=1)

        # Create sample aggregates
        async with storage.get_session() as session:
            # Today's data
            agg1 = DailyAggregate(
                date=today,
                endpoint="/v1/chat/completions",
                api_type="anthropic",
                model="claude-3-5-sonnet",
                total_requests=100,
                total_errors=5,
                avg_duration_ms=500.0,
                total_input_tokens=1000,
                total_output_tokens=500,
                total_cost_dollars=5.0,
            )
            session.add(agg1)

            # Yesterday's data
            agg2 = DailyAggregate(
                date=yesterday,
                endpoint="/v1/chat/completions",
                api_type="openai",
                model="gpt-4",
                total_requests=50,
                total_errors=2,
                avg_duration_ms=800.0,
                total_input_tokens=500,
                total_output_tokens=300,
                total_cost_dollars=10.0,
            )
            session.add(agg2)

            await session.commit()

        # Test filtering by date range
        today_aggs = await storage.get_daily_aggregates(
            start_date=today, end_date=today
        )
        assert len(today_aggs) == 1
        assert today_aggs[0].api_type == "anthropic"

        # Test filtering by API type
        openai_aggs = await storage.get_daily_aggregates(api_type="openai")
        assert len(openai_aggs) == 1
        assert openai_aggs[0].model == "gpt-4"

    @pytest.mark.asyncio
    async def test_cleanup_old_data(self, storage: MetricsStorage) -> None:
        """Test cleaning up old data."""
        old_time = datetime.utcnow() - timedelta(days=35)
        recent_time = datetime.utcnow() - timedelta(days=10)

        # Store old and recent data
        await storage.store_metrics_snapshot(
            "old_metric", "counter", {"test": "old"}, 1.0, old_time
        )
        await storage.store_metrics_snapshot(
            "recent_metric", "counter", {"test": "recent"}, 2.0, recent_time
        )

        # Clean up data older than 30 days
        await storage.cleanup_old_data(retention_days=30)

        # Verify old data is gone, recent data remains
        snapshots = await storage.get_metrics_snapshots()
        assert len(snapshots) == 1
        assert snapshots[0].metric_name == "recent_metric"

    @pytest.mark.asyncio
    async def test_get_storage_stats(self, storage: MetricsStorage) -> None:
        """Test getting storage statistics."""
        # Add some sample data
        await storage.store_metrics_snapshot(
            "test_metric", "counter", {"test": "value"}, 1.0
        )

        stats = await storage.get_storage_stats()

        assert isinstance(stats, dict)
        assert "metrics_snapshots_count" in stats
        assert "request_logs_count" in stats
        assert "daily_aggregates_count" in stats
        assert "database_url" in stats
        assert stats["metrics_snapshots_count"] == 1
        assert stats["request_logs_count"] == 0
        assert stats["daily_aggregates_count"] == 0

    @pytest.mark.asyncio
    async def test_global_storage_instance(self) -> None:
        """Test global storage instance management."""
        from ccproxy.metrics.storage import close_metrics_storage, get_metrics_storage

        # Get storage instance
        storage1 = await get_metrics_storage("sqlite+aiosqlite:///:memory:")
        storage2 = await get_metrics_storage("sqlite+aiosqlite:///:memory:")

        # Should be the same instance
        assert storage1 is storage2

        # Close storage
        await close_metrics_storage()

        # Getting storage again should create a new instance
        storage3 = await get_metrics_storage("sqlite+aiosqlite:///:memory:")
        assert storage3 is not storage1
