"""Tests for DuckDB storage backend."""

import asyncio
import tempfile
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest

from ccproxy.observability.storage.duckdb_simple import (
    SimpleDuckDBStorage as DuckDBStorage,
)


@pytest.mark.unit
class TestDuckDBStorage:
    """Test DuckDB storage backend functionality."""

    @pytest.fixture
    async def storage(self) -> AsyncGenerator[DuckDBStorage, None]:
        """Create temporary DuckDB storage for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_metrics.duckdb"
            storage = DuckDBStorage(database_path=db_path, pool_size=2)
            await storage.initialize()
            yield storage
            await storage.close()

    @pytest.fixture
    def sample_request_metric(self) -> dict[str, Any]:
        """Sample request metric for testing."""
        return {
            "timestamp": time.time(),
            "request_id": "req_123",
            "method": "POST",
            "endpoint": "messages",
            "model": "claude-3-sonnet",
            "status": "success",
            "response_time": 1.5,
            "tokens_input": 150,
            "tokens_output": 75,
            "cost_usd": 0.0023,
            "metadata": {"user_id": "user_456"},
        }

    @pytest.fixture
    def sample_operation_metric(self) -> dict[str, Any]:
        """Sample operation metric for testing."""
        return {
            "timestamp": time.time(),
            "request_id": "req_123",
            "operation_id": "op_789",
            "operation_name": "api_call",
            "duration_ms": 1200.0,
            "status": "success",
            "metadata": {"model": "claude-3-sonnet"},
        }

    async def test_storage_initialization(self) -> None:
        """Test storage initialization and schema creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_init.duckdb"
            storage = DuckDBStorage(database_path=db_path)

            assert not storage.is_enabled()

            await storage.initialize()

            # Check if database file was created
            assert db_path.exists()
            assert storage.is_enabled()

            await storage.close()

    async def test_store_single_request_metric(
        self, storage: DuckDBStorage, sample_request_metric: dict[str, Any]
    ) -> None:
        """Test storing a single request metric."""
        result = await storage.store(sample_request_metric)
        assert result is True

        # Verify the metric was stored
        results = await storage.query("SELECT * FROM requests")
        assert len(results) == 1

        stored = results[0]
        assert stored["request_id"] == "req_123"
        assert stored["method"] == "POST"
        assert stored["model"] == "claude-3-sonnet"
        assert stored["status"] == "success"
        assert stored["tokens_input"] == 150

    async def test_store_batch_metrics(
        self,
        storage: DuckDBStorage,
        sample_request_metric: dict[str, Any],
        sample_operation_metric: dict[str, Any],
    ) -> None:
        """Test storing a batch of metrics."""
        metrics = [sample_request_metric, sample_operation_metric]
        result = await storage.store_batch(metrics)
        assert result is True

        # Verify requests table
        requests = await storage.query("SELECT * FROM requests")
        assert len(requests) == 1
        assert requests[0]["request_id"] == "req_123"

        # Verify operations table
        operations = await storage.query("SELECT * FROM operations")
        assert len(operations) == 1
        assert operations[0]["operation_id"] == "op_789"

    async def test_query_with_limit(
        self, storage: DuckDBStorage, sample_request_metric: dict[str, Any]
    ) -> None:
        """Test querying with limit parameter."""
        # Store multiple metrics
        for i in range(5):
            metric = sample_request_metric.copy()
            metric["request_id"] = f"req_{i}"
            await storage.store(metric)

        # Query with limit
        results = await storage.query(
            "SELECT * FROM requests ORDER BY request_id", limit=3
        )
        assert len(results) == 3

    async def test_analytics_generation(
        self, storage: DuckDBStorage, sample_request_metric: dict[str, Any]
    ) -> None:
        """Test analytics data generation."""
        # Store some test data
        current_time = time.time()
        for i in range(3):
            metric = sample_request_metric.copy()
            metric["request_id"] = f"req_{i}"
            metric["timestamp"] = current_time - (i * 3600)  # 1 hour apart
            metric["cost_usd"] = 0.001 * (i + 1)
            await storage.store(metric)

        # Get analytics
        analytics = await storage.get_analytics()

        assert "summary" in analytics
        assert "hourly_data" in analytics
        assert "model_stats" in analytics

        summary = analytics["summary"]
        assert summary["total_requests"] == 3
        assert summary["successful_requests"] == 3
        assert summary["failed_requests"] == 0
        assert summary["total_cost_usd"] == 0.006  # 0.001 + 0.002 + 0.003

    async def test_analytics_with_filters(
        self, storage: DuckDBStorage, sample_request_metric: dict[str, Any]
    ) -> None:
        """Test analytics with time and model filters."""
        current_time = time.time()

        # Store metrics for different models
        for model in ["claude-3-sonnet", "claude-3-haiku"]:
            for i in range(2):
                metric = sample_request_metric.copy()
                metric["request_id"] = f"req_{model}_{i}"
                metric["model"] = model
                metric["timestamp"] = current_time
                await storage.store(metric)

        # Get analytics filtered by model
        analytics = await storage.get_analytics(model="claude-3-sonnet")

        summary = analytics["summary"]
        assert summary["total_requests"] == 2

        model_stats = analytics["model_stats"]
        assert len(model_stats) == 1
        assert model_stats[0]["model"] == "claude-3-sonnet"

    async def test_health_check(self, storage: DuckDBStorage) -> None:
        """Test storage health check."""
        health = await storage.health_check()

        assert health["status"] == "healthy"
        assert health["enabled"] is True
        assert "database_path" in health
        assert "request_count" in health
        assert "pool_size" in health

    async def test_error_metric_storage(self, storage: DuckDBStorage) -> None:
        """Test storing error metrics."""
        error_metric = {
            "timestamp": time.time(),
            "request_id": "req_error",
            "method": "POST",
            "endpoint": "messages",
            "model": "claude-3-sonnet",
            "status": "error",
            "response_time": 0.5,
            "error_type": "timeout_error",
            "error_message": "Request timed out",
            "metadata": {},
        }

        result = await storage.store(error_metric)
        assert result is True

        # Query error metrics
        results = await storage.query("SELECT * FROM requests WHERE status = 'error'")
        assert len(results) == 1

        stored = results[0]
        assert stored["status"] == "error"
        assert stored["error_type"] == "timeout_error"
        assert stored["error_message"] == "Request timed out"

    async def test_concurrent_storage(
        self, storage: DuckDBStorage, sample_request_metric: dict[str, Any]
    ) -> None:
        """Test concurrent storage operations."""

        async def store_metric(i: int) -> bool:
            metric = sample_request_metric.copy()
            metric["request_id"] = f"req_concurrent_{i}"
            return await storage.store(metric)

        # Store metrics concurrently
        tasks = [store_metric(i) for i in range(10)]
        results = await asyncio.gather(*tasks)

        # All should succeed
        assert all(results)

        # Verify all metrics were stored
        stored = await storage.query("SELECT COUNT(*) as count FROM requests")
        assert stored[0]["count"] == 10


@pytest.mark.unit
class TestDuckDBStorageGracefulDegradation:
    """Test DuckDB storage graceful degradation when DuckDB not available."""

    async def test_disabled_storage_operations(self) -> None:
        """Test storage operations when DuckDB is not available."""
        # Mock DuckDB as unavailable
        from ccproxy.observability.storage import duckdb as duckdb_module

        original_available = duckdb_module.DUCKDB_AVAILABLE
        duckdb_module.DUCKDB_AVAILABLE = False

        try:
            storage = DuckDBStorage()
            await storage.initialize()

            assert not storage.is_enabled()

            # Operations should gracefully fail
            result = await storage.store({"test": "data"})
            assert result is False

            results = await storage.query("SELECT 1")
            assert results == []

            analytics = await storage.get_analytics()
            assert analytics == {}

            health = await storage.health_check()
            assert health["status"] == "disabled"
            assert health["enabled"] is False

        finally:
            # Restore original state
            duckdb_module.DUCKDB_AVAILABLE = original_available
