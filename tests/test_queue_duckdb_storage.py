"""
Tests for queue-based DuckDB storage solution.

This module tests the queue-based approach that prevents deadlocks
when multiple concurrent requests attempt to write to DuckDB storage.
"""

import asyncio
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from sqlmodel import Session, select

from ccproxy.observability.storage.duckdb_simple import (
    AccessLogPayload,
    SimpleDuckDBStorage,
)
from ccproxy.observability.storage.models import AccessLog


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Create temporary database path for testing."""
    return tmp_path / "test_metrics.duckdb"


@pytest.fixture
def memory_storage() -> Generator[SimpleDuckDBStorage, None, None]:
    """Create in-memory DuckDB storage for testing."""
    storage = SimpleDuckDBStorage(":memory:")
    yield storage


@pytest.fixture
async def initialized_storage(
    memory_storage: SimpleDuckDBStorage,
) -> SimpleDuckDBStorage:
    """Create and initialize storage for testing."""
    await memory_storage.initialize()
    return memory_storage


@pytest.fixture
def sample_access_log() -> AccessLogPayload:
    """Create sample access log data for testing."""
    return {
        "request_id": "test-request-123",
        "timestamp": time.time(),
        "method": "POST",
        "endpoint": "/v1/messages",
        "path": "/v1/messages",
        "query": "",
        "client_ip": "127.0.0.1",
        "user_agent": "test-agent",
        "service_type": "proxy_service",
        "model": "claude-3-5-sonnet-20241022",
        "streaming": False,
        "status_code": 200,
        "duration_ms": 150.5,
        "duration_seconds": 0.1505,
        "tokens_input": 100,
        "tokens_output": 50,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "cost_usd": 0.002,
        "cost_sdk_usd": 0.0,
    }


class TestQueueBasedDuckDBStorage:
    """Test suite for queue-based DuckDB storage."""

    async def test_initialization_creates_background_worker(
        self, memory_storage: SimpleDuckDBStorage
    ) -> None:
        """Test that initialization starts the background worker."""
        assert not memory_storage._initialized
        assert memory_storage._background_worker_task is None

        await memory_storage.initialize()

        assert memory_storage._initialized
        assert memory_storage._background_worker_task is not None  # type: ignore[unreachable]
        assert not memory_storage._background_worker_task.done()

        await memory_storage.close()

    async def test_store_request_queues_data(
        self,
        initialized_storage: SimpleDuckDBStorage,
        sample_access_log: AccessLogPayload,
    ) -> None:
        """Test that store_request queues data instead of direct DB write."""
        # Initially queue should be empty
        assert initialized_storage._write_queue.qsize() == 0

        # Store request should queue the data
        success = await initialized_storage.store_request(sample_access_log)
        assert success is True

        # Queue should now have one item
        assert initialized_storage._write_queue.qsize() == 1

        await initialized_storage.close()

    async def test_background_worker_processes_queue(
        self,
        initialized_storage: SimpleDuckDBStorage,
        sample_access_log: AccessLogPayload,
    ) -> None:
        """Test that background worker processes queued items."""
        # Queue data
        await initialized_storage.store_request(sample_access_log)
        assert initialized_storage._write_queue.qsize() == 1

        # Give background worker time to process
        await asyncio.sleep(0.1)

        # Queue should be empty after processing
        assert initialized_storage._write_queue.qsize() == 0

        # Verify data was stored in database
        with Session(initialized_storage._engine) as session:
            result = session.exec(
                select(AccessLog).where(AccessLog.request_id == "test-request-123")
            ).first()
            assert result is not None
            assert result.request_id == "test-request-123"
            assert result.method == "POST"
            assert result.endpoint == "/v1/messages"

        await initialized_storage.close()

    async def test_concurrent_writes_no_deadlock(
        self, initialized_storage: SimpleDuckDBStorage
    ) -> None:
        """Test that multiple concurrent writes don't cause deadlocks."""
        # Create multiple access log entries
        access_logs = []
        for i in range(10):
            log_data: AccessLogPayload = {
                "request_id": f"concurrent-request-{i}",
                "timestamp": time.time(),
                "method": "POST",
                "endpoint": "/v1/messages",
                "path": "/v1/messages",
                "status_code": 200,
                "duration_ms": 100.0 + i,
                "duration_seconds": 0.1 + (i * 0.01),
                "tokens_input": 50 + i,
                "tokens_output": 25 + i,
                "cost_usd": 0.001 * (i + 1),
            }
            access_logs.append(log_data)

        # Submit all requests concurrently
        start_time = time.time()
        tasks = [initialized_storage.store_request(log) for log in access_logs]
        results = await asyncio.gather(*tasks)
        end_time = time.time()

        # All requests should succeed
        assert all(results), "Some concurrent writes failed"

        # Should complete quickly (no deadlocks)
        assert end_time - start_time < 1.0, (
            "Concurrent writes took too long (possible deadlock)"
        )

        # Give background worker time to process all items
        await asyncio.sleep(0.2)

        # Verify all data was stored
        with Session(initialized_storage._engine) as session:
            count = session.exec(select(AccessLog)).all()
            assert len(count) == 10, f"Expected 10 records, got {len(count)}"

        await initialized_storage.close()

    async def test_background_worker_handles_errors_gracefully(
        self,
        initialized_storage: SimpleDuckDBStorage,
        sample_access_log: AccessLogPayload,
    ) -> None:
        """Test that background worker continues processing after errors."""
        # Mock the sync store method to fail once then succeed
        original_method = initialized_storage._store_request_sync
        call_count = 0

        def mock_store_sync(data: AccessLogPayload) -> bool:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Simulated database error")
            return original_method(data)

        with patch.object(
            initialized_storage, "_store_request_sync", side_effect=mock_store_sync
        ):
            # Queue two requests
            log1: AccessLogPayload = {
                **sample_access_log,
                "request_id": "error-request-1",
            }
            log2: AccessLogPayload = {
                **sample_access_log,
                "request_id": "success-request-2",
            }

            await initialized_storage.store_request(log1)
            await initialized_storage.store_request(log2)

            # Give time for processing
            await asyncio.sleep(0.2)

            # Second request should succeed despite first failing
            with Session(initialized_storage._engine) as session:
                result = session.exec(
                    select(AccessLog).where(AccessLog.request_id == "success-request-2")
                ).first()
                assert result is not None

        await initialized_storage.close()

    async def test_graceful_shutdown_processes_remaining_queue(
        self,
        initialized_storage: SimpleDuckDBStorage,
        sample_access_log: AccessLogPayload,
    ) -> None:
        """Test that shutdown waits for queue processing to complete."""
        # Queue multiple items
        for i in range(3):
            log_data: AccessLogPayload = {
                **sample_access_log,
                "request_id": f"shutdown-test-{i}",
            }
            await initialized_storage.store_request(log_data)

        assert initialized_storage._write_queue.qsize() == 3

        # Close should process all queued items
        await initialized_storage.close()

        # Verify all items were processed (queue should be empty)
        assert initialized_storage._write_queue.qsize() == 0

    async def test_store_request_fails_when_not_initialized(
        self, memory_storage: SimpleDuckDBStorage, sample_access_log: AccessLogPayload
    ) -> None:
        """Test that store_request fails when storage is not initialized."""
        # Storage not initialized
        assert not memory_storage._initialized

        # Store request should fail
        success = await memory_storage.store_request(sample_access_log)
        assert success is False

    async def test_queue_timeout_handling(
        self, initialized_storage: SimpleDuckDBStorage
    ) -> None:
        """Test that background worker handles queue timeouts correctly."""
        # Background worker should be running and handling timeouts
        assert initialized_storage._background_worker_task is not None
        assert not initialized_storage._background_worker_task.done()

        # Wait a bit to ensure timeout handling works
        await asyncio.sleep(0.1)

        # Worker should still be running
        assert not initialized_storage._background_worker_task.done()

        await initialized_storage.close()

    async def test_file_based_storage(
        self, temp_db_path: Path, sample_access_log: AccessLogPayload
    ) -> None:
        """Test queue-based storage with file-based database."""
        storage = SimpleDuckDBStorage(temp_db_path)

        try:
            await storage.initialize()

            # Database file should be created
            assert temp_db_path.exists()

            # Store data
            success = await storage.store_request(sample_access_log)
            assert success is True

            # Give background worker time to process
            await asyncio.sleep(0.1)

            # Verify data persistence
            with Session(storage._engine) as session:
                result = session.exec(
                    select(AccessLog).where(AccessLog.request_id == "test-request-123")
                ).first()
                assert result is not None

        finally:
            await storage.close()

    async def test_health_check_with_queue_storage(
        self,
        initialized_storage: SimpleDuckDBStorage,
        sample_access_log: AccessLogPayload,
    ) -> None:
        """Test health check works with queue-based storage."""
        # Initial health check
        health = await initialized_storage.health_check()
        assert health["status"] == "healthy"
        assert health["enabled"] is True
        assert health["access_log_count"] == 0

        # Store some data
        await initialized_storage.store_request(sample_access_log)
        await asyncio.sleep(0.1)  # Let background worker process

        # Health check after data storage
        health_after = await initialized_storage.health_check()
        assert health_after["status"] == "healthy"
        assert health_after["access_log_count"] == 1

        await initialized_storage.close()

    async def test_multiple_storage_instances_no_conflict(
        self, temp_db_path: Path, sample_access_log: AccessLogPayload
    ) -> None:
        """Test that multiple storage instances can coexist without conflicts."""
        # Create two separate storage instances
        storage1 = SimpleDuckDBStorage(temp_db_path)
        storage2 = SimpleDuckDBStorage(":memory:")

        try:
            await storage1.initialize()
            await storage2.initialize()

            # Store data in both
            log1: AccessLogPayload = {
                **sample_access_log,
                "request_id": "storage1-request",
            }
            log2: AccessLogPayload = {
                **sample_access_log,
                "request_id": "storage2-request",
            }

            success1 = await storage1.store_request(log1)
            success2 = await storage2.store_request(log2)

            assert success1 is True
            assert success2 is True

            # Give time for processing
            await asyncio.sleep(0.2)

            # Verify isolation - each storage has its own data
            with Session(storage1._engine) as session:
                result1 = session.exec(
                    select(AccessLog).where(AccessLog.request_id == "storage1-request")
                ).first()
                assert result1 is not None

            with Session(storage2._engine) as session:
                result2 = session.exec(
                    select(AccessLog).where(AccessLog.request_id == "storage2-request")
                ).first()
                assert result2 is not None

        finally:
            await storage1.close()
            await storage2.close()


class TestQueueBasedStoragePerformance:
    """Performance tests for queue-based storage."""

    @pytest.mark.unit
    async def test_high_throughput_no_deadlock(
        self, initialized_storage: SimpleDuckDBStorage
    ) -> None:
        """Test high-throughput scenario doesn't cause deadlocks."""
        num_requests = 50
        access_logs = []

        # Generate many log entries
        for i in range(num_requests):
            log_data: AccessLogPayload = {
                "request_id": f"perf-test-{i}",
                "timestamp": time.time(),
                "method": "POST",
                "endpoint": "/v1/messages",
                "status_code": 200,
                "duration_ms": 100.0,
            }
            access_logs.append(log_data)

        # Submit all at once
        start_time = time.time()
        tasks = [initialized_storage.store_request(log) for log in access_logs]
        results = await asyncio.gather(*tasks)
        queue_time = time.time() - start_time

        # All should succeed quickly
        assert all(results), "Some high-throughput writes failed"
        assert queue_time < 2.0, f"Queuing took too long: {queue_time}s"

        # Give background worker time to process
        await asyncio.sleep(1.0)

        # Verify all processed
        with Session(initialized_storage._engine) as session:
            count = len(session.exec(select(AccessLog)).all())
            assert count == num_requests

        await initialized_storage.close()

    @pytest.mark.unit
    async def test_queue_memory_usage_bounded(
        self, initialized_storage: SimpleDuckDBStorage
    ) -> None:
        """Test that queue doesn't grow unbounded under load."""
        # Submit many requests rapidly
        for i in range(20):
            log_data: AccessLogPayload = {
                "request_id": f"memory-test-{i}",
                "timestamp": time.time(),
                "method": "POST",
                "endpoint": "/v1/messages",
                "status_code": 200,
                "duration_ms": 50.0,
            }
            await initialized_storage.store_request(log_data)

        # Queue should not grow excessively
        max_queue_size = initialized_storage._write_queue.qsize()
        assert max_queue_size <= 25, f"Queue size too large: {max_queue_size}"

        # Give time for processing
        await asyncio.sleep(0.5)

        # Queue should be mostly empty
        final_queue_size = initialized_storage._write_queue.qsize()
        assert final_queue_size <= 5, (
            f"Queue not processing efficiently: {final_queue_size}"
        )

        await initialized_storage.close()
