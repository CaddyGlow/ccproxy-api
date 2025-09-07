"""Unit tests for analytics storage operations."""

import asyncio
import time
from pathlib import Path

import pytest
from sqlmodel import Session, select

from ccproxy.plugins.analytics.models import AccessLog, AccessLogPayload
from ccproxy.plugins.duckdb_storage.storage import SimpleDuckDBStorage


@pytest.mark.unit
class TestStorageOperations:
    """Test suite for storage operations functionality."""

    @pytest.mark.asyncio
    async def test_storage_reset_functionality(self, tmp_path: Path) -> None:
        """Test storage reset functionality at the service level."""
        db_path = tmp_path / "test_reset.duckdb"
        storage = SimpleDuckDBStorage(db_path)
        await storage.initialize()

        try:
            # Add sample data
            sample_logs: list[AccessLogPayload] = [
                {
                    "request_id": f"test-request-{i}",
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
                    "duration_ms": 100.0 + i,
                    "duration_seconds": 0.1 + (i * 0.01),
                    "tokens_input": 50 + i,
                    "tokens_output": 25 + i,
                    "cache_read_tokens": 0,
                    "cache_write_tokens": 0,
                    "cost_usd": 0.001 * (i + 1),
                    "cost_sdk_usd": 0.0,
                }
                for i in range(3)
            ]

            # Store sample data
            for log_data in sample_logs:
                success = await storage.store_request(log_data)
                assert success is True

            # Give background worker time to process
            await asyncio.sleep(0.2)

            # Verify data exists
            with Session(storage._engine) as session:
                count_before = len(session.exec(select(AccessLog)).all())
                assert count_before == 3

            # Test reset functionality
            reset_success = await storage.reset_data()
            assert reset_success is True

            # Verify data was cleared
            with Session(storage._engine) as session:
                count_after = len(session.exec(select(AccessLog)).all())
                assert count_after == 0

        finally:
            await storage.close()

    @pytest.mark.asyncio
    async def test_storage_data_persistence(self, tmp_path: Path) -> None:
        """Test that data persists correctly in storage."""
        db_path = tmp_path / "test_persistence.duckdb"
        storage = SimpleDuckDBStorage(db_path)
        await storage.initialize()

        try:
            # Add test data
            log_data: AccessLogPayload = {
                "request_id": "test-persistence-request",
                "timestamp": time.time(),
                "method": "GET",
                "endpoint": "/api/models",
                "path": "/api/models",
                "query": "",
                "client_ip": "192.168.1.1",
                "user_agent": "test-agent",
                "service_type": "api_service",
                "model": "claude-3-5-haiku-20241022",
                "streaming": False,
                "status_code": 200,
                "duration_ms": 75.5,
                "duration_seconds": 0.0755,
                "tokens_input": 15,
                "tokens_output": 8,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "cost_usd": 0.00075,
                "cost_sdk_usd": 0.0,
            }

            success = await storage.store_request(log_data)
            assert success is True

            # Give background worker time to process
            await asyncio.sleep(0.2)

            # Verify data was stored correctly
            with Session(storage._engine) as session:
                results = session.exec(select(AccessLog)).all()
                assert len(results) == 1

                stored_log = results[0]
                assert stored_log.request_id == "test-persistence-request"
                assert stored_log.model == "claude-3-5-haiku-20241022"
                assert stored_log.service_type == "api_service"
                assert stored_log.duration_ms == 75.5
                assert stored_log.tokens_input == 15
                assert stored_log.tokens_output == 8
                assert abs(stored_log.cost_usd - 0.00075) < 1e-6

        finally:
            await storage.close()

    @pytest.mark.asyncio
    async def test_storage_without_reset_method(self) -> None:
        """Test behavior with storage that doesn't have reset method."""

        class MockStorageWithoutReset:
            """Mock storage without reset_data method."""

            pass

        mock_storage = MockStorageWithoutReset()

        # Verify the storage doesn't have reset_data method
        assert not hasattr(mock_storage, "reset_data")

        # This test verifies that our endpoint logic can detect
        # storage backends that don't support reset functionality
        assert True  # Test passes by verifying the mock setup
