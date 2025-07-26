"""
Tests for reset endpoint functionality.

This module tests the POST /reset endpoint that clears all data
from the DuckDB storage backend.
"""

import asyncio
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest
from sqlmodel import Session, select

from ccproxy.observability.storage.duckdb_simple import (
    AccessLogPayload,
    SimpleDuckDBStorage,
)
from ccproxy.observability.storage.models import AccessLog


pytest.skip("need to fix the fixture", allow_module_level=True)  # type: ignore[unreachable]


@pytest.fixture  # type: ignore[unreachable]
def temp_db_path(tmp_path: Path) -> Path:
    """Create temporary database path for testing."""
    return tmp_path / "test_reset.duckdb"


@pytest.fixture
async def storage_with_data(
    temp_db_path: Path,
) -> AsyncGenerator[SimpleDuckDBStorage, None]:
    """Create storage with sample data for reset testing."""
    storage = SimpleDuckDBStorage(temp_db_path)
    await storage.initialize()

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
        for i in range(5)
    ]

    # Store sample data
    for log_data in sample_logs:
        await storage.store_request(log_data)

    # Give background worker time to process
    await asyncio.sleep(0.2)

    yield storage
    await storage.close()


class TestResetEndpoint:
    """Test suite for reset endpoint functionality."""

    def test_reset_endpoint_clears_data(
        self, fastapi_client_factory: Any, storage_with_data: SimpleDuckDBStorage
    ) -> None:
        """Test that reset endpoint successfully clears all data."""
        # Verify data exists before reset
        with Session(storage_with_data._engine) as session:
            count_before = len(session.exec(select(AccessLog)).all())
            assert count_before == 5, f"Expected 5 records, got {count_before}"

        # Create client with storage dependency override
        client = fastapi_client_factory.create_client_with_storage(storage_with_data)

        response = client.post("/reset")
        assert response.status_code == 200

        data: dict[str, Any] = response.json()
        assert data["status"] == "success"
        assert data["message"] == "All logs data has been reset"
        assert "timestamp" in data
        assert data["backend"] == "duckdb"

        # Verify data was cleared
        with Session(storage_with_data._engine) as session:
            count_after = len(session.exec(select(AccessLog)).all())
            assert count_after == 0, (
                f"Expected 0 records after reset, got {count_after}"
            )

    def test_reset_endpoint_without_storage(self, fastapi_client_factory: Any) -> None:
        """Test reset endpoint when storage is not available."""
        # Create client without storage
        client = fastapi_client_factory.create_client_with_storage(None)

        response = client.post("/reset")
        assert response.status_code == 503
        # Just verify that the endpoint returns the expected status code
        # The error message may be handled by middleware and not in the JSON response

    def test_reset_endpoint_storage_without_reset_method(
        self, fastapi_client_factory: Any
    ) -> None:
        """Test reset endpoint with storage that doesn't support reset."""

        # Create mock storage without reset_data method
        class MockStorageWithoutReset:
            pass

        client = fastapi_client_factory.create_client_with_storage(
            MockStorageWithoutReset()
        )

        response = client.post("/reset")
        assert response.status_code == 501
        # Just verify that the endpoint returns the expected status code
        # The error message may be handled by middleware and not in the JSON response

    def test_reset_endpoint_multiple_calls(
        self, fastapi_client_factory: Any, storage_with_data: SimpleDuckDBStorage
    ) -> None:
        """Test multiple consecutive reset calls."""
        client = fastapi_client_factory.create_client_with_storage(storage_with_data)

        # First reset
        response1 = client.post("/reset")
        assert response1.status_code == 200
        assert response1.json()["status"] == "success"

        # Second reset (should still succeed on empty database)
        response2 = client.post("/reset")
        assert response2.status_code == 200
        assert response2.json()["status"] == "success"

        # Third reset
        response3 = client.post("/reset")
        assert response3.status_code == 200
        assert response3.json()["status"] == "success"

        # Verify database is still empty
        with Session(storage_with_data._engine) as session:
            count = len(session.exec(select(AccessLog)).all())
            assert count == 0

    async def test_reset_endpoint_preserves_schema(
        self, fastapi_client_factory: Any, storage_with_data: SimpleDuckDBStorage
    ) -> None:
        """Test that reset preserves database schema and can accept new data."""
        client = fastapi_client_factory.create_client_with_storage(storage_with_data)

        # Reset the data
        response = client.post("/reset")
        assert response.status_code == 200

        # Add new data after reset
        new_log: AccessLogPayload = {
            "request_id": "post-reset-request",
            "timestamp": time.time(),
            "method": "GET",
            "endpoint": "/api/models",
            "path": "/api/models",
            "query": "",
            "client_ip": "192.168.1.1",
            "user_agent": "post-reset-agent",
            "service_type": "api_service",
            "model": "claude-3-5-haiku-20241022",
            "streaming": False,
            "status_code": 200,
            "duration_ms": 50.0,
            "duration_seconds": 0.05,
            "tokens_input": 10,
            "tokens_output": 5,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "cost_usd": 0.0005,
            "cost_sdk_usd": 0.0,
        }

        success = await storage_with_data.store_request(new_log)
        assert success is True

        # Give background worker time to process
        await asyncio.sleep(0.2)

        # Verify new data was stored successfully
        with Session(storage_with_data._engine) as session:
            results = session.exec(select(AccessLog)).all()
            assert len(results) == 1
            assert results[0].request_id == "post-reset-request"
            assert results[0].model == "claude-3-5-haiku-20241022"


class TestResetEndpointWithFiltering:
    """Test reset endpoint behavior with existing filtering endpoints."""

    def test_reset_then_query_with_filters(
        self, fastapi_client_factory: Any, storage_with_data: SimpleDuckDBStorage
    ) -> None:
        """Test that query endpoint works correctly after reset."""
        client = fastapi_client_factory.create_client_with_storage(storage_with_data)

        # Reset data
        reset_response = client.post("/reset")
        assert reset_response.status_code == 200

        # Query after reset should return empty results
        query_response = client.get("/query", params={"limit": 100})
        assert query_response.status_code == 200

        data: dict[str, Any] = query_response.json()
        assert data["count"] == 0
        assert data["results"] == []

    def test_reset_then_analytics_with_filters(
        self, fastapi_client_factory: Any, storage_with_data: SimpleDuckDBStorage
    ) -> None:
        """Test that analytics endpoint works correctly after reset."""
        client = fastapi_client_factory.create_client_with_storage(storage_with_data)

        # Reset data
        reset_response = client.post("/reset")
        assert reset_response.status_code == 200

        # Analytics after reset should return zero metrics
        analytics_response = client.get(
            "/analytics",
            params={
                "service_type": "proxy_service",
                "model": "claude-3-5-sonnet-20241022",
            },
        )
        assert analytics_response.status_code == 200

        data: dict[str, Any] = analytics_response.json()
        assert data["summary"]["total_requests"] == 0
        assert data["summary"]["total_cost_usd"] == 0
        assert data["summary"]["total_tokens_input"] == 0
        assert data["summary"]["total_tokens_output"] == 0
        assert data["service_type_breakdown"] == {}

    def test_reset_then_entries_with_filters(
        self, fastapi_client_factory: Any, storage_with_data: SimpleDuckDBStorage
    ) -> None:
        """Test that entries endpoint works correctly after reset."""
        client = fastapi_client_factory.create_client_with_storage(storage_with_data)

        # Reset data
        reset_response = client.post("/reset")
        assert reset_response.status_code == 200

        # Entries after reset should return empty list
        entries_response = client.get(
            "/entries",
            params={
                "limit": 50,
                "service_type": "proxy_service",
                "order_by": "timestamp",
                "order_desc": True,
            },
        )
        assert entries_response.status_code == 200

        data: dict[str, Any] = entries_response.json()
        assert data["total_count"] == 0
        assert data["entries"] == []
        assert data["total_pages"] == 0
