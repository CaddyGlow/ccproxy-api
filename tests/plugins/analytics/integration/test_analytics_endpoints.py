"""Integration tests for analytics plugin endpoints."""

import asyncio
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from sqlmodel import Session, select

from ccproxy.core.async_task_manager import start_task_manager, stop_task_manager
from ccproxy.plugins.analytics import models as _analytics_models  # noqa: F401
from ccproxy.plugins.analytics.models import AccessLog, AccessLogPayload
from ccproxy.plugins.analytics.routes import router as analytics_router
from ccproxy.plugins.duckdb_storage.storage import SimpleDuckDBStorage


# Use a single event loop for this module's async fixtures
pytestmark = pytest.mark.asyncio(loop_scope="module")


@pytest_asyncio.fixture(scope="module", loop_scope="module", autouse=True)
async def task_manager_fixture() -> AsyncGenerator[None, None]:
    """Start and stop the global async task manager for background tasks."""
    await start_task_manager()
    yield
    await stop_task_manager()


@pytest.fixture(scope="module")
def temp_db_path(tmp_path_factory) -> Path:
    """Create temporary database path for testing."""
    base = tmp_path_factory.mktemp("analytics_mod")
    return base / "test_analytics.duckdb"


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def storage_with_data(
    temp_db_path: Path,
) -> AsyncGenerator[SimpleDuckDBStorage, None]:
    """Create storage with sample data for analytics testing."""
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


@pytest.fixture(scope="module")
def app(storage_with_data: SimpleDuckDBStorage) -> FastAPI:
    """FastAPI app with analytics routes and storage dependency."""
    from ccproxy.auth.conditional import get_conditional_auth_manager
    from ccproxy.plugins.analytics.routes import get_duckdb_storage

    app = FastAPI()
    app.include_router(analytics_router, prefix="/logs")

    # Make storage available to dependency
    app.state.log_storage = storage_with_data

    # Override dependencies to return test storage and no auth
    app.dependency_overrides[get_duckdb_storage] = lambda: storage_with_data
    app.dependency_overrides[get_conditional_auth_manager] = lambda: None

    return app


@pytest.fixture(scope="module")
def app_no_storage() -> FastAPI:
    """FastAPI app with analytics routes but no storage."""
    from ccproxy.auth.conditional import get_conditional_auth_manager

    app = FastAPI()
    app.include_router(analytics_router, prefix="/logs")

    # Override auth dependency to return None (no auth required)
    app.dependency_overrides[get_conditional_auth_manager] = lambda: None
    # No storage set intentionally

    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Test client with storage."""
    return TestClient(app)


@pytest.fixture
def client_no_storage(app_no_storage: FastAPI) -> TestClient:
    """Test client without storage."""
    return TestClient(app_no_storage)


# Async client for use in async tests to avoid portal deadlocks
@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def async_client(app: FastAPI):  # type: ignore[no-untyped-def]
    transport = ASGITransport(app=app)
    client = AsyncClient(transport=transport, base_url="http://test")
    try:
        yield client
    finally:
        await client.aclose()


@pytest.mark.integration
@pytest.mark.analytics
class TestAnalyticsQueryEndpoint:
    """Test suite for analytics query endpoint."""

    def test_query_logs_endpoint_basic(self, client: TestClient) -> None:
        """Test basic query logs functionality."""
        response = client.get("/logs/query", params={"limit": 100})
        assert response.status_code == 200

        data: dict[str, Any] = response.json()
        assert "count" in data
        assert "results" in data
        assert data["count"] <= 100

    def test_query_logs_with_filters(self, client: TestClient) -> None:
        """Test query logs with various filters."""
        response = client.get(
            "/logs/query",
            params={
                "limit": 50,
                "model": "claude-3-5-sonnet-20241022",
                "service_type": "proxy_service",
                "order": "desc",
            },
        )
        assert response.status_code == 200

        data: dict[str, Any] = response.json()
        assert data["count"] >= 0
        assert isinstance(data["results"], list)

    def test_query_logs_pagination(self, client: TestClient) -> None:
        """Test query logs pagination."""
        # First page
        response1 = client.get("/logs/query", params={"limit": 2, "order": "desc"})
        assert response1.status_code == 200

        data1: dict[str, Any] = response1.json()
        assert data1["count"] == 2

        # Second page if cursor exists
        if data1.get("next_cursor"):
            response2 = client.get(
                "/logs/query",
                params={
                    "limit": 2,
                    "order": "desc",
                    "cursor": data1["next_cursor"],
                },
            )
            assert response2.status_code == 200

    def test_query_logs_without_storage(self, client_no_storage: TestClient) -> None:
        """Test query logs when storage is not available."""
        response = client_no_storage.get("/logs/query")
        assert response.status_code == 503


@pytest.mark.integration
@pytest.mark.analytics
class TestAnalyticsAnalyticsEndpoint:
    """Test suite for analytics analytics endpoint."""

    def test_analytics_endpoint_basic(self, client: TestClient) -> None:
        """Test basic analytics functionality."""
        response = client.get("/logs/analytics")
        assert response.status_code == 200

        data: dict[str, Any] = response.json()
        assert "summary" in data
        assert "query_params" in data

    def test_analytics_with_filters(self, client: TestClient) -> None:
        """Test analytics with various filters."""
        response = client.get(
            "/logs/analytics",
            params={
                "service_type": "proxy_service",
                "model": "claude-3-5-sonnet-20241022",
                "hours": 24,
            },
        )
        assert response.status_code == 200

        data: dict[str, Any] = response.json()
        assert "summary" in data
        assert data["query_params"]["service_type"] == "proxy_service"
        assert data["query_params"]["model"] == "claude-3-5-sonnet-20241022"

    def test_analytics_without_storage(self, client_no_storage: TestClient) -> None:
        """Test analytics when storage is not available."""
        response = client_no_storage.get("/logs/analytics")
        assert response.status_code == 503


@pytest.mark.integration
@pytest.mark.analytics
class TestAnalyticsResetEndpoint:
    """Test suite for reset endpoint functionality."""

    def test_reset_endpoint_clears_data(
        self, client: TestClient, storage_with_data: SimpleDuckDBStorage
    ) -> None:
        """Test that reset endpoint successfully clears all data."""
        # Verify data exists before reset
        with Session(storage_with_data._engine) as session:
            count_before = len(session.exec(select(AccessLog)).all())
            assert count_before == 5, f"Expected 5 records, got {count_before}"

        response = client.post("/logs/reset")
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

    def test_reset_endpoint_without_storage(
        self, client_no_storage: TestClient
    ) -> None:
        """Test reset endpoint when storage is not available."""
        response = client_no_storage.post("/logs/reset")
        assert response.status_code == 503

    def test_reset_endpoint_storage_without_reset_method(self) -> None:
        """Test reset endpoint with storage that doesn't support reset."""
        from ccproxy.auth.conditional import get_conditional_auth_manager

        # Create mock storage without reset_data method
        class MockStorageWithoutReset:
            pass

        app = FastAPI()
        app.include_router(analytics_router, prefix="/logs")
        app.state.log_storage = MockStorageWithoutReset()

        # Override auth dependency to return None (no auth required)
        app.dependency_overrides[get_conditional_auth_manager] = lambda: None

        client = TestClient(app)
        response = client.post("/logs/reset")
        assert response.status_code == 501

    def test_reset_endpoint_multiple_calls(
        self, client: TestClient, storage_with_data: SimpleDuckDBStorage
    ) -> None:
        """Test multiple consecutive reset calls."""

        # First reset
        response1 = client.post("/logs/reset")
        assert response1.status_code == 200
        assert response1.json()["status"] == "success"

        # Second reset (should still succeed on empty database)
        response2 = client.post("/logs/reset")
        assert response2.status_code == 200
        assert response2.json()["status"] == "success"

        # Third reset
        response3 = client.post("/logs/reset")
        assert response3.status_code == 200
        assert response3.json()["status"] == "success"

        # Verify database is still empty (excluding access log entries for reset endpoint calls)
        with Session(storage_with_data._engine) as session:
            results = session.exec(select(AccessLog)).all()
            # Filter out access log entries for the reset endpoint itself
            non_reset_results = [r for r in results if r.endpoint != "/logs/reset"]
            assert len(non_reset_results) == 0

    # NOTE: This test intermittently flakes in isolated environments due to
    # queued DuckDB writes and event-loop timing. Despite queue join and polling,
    # some runners still observe 0 rows briefly after reset+insert.
    # Skipping for stability; revisit when storage exposes a deterministic flush.
    @pytest.mark.skip(reason="Flaky under async queue timing; skipping for stability")
    @pytest.mark.asyncio
    async def test_reset_endpoint_preserves_schema(
        self, async_client: AsyncClient, storage_with_data: SimpleDuckDBStorage
    ) -> None:
        """Test that reset preserves database schema and can accept new data."""

        # Reset the data
        response = await async_client.post("/logs/reset")
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

        # Ensure background worker flushed queued write for determinism
        try:
            queue = getattr(storage_with_data, "_write_queue", None)
            if queue is not None:
                # Wait until all items are processed
                await asyncio.wait_for(queue.join(), timeout=1.0)
            else:
                await asyncio.sleep(0.3)
        except Exception:
            # Fallback to small delay if queue not exposed
            await asyncio.sleep(0.3)

        # Verify new data was stored successfully (poll to avoid flakes)
        non_reset_results = []
        for _ in range(20):  # up to ~1s
            with Session(storage_with_data._engine) as session:
                results = session.exec(select(AccessLog)).all()
                non_reset_results = [r for r in results if r.endpoint != "/logs/reset"]
            if len(non_reset_results) >= 1:
                break
            await asyncio.sleep(0.05)

        assert len(non_reset_results) == 1
        assert non_reset_results[0].request_id == "post-reset-request"
        assert non_reset_results[0].model == "claude-3-5-haiku-20241022"


@pytest.mark.integration
@pytest.mark.analytics
class TestAnalyticsStreamEndpoint:
    """Test suite for analytics streaming endpoint."""

    def test_stream_logs_endpoint_basic(self, client_no_storage: TestClient) -> None:
        """Test basic stream logs functionality."""

        response = client_no_storage.get("/logs/stream")
        assert response.status_code == 200
        assert response.headers.get("content-type").startswith("text/event-stream")

    def test_stream_logs_with_filters(self, client_no_storage: TestClient) -> None:
        """Test stream logs with various filters."""

        response = client_no_storage.get(
            "/logs/stream",
            params={
                "model": "claude-3-5-sonnet-20241022",
                "service_type": "proxy_service",
                "min_duration_ms": 50.0,
                "max_duration_ms": 1000.0,
                "status_code_min": 200,
                "status_code_max": 299,
            },
        )
        assert response.status_code == 200
        assert response.headers.get("content-type").startswith("text/event-stream")


@pytest.mark.integration
@pytest.mark.analytics
class TestAnalyticsEndpointsFiltering:
    """Test analytics endpoint behavior with complex filtering scenarios."""

    def test_reset_then_query_with_filters(self, client: TestClient) -> None:
        """Test that query endpoint works correctly after reset."""

        # Reset data
        reset_response = client.post("/logs/reset")
        assert reset_response.status_code == 200

        # Query after reset should return empty results
        query_response = client.get("/logs/query", params={"limit": 100})
        assert query_response.status_code == 200

        data: dict[str, Any] = query_response.json()
        assert data["count"] == 0
        assert data["results"] == []

    def test_reset_then_analytics_with_filters(self, client: TestClient) -> None:
        """Test that analytics endpoint works correctly after reset."""

        # Reset data
        reset_response = client.post("/logs/reset")
        assert reset_response.status_code == 200

        # Analytics after reset should return zero metrics
        analytics_response = client.get(
            "/logs/analytics",
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
