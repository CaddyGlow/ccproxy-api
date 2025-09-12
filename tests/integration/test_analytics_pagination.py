"""
Integration test for analytics /logs/query with cursor pagination
and presence of provider, client_ip, and user_agent fields.
"""

import asyncio
import time
from collections.abc import AsyncGenerator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ccproxy.core.async_task_manager import start_task_manager, stop_task_manager

# Ensure SQLModel knows about AccessLog before storage init
from ccproxy.plugins.analytics import models as _analytics_models  # noqa: F401
from ccproxy.plugins.analytics.routes import get_duckdb_storage
from ccproxy.plugins.analytics.routes import router as analytics_router
from ccproxy.plugins.duckdb_storage.storage import SimpleDuckDBStorage


@pytest.fixture(autouse=True)
async def task_manager_fixture():
    """Start and stop the global async task manager for background tasks."""
    await start_task_manager()
    yield
    await stop_task_manager()


@pytest.fixture
async def storage() -> AsyncGenerator[SimpleDuckDBStorage, None]:
    """In-memory DuckDB storage initialized with analytics schema."""
    storage = SimpleDuckDBStorage(":memory:")
    await storage.initialize()
    try:
        yield storage
    finally:
        await storage.close()


@pytest.fixture
def app(storage: SimpleDuckDBStorage) -> FastAPI:
    """FastAPI app mounting analytics routes and overriding storage dep."""
    app = FastAPI()
    app.include_router(analytics_router, prefix="/logs")

    # Make storage available to dependency
    app.state.log_storage = storage

    # Override dependency to return our test storage
    app.dependency_overrides[get_duckdb_storage] = lambda: storage
    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


class TestAnalyticsQueryCursor:
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_query_with_cursor_pagination(
        self, storage: SimpleDuckDBStorage, client: TestClient
    ) -> None:
        """Stores 3 logs and paginates with a timestamp cursor."""
        base = time.time()
        logs = []
        for i in range(3):
            ts = base - (3 - i)  # strictly increasing across inserts
            logs.append(
                {
                    "request_id": f"req-{i}",
                    "timestamp": ts,
                    "method": "POST",
                    "endpoint": "/v1/messages",
                    "path": "/v1/messages",
                    "query": "",
                    "client_ip": f"127.0.0.{i + 1}",
                    "user_agent": "pytest-agent/1.0",
                    "service_type": "access_log",
                    "provider": "anthropic",
                    "model": "claude-3-5-sonnet-20241022",
                    "status_code": 200,
                    "duration_ms": 100.0 + i,
                    "duration_seconds": (100.0 + i) / 1000.0,
                    "tokens_input": 10 + i,
                    "tokens_output": 5 + i,
                    "cache_read_tokens": 0,
                    "cache_write_tokens": 0,
                    "cost_usd": 0.001 * (i + 1),
                    "cost_sdk_usd": 0.0,
                }
            )

        # Queue writes
        for entry in logs:
            await storage.store_request(entry)

        # Let background worker flush (optimized for tests)
        await asyncio.sleep(0.01)

        # First page: newest first, limit 2
        r1 = client.get("/logs/query", params={"limit": 2, "order": "desc"})
        assert r1.status_code == 200
        d1 = r1.json()
        assert d1["count"] == 2
        assert d1["has_more"] is True
        assert d1.get("next_cursor") is not None

        # Ensure provider and client_ip/user_agent are present
        for item in d1["results"]:
            assert item["provider"] == "anthropic"
            assert item["client_ip"].startswith("127.0.0.")
            assert item["user_agent"] == "pytest-agent/1.0"

        # Second page using returned cursor
        cursor = d1["next_cursor"]
        r2 = client.get(
            "/logs/query", params={"limit": 2, "order": "desc", "cursor": cursor}
        )
        assert r2.status_code == 200
        d2 = r2.json()
        assert d2["count"] == 1
        assert d2["has_more"] is False

        # Validate last record
        last = d2["results"][0]
        assert last["request_id"] in {"req-0", "req-1", "req-2"}
        assert last["provider"] == "anthropic"
        assert last["client_ip"].startswith("127.0.0.")
        assert last["user_agent"] == "pytest-agent/1.0"
