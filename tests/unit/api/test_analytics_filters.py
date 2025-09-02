from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from plugins.analytics import models as _analytics_models  # noqa: F401
from plugins.analytics.service import AnalyticsService
from plugins.analytics.routes import get_duckdb_storage
from plugins.analytics.routes import router as analytics_router
from plugins.duckdb_storage.storage import SimpleDuckDBStorage


@pytest.fixture(autouse=True)
async def storage() -> AsyncGenerator[SimpleDuckDBStorage, None]:
    s = SimpleDuckDBStorage(":memory:")
    await s.initialize()
    try:
        yield s
    finally:
        await s.close()


@pytest.fixture
def app(storage: SimpleDuckDBStorage) -> FastAPI:
    app = FastAPI()
    app.include_router(analytics_router, prefix="/logs")
    app.state.log_storage = storage
    app.dependency_overrides[get_duckdb_storage] = lambda: storage
    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


def _mk_log(ts: float, model: str, service_type: str, status: int) -> dict[str, object]:
    return {
        "request_id": f"rid-{ts}-{model}",
        "timestamp": ts,
        "method": "POST",
        "endpoint": "/v1/messages",
        "path": "/v1/messages",
        "model": model,
        "service_type": service_type,
        "status_code": status,
        "duration_ms": 1.0,
    }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_query_filters_model_and_time(storage: SimpleDuckDBStorage) -> None:
    base = time.time()
    logs = [
        _mk_log(base - 300, "claude-1", "access_log", 200),
        _mk_log(base - 100, "claude-2", "access_log", 200),
        _mk_log(base - 50, "claude-2", "access_log", 400),
    ]
    for entry in logs:
        await storage.store_request(entry)
    await asyncio.sleep(0.2)

    svc = AnalyticsService(storage._engine)
    data = svc.query_logs(
        limit=100,
        start_time=base - 120,
        end_time=base + 10,
        model="claude-2",
        order="asc",
    )
    assert data["count"] == 2
