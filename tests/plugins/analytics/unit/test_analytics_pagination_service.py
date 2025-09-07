"""Unit tests for AnalyticsService pagination functionality."""

from __future__ import annotations

import asyncio
import time

import pytest

from ccproxy.plugins.analytics import models as _analytics_models  # noqa: F401
from ccproxy.plugins.analytics.service import AnalyticsService
from ccproxy.plugins.duckdb_storage.storage import SimpleDuckDBStorage


def _mk(ts: float, rid: str) -> dict[str, object]:
    return {
        "request_id": rid,
        "timestamp": ts,
        "method": "POST",
        "endpoint": "/v1/messages",
        "path": "/v1/messages",
        "model": "claude-x",
        "service_type": "access_log",
        "status_code": 200,
        "duration_ms": 1.0,
    }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_service_pagination_asc_desc() -> None:
    """Test pagination with ascending and descending order."""
    storage = SimpleDuckDBStorage(":memory:")
    await storage.initialize()
    try:
        base = time.time()
        # Older -> Newer: t1 < t2 < t3
        t1, t2, t3 = base - 30, base - 20, base - 10
        for ts, rid in [(t1, "a"), (t2, "b"), (t3, "c")]:
            await storage.store_request(_mk(ts, rid))
        await asyncio.sleep(0.2)

        svc = AnalyticsService(storage._engine)

        # Descending: expect c,b then a
        p1d = svc.query_logs(limit=2, order="desc")
        assert p1d["count"] == 2
        assert p1d["next_cursor"] is not None
        p2d = svc.query_logs(limit=2, order="desc", cursor=p1d["next_cursor"])
        assert p2d["count"] == 1

        # Ascending: expect a,b then c
        p1a = svc.query_logs(limit=2, order="asc")
        assert p1a["count"] == 2
        assert p1a["next_cursor"] is not None
        p2a = svc.query_logs(limit=2, order="asc", cursor=p1a["next_cursor"])
        assert p2a["count"] == 1
    finally:
        await storage.close()
