from __future__ import annotations

import random
import time
from pathlib import Path

import anyio
import pytest

from ccproxy.plugins.analytics.service import AnalyticsService
from ccproxy.plugins.duckdb_storage.storage import SimpleDuckDBStorage


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_duckdb_analytics_pagination_stress(tmp_path: Path) -> None:
    storage = SimpleDuckDBStorage(tmp_path / "pagination.duckdb")
    await storage.initialize()
    engine = storage._engine
    assert engine is not None

    total_logs = 1500
    base_time = time.time()

    for index in range(total_logs):
        payload = {
            "request_id": f"log-{index}",
            "timestamp": base_time - index,
            "method": "POST",
            "endpoint": f"/v1/tools/{index % 5}",
            "path": f"/v1/tools/{index % 5}",
            "query": "",
            "client_ip": "127.0.0.1",
            "user_agent": "pytest-agent",
            "service_type": "analytics-stress",
            "provider": "duckdb",
            "model": f"model-{index % 3}",
            "streaming": False,
            "status_code": 200,
            "duration_ms": random.uniform(5, 40),
            "duration_seconds": 0.02,
            "tokens_input": index % 11,
            "tokens_output": index % 7,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "cost_usd": 0.0001,
        }
        await storage.store_request(payload)

    svc = AnalyticsService(engine)
    limit = 200
    cursor = None
    fetched = 0
    iterations = 0

    while True:
        page = svc.query_logs(limit=limit, cursor=cursor, order="desc")
        fetched += page["count"]
        iterations += 1

        if not page["has_more"]:
            break

        cursor = page["next_cursor"]
        assert cursor is not None
        assert page["count"] == limit
        assert page["results"][0]["timestamp"] >= page["results"][-1]["timestamp"]

        # Yield control periodically to avoid tight loop in async context
        await anyio.sleep(0)

    assert fetched == total_logs
    assert iterations >= 2
