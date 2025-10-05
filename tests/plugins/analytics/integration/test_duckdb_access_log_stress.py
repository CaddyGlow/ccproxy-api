from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import anyio
import pytest
from sqlmodel import Session, select

from ccproxy.core.plugins.hooks.base import HookContext
from ccproxy.core.plugins.hooks.events import HookEvent
from ccproxy.plugins.access_log.config import AccessLogConfig
from ccproxy.plugins.access_log.hook import AccessLogHook
from ccproxy.plugins.analytics.ingest import AnalyticsIngestService
from ccproxy.plugins.analytics.models import AccessLog
from ccproxy.plugins.duckdb_storage.storage import SimpleDuckDBStorage


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_duckdb_access_log_ingest_stress(tmp_path: Path) -> None:
    """High-volume ingest ensures AccessLogHook + DuckDB stay in sync."""

    storage = SimpleDuckDBStorage(tmp_path / "stress.duckdb")
    await storage.initialize()
    engine = storage._engine
    assert engine is not None

    ingest_service = AnalyticsIngestService(engine)
    access_log_config = AccessLogConfig(
        client_log_file=str(tmp_path / "access.log"),
        buffer_size=32,
        flush_interval=0.05,
        provider_enabled=False,
    )

    hook = AccessLogHook(access_log_config)
    hook.ingest_service = ingest_service

    total_requests = 200

    async def emit_request(request_index: int) -> None:
        request_id = f"stress-{request_index}"
        path = f"/v1/test/{request_index}"

        start_context = HookContext(
            event=HookEvent.REQUEST_STARTED,
            timestamp=datetime.now(UTC),
            data={
                "request_id": request_id,
                "method": "POST",
                "url": f"https://api.example.com{path}",
                "path": path,
                "client_ip": "127.0.0.1",
                "user_agent": "pytest-agent",
                "service_type": "stress",
                "provider": "duckdb",
                "model": "stress-model",
            },
            metadata={},
        )

        completion_context = HookContext(
            event=HookEvent.REQUEST_COMPLETED,
            timestamp=datetime.now(UTC),
            data={
                "request_id": request_id,
                "status_code": 200,
                "body_size": 512,
                "response_status": 200,
            },
            metadata={
                "tokens_input": request_index % 7,
                "tokens_output": (request_index * 2) % 11,
                "cost_usd": 0.01,
            },
        )

        await hook(start_context)
        await anyio.sleep(0)  # encourage interleaving between tasks
        await hook(completion_context)

    async with anyio.create_task_group() as task_group:
        for index in range(total_requests):
            task_group.start_soon(emit_request, index)

    await hook.close()

    with Session(engine) as session:
        rows = session.exec(select(AccessLog)).all()

    assert len(rows) == total_requests

    sample = rows[0]
    assert sample.service_type == "stress"
    assert sample.provider == "duckdb"
    assert sample.user_agent == "pytest-agent"
