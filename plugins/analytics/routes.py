from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from ccproxy.api.dependencies import DuckDBStorageDep


router = APIRouter(tags=["plugin-analytics"])


@router.get("/query")
async def query_logs(
    storage: DuckDBStorageDep,
    limit: int = Query(1000, ge=1, le=10000, description="Maximum number of results"),
    start_time: float | None = Query(None, description="Start timestamp filter"),
    end_time: float | None = Query(None, description="End timestamp filter"),
    model: str | None = Query(None, description="Model filter"),
    service_type: str | None = Query(None, description="Service type filter"),
) -> dict[str, Any]:
    if not storage:
        raise HTTPException(status_code=503, detail="Storage backend not available")

    if hasattr(storage, "_engine") and storage._engine:
        try:
            from .service import AnalyticsService

            svc = AnalyticsService(storage._engine)
            return svc.query_logs(
                limit=limit,
                start_time=start_time,
                end_time=end_time,
                model=model,
                service_type=service_type,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    raise HTTPException(status_code=503, detail="Storage engine not available")


@router.get("/analytics")
async def get_logs_analytics(
    storage: DuckDBStorageDep,
    start_time: float | None = Query(None, description="Start timestamp (Unix time)"),
    end_time: float | None = Query(None, description="End timestamp (Unix time)"),
    model: str | None = Query(None, description="Filter by model name"),
    service_type: str | None = Query(
        None,
        description="Filter by service type. Supports comma-separated values and !negation",
    ),
    hours: int | None = Query(24, ge=1, le=168, description="Hours of data to analyze"),
) -> dict[str, Any]:
    if not storage:
        raise HTTPException(status_code=503, detail="Storage backend not available")

    # Default window
    if start_time is None and end_time is None and hours:
        end_time = time.time()
        start_time = end_time - (hours * 3600)

    if hasattr(storage, "_engine") and storage._engine:
        try:
            from .service import AnalyticsService

            svc = AnalyticsService(storage._engine)
            analytics = svc.get_analytics(
                start_time=start_time,
                end_time=end_time,
                model=model,
                service_type=service_type,
                hours=hours,
            )
            analytics["query_params"] = {
                "start_time": start_time,
                "end_time": end_time,
                "model": model,
                "service_type": service_type,
                "hours": hours,
            }
            return analytics
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Analytics query failed: {str(e)}")

    raise HTTPException(status_code=503, detail="Storage engine not available")


@router.get("/stream")
async def stream_logs(
    request,
    model: str | None = Query(None, description="Filter by model name"),
    service_type: str | None = Query(None, description="Filter by service type"),
    min_duration_ms: float | None = Query(None, description="Min duration (ms)"),
    max_duration_ms: float | None = Query(None, description="Max duration (ms)"),
    status_code_min: int | None = Query(None, description="Min status code"),
    status_code_max: int | None = Query(None, description="Max status code"),
) -> StreamingResponse:
    async def event_generator():
        from ccproxy.core.request_context import get_request_event_stream

        try:
            async for event in get_request_event_stream():
                data = event
                if model and data.get("model") != model:
                    continue
                if service_type and data.get("service_type") != service_type:
                    continue
                if min_duration_ms and data.get("duration_ms", 0) < min_duration_ms:
                    continue
                if max_duration_ms and data.get("duration_ms", 0) > max_duration_ms:
                    continue
                if status_code_min and data.get("status_code", 0) < status_code_min:
                    continue
                if status_code_max and data.get("status_code", 0) > status_code_max:
                    continue

                yield f"data: {data}\n\n"
        except Exception as e:  # pragma: no cover - stream errors aren't fatal
            yield f"event: error\ndata: {str(e)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
