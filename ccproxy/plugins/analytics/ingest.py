from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Any

from sqlmodel import Session

from .models import AccessLog


class AnalyticsIngestService:
    """Ingest access logs directly via SQLModel.

    This service accepts a SQLAlchemy/SQLModel engine and writes AccessLog rows
    without delegating to a storage-specific `store_request` API.
    """

    def __init__(self, engine: Any | None):
        self._engine = engine

    async def ingest(self, log_data: dict[str, Any]) -> bool:
        """Normalize payload and persist using SQLModel.

        Args:
            log_data: Access log fields captured by hooks

        Returns:
            True on success, False otherwise
        """
        if not self._engine:
            return False

        # Normalize timestamp to datetime
        ts_value = log_data.get("timestamp", time.time())
        if isinstance(ts_value, int | float):
            ts_dt = datetime.fromtimestamp(ts_value)
        else:
            ts_dt = ts_value

        # Prefer explicit endpoint then path
        endpoint = log_data.get("endpoint", log_data.get("path", ""))

        # Map incoming dict to AccessLog fields; defaults keep schema stable
        row = AccessLog(
            request_id=str(log_data.get("request_id", "")),
            timestamp=ts_dt,
            method=str(log_data.get("method", "")),
            endpoint=str(endpoint),
            path=str(log_data.get("path", "")),
            query=str(log_data.get("query", "")),
            client_ip=str(log_data.get("client_ip", "")),
            user_agent=str(log_data.get("user_agent", "")),
            service_type=str(log_data.get("service_type", "access_log")),
            provider=str(log_data.get("provider", "")),
            model=str(log_data.get("model", "")),
            streaming=bool(log_data.get("streaming", False)),
            status_code=int(log_data.get("status_code", 200)),
            duration_ms=float(log_data.get("duration_ms", 0.0)),
            duration_seconds=float(
                log_data.get("duration_seconds", log_data.get("duration_ms", 0.0))
            )
            / 1000.0
            if "duration_seconds" not in log_data
            else float(log_data.get("duration_seconds", 0.0)),
            tokens_input=int(log_data.get("tokens_input", 0)),
            tokens_output=int(log_data.get("tokens_output", 0)),
            cache_read_tokens=int(log_data.get("cache_read_tokens", 0)),
            cache_write_tokens=int(log_data.get("cache_write_tokens", 0)),
            cost_usd=float(log_data.get("cost_usd", 0.0)),
            cost_sdk_usd=float(log_data.get("cost_sdk_usd", 0.0)),
        )

        try:
            # Execute the DB write in a thread to avoid blocking the event loop
            return await asyncio.to_thread(self._insert_sync, row)
        except Exception:
            return False

    def _insert_sync(self, row: AccessLog) -> bool:
        with Session(self._engine) as session:
            session.add(row)
            session.commit()
        return True
