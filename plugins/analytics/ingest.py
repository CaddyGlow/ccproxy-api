from __future__ import annotations

import time
from typing import Any


class AnalyticsIngestService:
    """Facade service to ingest access log records into storage.

    This service depends on a storage object that implements `store_request(dict)`.
    """

    def __init__(self, storage: Any):
        self._storage = storage

    async def ingest(self, log_data: dict[str, Any]) -> bool:
        """Normalize and forward log data to storage.

        Args:
            log_data: Access log fields captured by hooks

        Returns:
            True on success, False otherwise
        """
        if not self._storage or not hasattr(self._storage, "store_request"):
            return False

        ts = log_data.get("timestamp", time.time())

        payload: dict[str, Any] = {
            "request_id": log_data.get("request_id", ""),
            "timestamp": ts,
            "method": log_data.get("method", ""),
            # Prefer explicit endpoint then path
            "endpoint": log_data.get("endpoint", log_data.get("path", "")),
            "path": log_data.get("path", ""),
            "query": log_data.get("query", ""),
            "client_ip": log_data.get("client_ip", ""),
            "user_agent": log_data.get("user_agent", ""),
            "service_type": log_data.get("service_type", "access_log"),
            "model": log_data.get("model", ""),
            "streaming": bool(log_data.get("streaming", False)),
            "status_code": int(log_data.get("status_code", 200)),
            "duration_ms": float(log_data.get("duration_ms", 0.0)),
            "duration_seconds": float(log_data.get("duration_ms", 0.0)) / 1000.0,
            "tokens_input": int(log_data.get("tokens_input", 0)),
            "tokens_output": int(log_data.get("tokens_output", 0)),
            "cache_read_tokens": int(log_data.get("cache_read_tokens", 0)),
            "cache_write_tokens": int(log_data.get("cache_write_tokens", 0)),
            "cost_usd": float(log_data.get("cost_usd", 0.0)),
            "cost_sdk_usd": float(log_data.get("cost_sdk_usd", 0.0)),
        }

        try:
            return await self._storage.store_request(payload)
        except Exception:
            return False

