"""Simple logging middleware for DuckDB storage.

This middleware provides direct async logging to DuckDB without the complexity
of queues or batch processing. Suitable for low-traffic dev environments.
"""

import asyncio
import time
import uuid
from collections.abc import Callable
from contextlib import suppress
from typing import Any, Optional, cast

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from ccproxy.config.settings import get_settings
from ccproxy.observability.storage.duckdb_simple import SimpleDuckDBStorage


logger = structlog.get_logger(__name__)


class DuckDBLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware that logs requests directly to DuckDB."""

    def __init__(self, app: Any):
        super().__init__(app)
        self.settings = get_settings()
        self.storage: SimpleDuckDBStorage | None = None
        self._initialized = False

    async def _ensure_storage(self) -> SimpleDuckDBStorage | None:
        """Lazily initialize storage on first request."""
        if not self._initialized and self.settings.observability.duckdb_enabled:
            try:
                self.storage = SimpleDuckDBStorage(
                    database_path=self.settings.observability.duckdb_path
                )
                await self.storage.initialize()
                self._initialized = True
                logger.info("duckdb_logging_middleware_initialized")
            except Exception as e:
                logger.error("duckdb_middleware_init_failed", error=str(e))
                self._initialized = True  # Don't retry
        return self.storage

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Any]
    ) -> Response:
        """Process request and log to DuckDB."""
        # Get request ID from context if available, otherwise generate
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            # Try to get from request state (set by RequestIDMiddleware)
            request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

        # Capture request start time
        start_time = time.time()

        # Process the request
        response = await call_next(request)

        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000

        # Prepare log data
        log_data = {
            "timestamp": start_time,
            "request_id": request_id,
            "method": request.method,
            "endpoint": str(request.url.path),
            "status": "success" if response.status_code < 400 else "error",
            "response_time": duration_ms / 1000.0,  # Convert to seconds for consistency
            "metadata": {
                "status_code": response.status_code,
                "client_ip": request.client.host if request.client else None,
                "user_agent": request.headers.get("User-Agent"),
                "query_params": str(request.url.query) if request.url.query else None,
            },
        }

        # Extract additional data from response headers if available
        # These headers are set by the proxy service with metrics data
        if hasattr(response, "headers"):
            headers = response.headers
            if "x-model" in headers:
                log_data["model"] = headers["x-model"]
            if "x-service-type" in headers:
                log_data["service_type"] = headers["x-service-type"]
            if "x-tokens-input" in headers:
                with suppress(ValueError):
                    log_data["tokens_input"] = int(headers["x-tokens-input"])
            if "x-tokens-output" in headers:
                with suppress(ValueError):
                    log_data["tokens_output"] = int(headers["x-tokens-output"])
            if "x-cost-usd" in headers:
                with suppress(ValueError):
                    log_data["cost_usd"] = float(headers["x-cost-usd"])

        # Fire and forget async write to DuckDB
        storage = await self._ensure_storage()
        if storage:
            asyncio.create_task(self._write_log(storage, log_data))

        return cast(Response, response)

    async def _write_log(
        self, storage: SimpleDuckDBStorage, log_data: dict[str, Any]
    ) -> None:
        """Write log data to DuckDB asynchronously."""
        try:
            await storage.store_request(log_data)
        except Exception as e:
            # Log error but don't fail the request
            logger.error(
                "duckdb_write_failed",
                error=str(e),
                request_id=log_data.get("request_id"),
            )
