"""Request logging hook that captures full request lifecycle."""

import time
from typing import Any

import structlog

from ccproxy.hooks.base import Hook, HookContext
from ccproxy.hooks.events import HookEvent
from ccproxy.observability.access_logger import log_request_access, log_request_start
from ccproxy.observability.context import RequestContext


logger = structlog.get_logger(__name__)


class RequestLoggingHook(Hook):
    """Handles complete request logging lifecycle, replacing middleware functionality."""

    name = "request_logger"
    events = [
        HookEvent.REQUEST_STARTED,
        HookEvent.REQUEST_COMPLETED,
        HookEvent.REQUEST_FAILED,
    ]
    priority = 50  # Default priority for access logging

    def __init__(
        self,
        logger: structlog.BoundLogger | None = None,
        storage: Any = None,
        metrics: Any = None,
    ):
        """Initialize the request logging hook.

        Args:
            logger: Optional structured logger instance
            storage: Optional DuckDB storage instance
            metrics: Optional PrometheusMetrics instance
        """
        self.logger = logger or structlog.get_logger(__name__)
        self.storage = storage
        self.metrics = metrics

    async def __call__(self, context: HookContext) -> None:
        """Handle request logging based on event type.

        Args:
            context: Hook context containing request metadata
        """
        if context.event == HookEvent.REQUEST_STARTED:
            await self._handle_request_start(context)
        elif context.event == HookEvent.REQUEST_COMPLETED:
            await self._handle_request_complete(context)
        elif context.event == HookEvent.REQUEST_FAILED:
            await self._handle_request_failed(context)

    async def _handle_request_start(self, context: HookContext) -> None:
        """Log request start and initialize tracking.

        Args:
            context: Hook context with request data
        """
        request = context.request
        if not request:
            return

        # Extract client info
        client_ip = "unknown"
        if request.client:
            client_ip = request.client.host

        # Extract request info
        method = request.method
        path = str(request.url.path)
        query = str(request.url.query) if request.url.query else None
        user_agent = request.headers.get("user-agent", "unknown")

        # Get request ID from context if available
        request_id = None
        if hasattr(request.state, "request_id"):
            request_id = request.state.request_id
        elif hasattr(request.state, "context") and isinstance(
            request.state.context, RequestContext
        ):
            request_id = request.state.context.request_id

        # Log request start
        log_request_start(
            request_id=request_id or "unknown",
            method=method,
            path=path,
            client_ip=client_ip,
            user_agent=user_agent,
            query=query,
        )

    async def _handle_request_complete(self, context: HookContext) -> None:
        """Log completed request with full metadata.

        Args:
            context: Hook context with request/response data
        """
        # Extract request context if available
        request = context.request
        if not request or not hasattr(request.state, "context"):
            # Fall back to basic logging if no context
            await self._log_basic_access(context, status="completed")
            return

        request_context = request.state.context
        if not isinstance(request_context, RequestContext):
            await self._log_basic_access(context, status="completed")
            return

        # Extract metadata from hook context data
        metadata = context.data or {}

        # Update request context metadata with hook data
        request_context.metadata.update(metadata)

        # Extract additional info
        client_ip = "unknown"
        if request.client:
            client_ip = request.client.host

        user_agent = request.headers.get("user-agent", "unknown")
        method = request.method
        path = str(request.url.path)
        query = str(request.url.query) if request.url.query else None

        # Get status code from response or metadata
        status_code = metadata.get("status", 200)
        if context.response and hasattr(context.response, "status_code"):
            status_code = context.response.status_code

        # Log comprehensive access information
        await log_request_access(
            context=request_context,
            status_code=status_code,
            client_ip=client_ip,
            user_agent=user_agent,
            method=method,
            path=path,
            query=query,
            storage=self.storage,
            metrics=self.metrics,
        )

    async def _handle_request_failed(self, context: HookContext) -> None:
        """Log failed request with error details.

        Args:
            context: Hook context with error information
        """
        # Extract request context if available
        request = context.request
        if not request or not hasattr(request.state, "context"):
            await self._log_basic_access(context, status="failed")
            return

        request_context = request.state.context
        if not isinstance(request_context, RequestContext):
            await self._log_basic_access(context, status="failed")
            return

        # Extract metadata from hook context data
        metadata = context.data or {}

        # Update request context metadata with hook data
        request_context.metadata.update(metadata)

        # Add error information
        if context.error:
            request_context.metadata["error"] = context.error
            request_context.metadata["error_type"] = type(context.error).__name__
            request_context.metadata["error_message"] = str(context.error)

        # Extract additional info
        client_ip = "unknown"
        if request.client:
            client_ip = request.client.host

        user_agent = request.headers.get("user-agent", "unknown")
        method = request.method
        path = str(request.url.path)
        query = str(request.url.query) if request.url.query else None

        # Get status code from metadata (defaults to 500 for errors)
        status_code = metadata.get("status", 500)

        # Log comprehensive access information with error
        await log_request_access(
            context=request_context,
            status_code=status_code,
            client_ip=client_ip,
            user_agent=user_agent,
            method=method,
            path=path,
            query=query,
            error_message=str(context.error) if context.error else None,
            storage=self.storage,
            metrics=self.metrics,
        )

    async def _log_basic_access(
        self, context: HookContext, status: str = "unknown"
    ) -> None:
        """Log basic access information when full context is not available.

        Args:
            context: Hook context
            status: Request status (completed/failed)
        """
        metadata = context.data or {}
        log_data = {
            "event_type": f"request_{status}",
            "timestamp": time.time(),
            **metadata,
        }

        # Remove None values
        log_data = {k: v for k, v in log_data.items() if v is not None}

        if status == "failed":
            if context.error:
                log_data["error"] = str(context.error)
                log_data["error_type"] = type(context.error).__name__
            self.logger.error("access_log", **log_data)
        else:
            self.logger.info("access_log", **log_data)
