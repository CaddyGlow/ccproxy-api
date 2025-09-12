"""Streaming request handler for SSE and chunked responses."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import httpx
import structlog

from ccproxy.core.plugins.hooks import HookManager
from ccproxy.core.request_context import RequestContext
from ccproxy.streaming.deferred import DeferredStreaming


if TYPE_CHECKING:
    from ccproxy.services.handler_config import HandlerConfig


logger = structlog.get_logger(__name__)


class StreamingHandler:
    """Manages streaming request processing with header preservation and SSE adaptation."""

    def __init__(
        self,
        hook_manager: HookManager | None = None,
    ) -> None:
        """Initialize with hook manager for stream events.

        Args:
            hook_manager: Optional hook manager for emitting stream events
        """
        self.hook_manager = hook_manager

    def should_stream_response(self, headers: dict[str, str]) -> bool:
        """Check Accept header for streaming indicators.

        - Looks for 'text/event-stream' in Accept header
        - Also checks for generic 'stream' indicator
        - Case-insensitive comparison
        """
        # Case-insensitive access for Accept header
        accept_header = ""
        try:
            accept_header = next(
                (v for k, v in headers.items() if k.lower() == "accept"),
                "",
            ).lower()
        except Exception:
            accept_header = headers.get("accept", "").lower()
        return "text/event-stream" in accept_header or "stream" in accept_header

    async def should_stream(
        self, request_body: bytes, handler_config: HandlerConfig
    ) -> bool:
        """Check if request body has stream:true flag.

        - Returns False if provider doesn't support streaming
        - Parses JSON body for 'stream' field
        - Handles parse errors gracefully
        """
        if not handler_config.supports_streaming:
            return False

        try:
            data = json.loads(request_body)
            return data.get("stream", False) is True
        except (json.JSONDecodeError, TypeError):
            return False

    async def handle_streaming_request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes,
        handler_config: HandlerConfig,
        request_context: RequestContext,
        on_headers: Any | None = None,
        client_config: dict[str, Any] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> DeferredStreaming:
        """Create a deferred streaming response that preserves headers.

        This always returns a DeferredStreaming response which:
        - Defers the actual HTTP request until FastAPI sends the response
        - Captures all upstream headers correctly
        - Supports SSE processing through handler_config
        - Provides request tracing and metrics
        """

        # Use provided client or create a short-lived one
        owns_client = False
        if client is None:
            client = httpx.AsyncClient(**(client_config or {}))
            owns_client = True

        # Log that we're creating a deferred response
        logger.debug(
            "streaming_handler_creating_deferred_response",
            url=url,
            method=method,
            has_sse_adapter=bool(handler_config.response_adapter),
            adapter_type=type(handler_config.response_adapter).__name__
            if handler_config.response_adapter
            else None,
        )

        # Return the deferred response with format adapter from handler config
        return DeferredStreaming(
            method=method,
            url=url,
            headers=headers,
            body=body,
            client=client,
            media_type="text/event-stream",
            handler_config=handler_config,  # Contains format adapter if needed
            request_context=request_context,
            hook_manager=self.hook_manager,
            on_headers=on_headers,
            close_client_on_finish=owns_client,
        )
