"""Streaming request handler for SSE and chunked responses."""

import json
from typing import Any

import httpx
import structlog

from ccproxy.observability.context import RequestContext
from ccproxy.observability.metrics import PrometheusMetrics
from ccproxy.services.handler_config import HandlerConfig
from ccproxy.services.tracing import CoreRequestTracer
from ccproxy.streaming.deferred_streaming import DeferredStreaming
from plugins.pricing.service import PricingService


logger = structlog.get_logger(__name__)


class StreamingHandler:
    """Manages streaming request processing with header preservation and SSE adaptation."""

    def __init__(
        self,
        metrics: PrometheusMetrics | None = None,
        verbose_streaming: bool = False,
        request_tracer: CoreRequestTracer | None = None,
        pricing_service: PricingService | None = None,
    ) -> None:
        """Initialize with metrics collector and debug settings.

        - Sets up Prometheus metrics if provided
        - Configures verbose streaming from environment
        - Optional request tracer for verbose logging
        """
        self.metrics = metrics
        self.verbose_streaming = verbose_streaming
        self.request_tracer = request_tracer
        self.pricing_service = pricing_service

    def should_stream_response(self, headers: dict[str, str]) -> bool:
        """Check Accept header for streaming indicators.

        - Looks for 'text/event-stream' in Accept header
        - Also checks for generic 'stream' indicator
        - Case-insensitive comparison
        """
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
        # Use provided client or create one
        if client is None:
            client = httpx.AsyncClient(**(client_config or {}))

        # Log that we're creating a deferred response
        logger.debug(
            "streaming_handler_creating_deferred_response",
            url=url,
            method=method,
            has_sse_adapter=bool(handler_config.response_adapter),
        )

        # Return the deferred response with all features
        return DeferredStreaming(
            method=method,
            url=url,
            headers=headers,
            body=body,
            client=client,
            media_type="text/event-stream",
            handler_config=handler_config,
            request_context=request_context,
            request_tracer=self.request_tracer,
            metrics=self.metrics,
            verbose_streaming=self.verbose_streaming,
            pricing_service=self.pricing_service,
        )
