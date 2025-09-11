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
    from ccproxy.services.adapters.format_registry import FormatAdapterRegistry
    from ccproxy.services.handler_config import HandlerConfig


logger = structlog.get_logger(__name__)


class StreamingHandler:
    """Manages streaming request processing with header preservation and SSE adaptation."""

    def __init__(
        self,
        hook_manager: HookManager | None = None,
        format_registry: FormatAdapterRegistry | None = None,
    ) -> None:
        """Initialize with hook manager for stream events.

        Args:
            hook_manager: Optional hook manager for emitting stream events
            format_registry: Optional format registry for format chain conversion
        """
        self.hook_manager = hook_manager
        self.format_registry = format_registry

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
        format_chain: list[str] | None = None,
        client_config: dict[str, Any] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> DeferredStreaming:
        """Create a deferred streaming response that preserves headers.

        This always returns a DeferredStreaming response which:
        - Defers the actual HTTP request until FastAPI sends the response
        - Captures all upstream headers correctly
        - Supports SSE processing through handler_config
        - Provides request tracing and metrics
        - Supports format chain conversion for streaming responses
        """
        # Apply request format chain if provided
        processed_body = body
        if format_chain and self.format_registry and len(format_chain) > 1:
            processed_body = await self._execute_request_format_chain(
                body, format_chain
            )

        # Create enhanced handler_config with streaming format conversion
        enhanced_config = self._build_streaming_config(handler_config, format_chain)

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
            has_sse_adapter=bool(enhanced_config.response_adapter),
            format_chain=format_chain,
            has_format_registry=self.format_registry is not None,
        )

        # Return the deferred response with format chain support
        return DeferredStreaming(
            method=method,
            url=url,
            headers=headers,
            body=processed_body,  # Use format-converted body
            client=client,
            media_type="text/event-stream",
            handler_config=enhanced_config,  # Enhanced with stream processors
            request_context=request_context,
            hook_manager=self.hook_manager,
            close_client_on_finish=owns_client,
        )

    async def _execute_request_format_chain(
        self, body: bytes, format_chain: list[str]
    ) -> bytes:
        """Execute format conversion chain on request body."""
        import json

        if not self.format_registry:
            logger.debug("format_chain_skipped", reason="no_registry")
            return body

        # Parse body as JSON for format conversion
        try:
            current_data = json.loads(body.decode()) if body else {}
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.debug("format_chain_parse_failed", error=str(e))
            return body

        # Apply format chain transformations
        for i in range(len(format_chain) - 1):
            from_format = format_chain[i]
            to_format = format_chain[i + 1]

            try:
                adapter = self.format_registry.get_if_exists(from_format, to_format)
                if not adapter:
                    logger.debug(
                        "format_chain_adapter_missing",
                        from_format=from_format,
                        to_format=to_format,
                    )
                    continue

                current_data = await adapter.adapt_request(current_data)
                logger.debug(
                    "format_chain_step_completed",
                    from_format=from_format,
                    to_format=to_format,
                    step=i + 1,
                )
            except Exception as e:
                logger.debug(
                    "format_chain_step_failed",
                    from_format=from_format,
                    to_format=to_format,
                    step=i + 1,
                    error=str(e),
                )
                # Don't raise - continue with original format
                break

        # Convert back to bytes
        return json.dumps(current_data).encode()

    def _build_streaming_config(
        self, handler_config: HandlerConfig, format_chain: list[str] | None
    ) -> HandlerConfig:
        """Build enhanced handler config with streaming format conversion."""
        if not format_chain or len(format_chain) < 2:
            return handler_config

        # Create streaming format processor for response conversion
        streaming_processor = self._create_streaming_format_processor(format_chain)

        # Import here to avoid circular imports
        from ccproxy.services.handler_config import HandlerConfig

        # Enhanced handler config with streaming processor
        return HandlerConfig(
            supports_streaming=handler_config.supports_streaming,
            request_transformer=handler_config.request_transformer,
            response_adapter=streaming_processor,  # Use streaming processor
            format_context=handler_config.format_context,
        )

    def _create_streaming_format_processor(self, format_chain: list[str]) -> Any:
        """Create streaming format processor for response conversion using format registry."""
        if not self.format_registry:
            logger.debug("streaming_processor_no_registry")
            return None

        # Get the format adapter from registry for the full chain conversion
        from_format = format_chain[0]
        to_format = format_chain[-1]

        try:
            # Get the format adapter that should handle streaming conversion
            adapter = self.format_registry.get_if_exists(from_format, to_format)
            if not adapter:
                logger.debug(
                    "streaming_format_adapter_missing",
                    from_format=from_format,
                    to_format=to_format,
                )
                return None

            # Check if the adapter has streaming capabilities
            if hasattr(adapter, "create_stream_processor"):
                # Adapter can create a streaming processor
                return adapter.create_stream_processor()
            elif hasattr(adapter, "get_stream_processor"):
                # Adapter provides a streaming processor
                return adapter.get_stream_processor()
            else:
                # Adapter doesn't support streaming - log and return None
                logger.debug(
                    "adapter_no_streaming_support",
                    adapter_type=type(adapter).__name__,
                    from_format=from_format,
                    to_format=to_format,
                )
                return None

        except Exception as e:
            logger.debug(
                "streaming_processor_creation_failed",
                from_format=from_format,
                to_format=to_format,
                error=str(e),
            )
            return None
