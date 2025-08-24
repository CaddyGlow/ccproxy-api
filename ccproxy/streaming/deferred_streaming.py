"""Deferred streaming response that preserves headers.

This implementation solves the header timing issue and supports SSE processing.
"""

import json
from collections.abc import AsyncGenerator, AsyncIterator
from typing import TYPE_CHECKING, Any

import httpx
import structlog
from starlette.responses import Response, StreamingResponse


if TYPE_CHECKING:
    from ccproxy.adapters.base import APIAdapter
    from ccproxy.observability.context import RequestContext
    from ccproxy.observability.metrics import PrometheusMetrics
    from ccproxy.services.handler_config import HandlerConfig
    from ccproxy.services.tracing import CoreRequestTracer


logger = structlog.get_logger(__name__)


class DeferredStreaming(Response):
    """Deferred response that starts the stream to get headers and optionally processes SSE."""

    def __init__(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes,
        client: httpx.AsyncClient,
        media_type: str = "text/event-stream",
        handler_config: "HandlerConfig | None" = None,
        request_context: "RequestContext | None" = None,
        request_tracer: "CoreRequestTracer | None" = None,
        metrics: "PrometheusMetrics | None" = None,
        verbose_streaming: bool = False,
    ):
        """Store request details to execute later.

        Args:
            method: HTTP method
            url: Target URL
            headers: Request headers
            body: Request body
            client: HTTP client to use
            media_type: Response media type
            handler_config: Optional handler config for SSE processing
            request_context: Optional request context for tracking
            request_tracer: Optional request tracer for verbose logging
            metrics: Optional metrics collector
            verbose_streaming: Enable verbose streaming logs
        """
        super().__init__(content=b"", media_type=media_type)
        self.method = method
        self.url = url
        self.request_headers = headers
        self.body = body
        self.client = client
        self.media_type = media_type
        self.handler_config = handler_config
        self.request_context = request_context
        self.request_tracer = request_tracer
        self.metrics = metrics
        self.verbose_streaming = verbose_streaming

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        """Execute the request when ASGI calls us."""
        # Prepare extensions for request ID tracking
        extensions = {}
        request_id = None
        if self.request_context and hasattr(self.request_context, "request_id"):
            request_id = self.request_context.request_id
            extensions["request_id"] = request_id

        # Start the streaming request
        async with self.client.stream(
            method=self.method,
            url=self.url,
            headers=self.request_headers,
            content=bytes(self.body)
            if isinstance(self.body, memoryview)
            else self.body,
            timeout=httpx.Timeout(300.0),
            extensions=extensions,
        ) as response:
            # Get all headers from upstream
            upstream_headers = dict(response.headers)

            # Store headers in request context
            if self.request_context and hasattr(self.request_context, "metadata"):
                self.request_context.metadata["response_headers"] = upstream_headers

            # Remove hop-by-hop headers
            for key in ["content-length", "transfer-encoding", "connection"]:
                upstream_headers.pop(key, None)

            # Add streaming-specific headers
            final_headers: dict[str, str] = {
                **upstream_headers,
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
                "Content-Type": self.media_type or "text/event-stream",
            }
            if request_id:
                final_headers["X-Request-ID"] = request_id

            # Create generator for the body
            async def body_generator() -> AsyncGenerator[bytes, None]:
                total_chunks = 0
                total_bytes = 0

                # Create metrics collector for token usage extraction
                from ccproxy.utils.streaming_metrics import StreamingMetricsCollector

                collector = StreamingMetricsCollector(request_id=request_id)

                # Trace stream start
                if self.request_tracer and request_id:
                    await self.request_tracer.trace_stream_start(
                        request_id=request_id, headers=self.request_headers
                    )

                try:
                    # Check for error status
                    if response.status_code >= 400:
                        error_body = await response.aread()
                        yield error_body
                        return

                    # Stream the response with optional SSE processing
                    if self.handler_config and self.handler_config.response_adapter:
                        async for chunk in self._process_sse_events(
                            response, self.handler_config.response_adapter
                        ):
                            total_chunks += 1
                            total_bytes += len(chunk)
                            # Trace each chunk if tracer is available
                            if self.request_tracer and request_id:
                                await self.request_tracer.trace_stream_chunk(
                                    request_id=request_id,
                                    chunk=chunk,
                                    chunk_number=total_chunks,
                                )
                            yield chunk
                    else:
                        async for chunk in response.aiter_bytes():
                            total_chunks += 1
                            total_bytes += len(chunk)

                            # Process chunk for token usage extraction
                            chunk_str = chunk.decode("utf-8", errors="ignore")

                            # trace: Log first few chunks to see what we're processing
                            if total_chunks <= 3:
                                logger.trace(
                                    "deferred_streaming_chunk_trace",
                                    chunk_length=len(chunk_str),
                                    chunk_preview=chunk_str[:200],
                                    chunk_number=total_chunks,
                                    request_id=request_id,
                                )

                            is_final = collector.process_chunk(chunk_str)

                            # trace: Log collector state
                            logger.trace(
                                "deferred_streaming_collector_state",
                                is_final=is_final,
                                metrics=collector.get_metrics(),
                                request_id=request_id,
                            )

                            # Trace each chunk if tracer is available
                            if self.request_tracer and request_id:
                                await self.request_tracer.trace_stream_chunk(
                                    request_id=request_id,
                                    chunk=chunk,
                                    chunk_number=total_chunks,
                                )
                            yield chunk

                    # Store final usage metrics in request context
                    usage_metrics = collector.get_metrics()
                    if usage_metrics and self.request_context:
                        # Get model from request context metadata for logging
                        model = None
                        if hasattr(self.request_context, "metadata"):
                            model = self.request_context.metadata.get("model")

                        if model:
                            logger.debug(
                                "deferred_streaming_final_metrics",
                                model=model,
                                usage_metrics=usage_metrics,
                                request_id=request_id,
                                tokens_input=usage_metrics.get("tokens_input"),
                                tokens_output=usage_metrics.get("tokens_output"),
                                cache_read_tokens=usage_metrics.get(
                                    "cache_read_tokens"
                                ),
                                cache_write_tokens=usage_metrics.get(
                                    "cache_write_tokens"
                                ),
                            )

                        # Store usage metrics in request context for provider cost calculation
                        if hasattr(self.request_context, "metadata"):
                            self.request_context.metadata.update(usage_metrics)

                        # Calculate cost if this is a Claude API request (before access logging)
                        await self._calculate_cost_if_claude_api()

                    # Update metrics if available
                    if self.request_context and hasattr(
                        self.request_context, "metrics"
                    ):
                        self.request_context.metrics["stream_chunks"] = total_chunks
                        self.request_context.metrics["stream_bytes"] = total_bytes

                    # Trace stream completion
                    if self.request_tracer and request_id:
                        await self.request_tracer.trace_stream_complete(
                            request_id=request_id,
                            total_chunks=total_chunks,
                            total_bytes=total_bytes,
                        )

                except httpx.TimeoutException as e:
                    logger.error(
                        "streaming_request_timeout",
                        url=self.url,
                        error=str(e),
                        exc_info=e,
                    )
                    error_msg = json.dumps({"error": "Request timeout"}).encode()
                    yield error_msg
                except httpx.ConnectError as e:
                    logger.error(
                        "streaming_connect_error",
                        url=self.url,
                        error=str(e),
                        exc_info=e,
                    )
                    error_msg = json.dumps({"error": "Connection failed"}).encode()
                    yield error_msg
                except httpx.HTTPError as e:
                    logger.error(
                        "streaming_http_error", url=self.url, error=str(e), exc_info=e
                    )
                    error_msg = json.dumps({"error": f"HTTP error: {str(e)}"}).encode()
                    yield error_msg
                except Exception as e:
                    logger.error(
                        "streaming_request_unexpected_error",
                        url=self.url,
                        error=str(e),
                        exc_info=e,
                    )
                    error_msg = json.dumps({"error": str(e)}).encode()
                    yield error_msg

            # Create the actual streaming response with headers
            from ccproxy.observability.streaming_response import (
                StreamingResponseWithLogging,
            )

            # Create response based on whether we have request context
            actual_response: Response
            if self.request_context:
                actual_response = StreamingResponseWithLogging(
                    content=body_generator(),
                    status_code=response.status_code,
                    headers=final_headers,
                    media_type=self.media_type,
                    request_context=self.request_context,
                    metrics=self.metrics,
                )
            else:
                # Use regular StreamingResponse if no request context
                actual_response = StreamingResponse(
                    content=body_generator(),
                    status_code=response.status_code,
                    headers=final_headers,
                    media_type=self.media_type,
                )

            # Delegate to the actual response
            await actual_response(scope, receive, send)

    async def _process_sse_events(
        self, response: httpx.Response, adapter: "APIAdapter"
    ) -> AsyncGenerator[bytes, None]:
        """Parse and adapt SSE events from response stream.

        - Parse raw SSE bytes to JSON chunks
        - Pass entire JSON stream through adapter (maintains state)
        - Serialize adapted chunks back to SSE format
        """
        # Create streaming pipeline:
        # 1. Parse raw SSE bytes to JSON chunks
        json_stream = self._parse_sse_to_json_stream(response.aiter_bytes())

        # 2. Pass entire JSON stream through adapter (maintains state)
        adapted_stream = adapter.adapt_stream(json_stream)

        # 3. Serialize adapted chunks back to SSE format
        async for sse_bytes in self._serialize_json_to_sse_stream(adapted_stream):  # type: ignore[arg-type]
            yield sse_bytes

    async def _parse_sse_to_json_stream(
        self, raw_stream: AsyncIterator[bytes]
    ) -> AsyncIterator[dict[str, Any]]:
        """Parse raw SSE bytes stream into JSON chunks.

        Yields JSON objects extracted from SSE events without buffering
        the entire response.
        """
        buffer = b""

        async for chunk in raw_stream:
            buffer += chunk

            # Process complete SSE events in buffer
            while b"\n\n" in buffer:
                event_end = buffer.index(b"\n\n") + 2
                event_data = buffer[:event_end]
                buffer = buffer[event_end:]

                # Parse SSE event
                event_lines = (
                    event_data.decode("utf-8", errors="ignore").strip().split("\n")
                )
                data_lines = [
                    line[6:] for line in event_lines if line.startswith("data: ")
                ]

                if data_lines:
                    data = "".join(data_lines)
                    if data == "[DONE]":
                        continue

                    try:
                        json_obj = json.loads(data)
                        yield json_obj
                    except json.JSONDecodeError:
                        if self.verbose_streaming:
                            logger.debug("sse_parse_error", data=data)
                        continue

    async def _serialize_json_to_sse_stream(
        self, json_stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncIterator[bytes]:
        """Serialize JSON chunks back to SSE format.

        Converts JSON objects to SSE event format:
        data: {json}\\n\\n
        """
        async for json_obj in json_stream:
            # Convert to SSE format
            json_str = json.dumps(json_obj, ensure_ascii=False)
            sse_event = f"data: {json_str}\n\n"
            yield sse_event.encode("utf-8")

        # Send final [DONE] event
        yield b"data: [DONE]\n\n"

    async def _calculate_cost_if_claude_api(self) -> None:
        """Calculate cost for Claude API requests using pricing service if available."""
        if not self.request_context or not hasattr(self.request_context, "metadata"):
            return

        metadata = self.request_context.metadata
        service_type = metadata.get("service_type")

        # Only calculate cost for Claude API requests
        if service_type != "claude_api":
            return

        model = metadata.get("model")
        tokens_input = metadata.get("tokens_input")
        tokens_output = metadata.get("tokens_output")

        # Skip if we don't have essential data
        if not model or (not tokens_input and not tokens_output):
            logger.debug(
                "deferred_streaming_cost_calculation_skipped",
                reason="missing_model_or_tokens",
                model=model,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                request_id=getattr(self.request_context, "request_id", "unknown"),
            )
            return

        try:
            # Get pricing service from app state via plugin registry
            pricing_service = await self._get_pricing_service()
            if not pricing_service:
                logger.debug(
                    "deferred_streaming_cost_calculation_skipped",
                    reason="pricing_service_not_available",
                    request_id=getattr(self.request_context, "request_id", "unknown"),
                )
                return

            # Calculate cost using the cost calculator utility
            from ccproxy.utils.cost_calculator import calculate_token_cost

            cost_usd = await calculate_token_cost(
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                model=model,
                cache_read_tokens=metadata.get("cache_read_tokens"),
                cache_write_tokens=metadata.get("cache_write_tokens"),
                pricing_service=pricing_service,
            )

            if cost_usd is not None:
                # Update metadata with calculated cost
                metadata["cost_usd"] = cost_usd

                logger.debug(
                    "deferred_streaming_cost_calculated",
                    model=model,
                    cost_usd=cost_usd,
                    tokens_input=tokens_input,
                    tokens_output=tokens_output,
                    cache_read_tokens=metadata.get("cache_read_tokens"),
                    cache_write_tokens=metadata.get("cache_write_tokens"),
                    request_id=getattr(self.request_context, "request_id", "unknown"),
                )
            else:
                logger.debug(
                    "deferred_streaming_cost_calculation_failed",
                    reason="cost_calculator_returned_none",
                    model=model,
                    request_id=getattr(self.request_context, "request_id", "unknown"),
                )

        except Exception as e:
            logger.debug(
                "deferred_streaming_cost_calculation_error",
                error=str(e),
                model=model,
                request_id=getattr(self.request_context, "request_id", "unknown"),
                exc_info=e,
            )

    async def _get_pricing_service(self) -> Any | None:
        """Get pricing service from plugin registry."""
        try:
            # Check if we have a handler config with plugin registry access
            if self.handler_config and hasattr(self.handler_config, "plugin_registry"):
                plugin_registry = self.handler_config.plugin_registry
                pricing_runtime = plugin_registry.get_runtime("pricing")
                if pricing_runtime and hasattr(pricing_runtime, "get_pricing_service"):
                    return pricing_runtime.get_pricing_service()

            return None

        except Exception as e:
            logger.debug(
                "deferred_streaming_pricing_service_access_failed",
                error=str(e),
                request_id=getattr(self.request_context, "request_id", "unknown"),
            )
            return None
