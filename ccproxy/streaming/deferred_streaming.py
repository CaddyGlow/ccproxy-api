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

    # Import the specific implementation that has both interfaces
    from plugins.request_tracer.tracer import RequestTracerImpl


logger = structlog.get_logger(__name__)


class DeferredStreaming(Response):
    """Deferred response that starts the stream to get headers and processes SSE."""

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
        request_tracer: "RequestTracerImpl | None" = None,
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

                # Get metrics collector from handler config (provider-specific)
                collector = None
                if self.handler_config and hasattr(
                    self.handler_config, "metrics_collector"
                ):
                    collector = self.handler_config.metrics_collector
                    if not collector:
                        logger.debug(
                            "deferred_streaming_no_metrics_collector",
                            service_type=getattr(
                                self.request_context, "metadata", {}
                            ).get("service_type")
                            if self.request_context
                            else None,
                            request_id=request_id,
                        )

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
                        # Metrics collection happens inside _process_sse_events now
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
                        # Check if response is SSE format based on content-type OR if
                        # it's Codex
                        content_type = response.headers.get("content-type", "").lower()
                        # Codex doesn't send content-type header but uses SSE format
                        is_codex = (
                            self.request_context
                            and self.request_context.metadata.get("service_type")
                            == "codex"
                        )
                        is_sse_format = "text/event-stream" in content_type or is_codex

                        if is_sse_format and collector:
                            # Buffer and parse SSE events for metrics extraction
                            sse_buffer = b""
                            async for chunk in response.aiter_bytes():
                                total_chunks += 1
                                total_bytes += len(chunk)
                                sse_buffer += chunk

                                # Process complete SSE events in buffer
                                while b"\n\n" in sse_buffer:
                                    event_end = sse_buffer.index(b"\n\n") + 2
                                    event_data = sse_buffer[:event_end]
                                    sse_buffer = sse_buffer[event_end:]

                                    # Process the complete SSE event with collector
                                    event_str = event_data.decode(
                                        "utf-8", errors="ignore"
                                    )

                                    # trace: Log SSE event
                                    if total_chunks <= 3:
                                        logger.trace(
                                            "deferred_streaming_sse_event",
                                            event_preview=event_str[:200],
                                            event_number=total_chunks,
                                            request_id=request_id,
                                        )

                                    is_final = collector.process_chunk(event_str)

                                    # trace: Log collector state
                                    logger.trace(
                                        "deferred_streaming_collector_state",
                                        is_final=is_final,
                                        metrics=collector.get_metrics()
                                        if hasattr(collector, "get_metrics")
                                        else None,
                                        request_id=request_id,
                                    )

                                    # Yield the complete event
                                    yield event_data

                                # Trace each chunk if tracer is available
                                if self.request_tracer and request_id:
                                    await self.request_tracer.trace_stream_chunk(
                                        request_id=request_id,
                                        chunk=chunk,
                                        chunk_number=total_chunks,
                                    )

                            # Yield any remaining data in buffer
                            if sse_buffer:
                                yield sse_buffer
                        else:
                            # Stream the raw response without SSE parsing
                            async for chunk in response.aiter_bytes():
                                total_chunks += 1
                                total_bytes += len(chunk)

                                # Process chunk for token usage extraction if collector
                                # available
                                if collector and not is_sse_format:
                                    chunk_str = chunk.decode("utf-8", errors="ignore")

                                    # trace: Log first few chunks to see what we're
                                    # processing
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
                                        metrics=collector.get_metrics()
                                        if hasattr(collector, "get_metrics")
                                        else None,
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
                    usage_metrics = (
                        collector.get_metrics()
                        if collector and hasattr(collector, "get_metrics")
                        else {}
                    )
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

                        # Store usage metrics in request context
                        if hasattr(self.request_context, "metadata"):
                            self.request_context.metadata.update(usage_metrics)

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
        - Optionally process raw chunks with metrics collector
        - Pass entire JSON stream through adapter (maintains state)
        - Serialize adapted chunks back to SSE format
        - Optionally process converted chunks with metrics collector
        """
        # Get metrics collector if available
        collector = None
        if self.handler_config and hasattr(self.handler_config, "metrics_collector"):
            collector = self.handler_config.metrics_collector

        # Create streaming pipeline:
        # 1. Parse raw SSE bytes to JSON chunks, optionally collecting metrics from raw
        # format
        json_stream = self._parse_sse_to_json_stream(response.aiter_bytes(), collector)

        # 2. Pass entire JSON stream through adapter (maintains state)
        adapted_stream = adapter.adapt_stream(json_stream)

        # 3. Serialize adapted chunks back to SSE format, optionally collecting metrics
        # from converted format
        async for sse_bytes in self._serialize_json_to_sse_stream(
            adapted_stream,
            collector,
        ):
            yield sse_bytes

    async def _parse_sse_to_json_stream(
        self, raw_stream: AsyncIterator[bytes], collector: Any = None
    ) -> AsyncIterator[dict[str, Any]]:
        """Parse raw SSE bytes stream into JSON chunks.

        Yields JSON objects extracted from SSE events without buffering
        the entire response. Optionally processes raw chunks with metrics collector
        before parsing.

        Args:
            raw_stream: Raw bytes stream from provider
            collector: Optional metrics collector to process raw provider format
        """
        buffer = b""

        async for chunk in raw_stream:
            # Process raw chunk with collector if available (before any conversion)
            if collector and hasattr(collector, "process_raw_chunk"):
                chunk_str = chunk.decode("utf-8", errors="ignore")
                is_final = collector.process_raw_chunk(chunk_str)

                if self.verbose_streaming:
                    logger.debug(
                        "raw_chunk_metrics_processing",
                        is_final=is_final,
                        metrics=collector.get_metrics()
                        if hasattr(collector, "get_metrics")
                        else None,
                    )

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
        self, json_stream: AsyncIterator[dict[str, Any]], collector: Any = None
    ) -> AsyncGenerator[bytes, None]:
        """Serialize JSON chunks back to SSE format.

        Converts JSON objects to SSE event format:
        data: {json}\\n\\n

        Args:
            json_stream: Stream of JSON objects after format conversion
            collector: Optional metrics collector to process converted format
        """
        async for json_obj in json_stream:
            # Convert to SSE format
            json_str = json.dumps(json_obj, ensure_ascii=False)
            sse_event = f"data: {json_str}\n\n"
            sse_bytes = sse_event.encode("utf-8")

            # Process converted chunk with collector if available
            if collector and hasattr(collector, "process_converted_chunk"):
                is_final = collector.process_converted_chunk(sse_event)

                if self.verbose_streaming:
                    logger.debug(
                        "converted_chunk_metrics_processing",
                        is_final=is_final,
                        metrics=collector.get_metrics()
                        if hasattr(collector, "get_metrics")
                        else None,
                    )

            yield sse_bytes

        # Send final [DONE] event
        yield b"data: [DONE]\n\n"
