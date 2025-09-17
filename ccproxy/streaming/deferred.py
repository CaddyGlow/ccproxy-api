"""Deferred streaming response that preserves headers.

This implementation solves the header timing issue and supports SSE processing.
"""

import contextlib
import json
from collections.abc import AsyncGenerator, AsyncIterator
from datetime import datetime
from typing import TYPE_CHECKING, Any

import httpx
import structlog
from starlette.responses import Response, StreamingResponse

from ccproxy.core.plugins.hooks import HookEvent, HookManager
from ccproxy.core.plugins.hooks.base import HookContext


if TYPE_CHECKING:
    from ccproxy.core.request_context import RequestContext
    from ccproxy.services.handler_config import HandlerConfig


logger = structlog.get_logger(__name__)


class DeferredStreaming(StreamingResponse):
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
        hook_manager: HookManager | None = None,
        close_client_on_finish: bool = False,
        on_headers: Any | None = None,
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
            hook_manager: Optional hook manager for emitting stream events
        """
        # Store attributes first
        self.method = method
        self.url = url
        self.request_headers = headers
        self.body = body
        self.client = client
        self.media_type = media_type
        self.handler_config = handler_config
        self.request_context = request_context
        self.hook_manager = hook_manager
        self._close_client_on_finish = close_client_on_finish
        self.on_headers = on_headers

        # Create an async generator for the streaming content
        async def generate_content() -> AsyncGenerator[bytes, None]:
            # This will be replaced when __call__ is invoked
            yield b""

        # Initialize StreamingResponse with a generator
        super().__init__(content=generate_content(), media_type=media_type)

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

            # Invoke on_headers hook (allows choosing adapter/behavior based on upstream)
            if callable(self.on_headers):
                try:
                    result = self.on_headers(upstream_headers, self.request_context)
                    if hasattr(result, "__await__"):
                        result = await result  # support async
                    # If hook returns a new response adapter, set it
                    if result is not None and self.handler_config is not None:
                        try:
                            # If result is a tuple (adapter, media_type), unpack
                            if isinstance(result, tuple):
                                adapter, media_type = result
                                self.handler_config = type(self.handler_config)(
                                    supports_streaming=self.handler_config.supports_streaming,
                                    request_transformer=self.handler_config.request_transformer,
                                    response_adapter=adapter,
                                    response_transformer=self.handler_config.response_transformer,
                                    preserve_header_case=self.handler_config.preserve_header_case,
                                    sse_parser=self.handler_config.sse_parser,
                                    format_context=self.handler_config.format_context,
                                )
                                if media_type:
                                    self.media_type = media_type
                            else:
                                self.handler_config = type(self.handler_config)(
                                    supports_streaming=self.handler_config.supports_streaming,
                                    request_transformer=self.handler_config.request_transformer,
                                    response_adapter=result,
                                    response_transformer=self.handler_config.response_transformer,
                                    preserve_header_case=self.handler_config.preserve_header_case,
                                    sse_parser=self.handler_config.sse_parser,
                                    format_context=self.handler_config.format_context,
                                )
                        except Exception:
                            # If we can't rebuild dataclass (frozen, etc.), skip updating
                            pass
                except Exception as e:
                    logger.debug(
                        "on_headers_hook_failed",
                        error=str(e),
                        category="streaming_headers",
                    )

            # Store headers in request context
            if self.request_context and hasattr(self.request_context, "metadata"):
                self.request_context.metadata["response_headers"] = upstream_headers

            # Remove hop-by-hop headers
            for key in ["content-length", "transfer-encoding", "connection"]:
                upstream_headers.pop(key, None)

            # Add headers; for errors, preserve provider content-type
            is_error_status = response.status_code >= 400
            content_type_header = (
                response.headers.get("content-type") if is_error_status else None
            )
            final_headers: dict[str, str] = {
                **upstream_headers,
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
                "Content-Type": content_type_header
                or (self.media_type or "text/event-stream"),
            }
            if request_id:
                final_headers["X-Request-ID"] = request_id

            # Create generator for the body
            async def body_generator() -> AsyncGenerator[bytes, None]:
                total_chunks = 0
                total_bytes = 0

                # Emit PROVIDER_STREAM_START hook
                if self.hook_manager:
                    try:
                        # Extract provider from URL or context
                        provider = "unknown"
                        if self.request_context and hasattr(
                            self.request_context, "metadata"
                        ):
                            provider = self.request_context.metadata.get(
                                "service_type", "unknown"
                            )

                        stream_start_context = HookContext(
                            event=HookEvent.PROVIDER_STREAM_START,
                            timestamp=datetime.now(),
                            provider=provider,
                            data={
                                "url": self.url,
                                "method": self.method,
                                "headers": dict(self.request_headers),
                                "request_id": request_id,
                            },
                            metadata={
                                "request_id": request_id,
                            },
                        )
                        await self.hook_manager.emit_with_context(stream_start_context)
                    except Exception as e:
                        logger.debug(
                            "hook_emission_failed",
                            event="PROVIDER_STREAM_START",
                            error=str(e),
                            category="hooks",
                        )

                # Local helper to adapt and emit an error SSE event (single chunk)
                async def _emit_error_sse(
                    error_obj: dict[str, Any],
                ) -> AsyncGenerator[bytes, None]:
                    adapted: dict[str, Any] | None = None
                    try:
                        if self.handler_config and self.handler_config.response_adapter:
                            # For now, skip adapter-based error processing to avoid type issues
                            # Just use the error as-is until we fully resolve adapter interfaces
                            adapted = error_obj
                        else:
                            adapted = error_obj
                    except Exception as e:
                        logger.debug(
                            "streaming_error_adaptation_failed",
                            error=str(e),
                            category="streaming_conversion",
                        )
                        adapted = error_obj

                    async def _single() -> AsyncIterator[dict[str, Any]]:
                        yield adapted or error_obj

                    async for sse_bytes in self._serialize_json_to_sse_stream(
                        _single(), include_done=False
                    ):
                        yield sse_bytes

                try:
                    # Check for error status
                    if response.status_code >= 400:
                        # Forward provider error body as-is (no SSE wrapping)
                        raw_error = await response.aread()
                        yield raw_error
                        return

                    # Stream the response with optional SSE processing
                    if self.handler_config and self.handler_config.response_adapter:
                        logger.debug(
                            "streaming_format_adapter_detected",
                            adapter_type=type(
                                self.handler_config.response_adapter
                            ).__name__,
                            request_id=request_id,
                            url=self.url,
                            category="streaming_conversion",
                        )
                        # Process SSE events with format adaptation
                        async for chunk in self._process_sse_events(
                            response, self.handler_config.response_adapter
                        ):
                            total_chunks += 1
                            total_bytes += len(chunk)

                            # Emit PROVIDER_STREAM_CHUNK hook
                            if self.hook_manager:
                                try:
                                    provider = "unknown"
                                    if self.request_context and hasattr(
                                        self.request_context, "metadata"
                                    ):
                                        provider = self.request_context.metadata.get(
                                            "service_type", "unknown"
                                        )

                                    chunk_context = HookContext(
                                        event=HookEvent.PROVIDER_STREAM_CHUNK,
                                        timestamp=datetime.now(),
                                        provider=provider,
                                        data={
                                            "chunk": chunk,
                                            "chunk_number": total_chunks,
                                            "chunk_size": len(chunk),
                                            "request_id": request_id,
                                        },
                                        metadata={"request_id": request_id},
                                    )
                                    await self.hook_manager.emit_with_context(
                                        chunk_context
                                    )
                                except Exception as e:
                                    logger.trace(
                                        "hook_emission_failed",
                                        event="PROVIDER_STREAM_CHUNK",
                                        error=str(e),
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

                        logger.debug(
                            "streaming_no_format_adapter",
                            content_type=content_type,
                            is_codex=is_codex,
                            is_sse_format=is_sse_format,
                            request_id=request_id,
                            category="streaming_conversion",
                        )

                        if is_sse_format:
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

                                    # Emit PROVIDER_STREAM_CHUNK hook for SSE event
                                    if self.hook_manager:
                                        try:
                                            provider = "unknown"
                                            if self.request_context and hasattr(
                                                self.request_context, "metadata"
                                            ):
                                                provider = (
                                                    self.request_context.metadata.get(
                                                        "service_type", "unknown"
                                                    )
                                                )

                                            chunk_context = HookContext(
                                                event=HookEvent.PROVIDER_STREAM_CHUNK,
                                                timestamp=datetime.now(),
                                                provider=provider,
                                                data={
                                                    "chunk": event_data,
                                                    "chunk_number": total_chunks,
                                                    "chunk_size": len(event_data),
                                                    "request_id": request_id,
                                                },
                                                metadata={"request_id": request_id},
                                            )
                                            await self.hook_manager.emit_with_context(
                                                chunk_context
                                            )
                                        except Exception as e:
                                            logger.trace(
                                                "hook_emission_failed",
                                                event="PROVIDER_STREAM_CHUNK",
                                                error=str(e),
                                            )

                                    # Yield the complete event
                                    yield event_data

                            # Yield any remaining data in buffer
                            if sse_buffer:
                                yield sse_buffer
                        else:
                            # Stream the raw response without SSE parsing
                            async for chunk in response.aiter_bytes():
                                total_chunks += 1
                                total_bytes += len(chunk)

                                # Emit PROVIDER_STREAM_CHUNK hook
                                if self.hook_manager:
                                    try:
                                        provider = "unknown"
                                        if self.request_context and hasattr(
                                            self.request_context, "metadata"
                                        ):
                                            provider = (
                                                self.request_context.metadata.get(
                                                    "service_type", "unknown"
                                                )
                                            )

                                        chunk_context = HookContext(
                                            event=HookEvent.PROVIDER_STREAM_CHUNK,
                                            timestamp=datetime.now(),
                                            provider=provider,
                                            data={
                                                "chunk": chunk,
                                                "chunk_number": total_chunks,
                                                "chunk_size": len(chunk),
                                                "request_id": request_id,
                                            },
                                            metadata={"request_id": request_id},
                                        )
                                        await self.hook_manager.emit_with_context(
                                            chunk_context
                                        )
                                    except Exception as e:
                                        logger.trace(
                                            "hook_emission_failed",
                                            event="PROVIDER_STREAM_CHUNK",
                                            error=str(e),
                                        )

                                yield chunk

                    # Update metrics if available
                    if self.request_context and hasattr(
                        self.request_context, "metrics"
                    ):
                        self.request_context.metrics["stream_chunks"] = total_chunks
                        self.request_context.metrics["stream_bytes"] = total_bytes

                    # Emit PROVIDER_STREAM_END hook
                    if self.hook_manager:
                        try:
                            provider = "unknown"
                            if self.request_context and hasattr(
                                self.request_context, "metadata"
                            ):
                                provider = self.request_context.metadata.get(
                                    "service_type", "unknown"
                                )

                            logger.debug(
                                "emitting_provider_stream_end_hook",
                                request_id=request_id,
                                provider=provider,
                                total_chunks=total_chunks,
                                total_bytes=total_bytes,
                            )

                            stream_end_context = HookContext(
                                event=HookEvent.PROVIDER_STREAM_END,
                                timestamp=datetime.now(),
                                provider=provider,
                                data={
                                    "url": self.url,
                                    "method": self.method,
                                    "request_id": request_id,
                                    "total_chunks": total_chunks,
                                    "total_bytes": total_bytes,
                                },
                                metadata={
                                    "request_id": request_id,
                                },
                            )
                            await self.hook_manager.emit_with_context(
                                stream_end_context
                            )
                            logger.debug(
                                "provider_stream_end_hook_emitted",
                                request_id=request_id,
                            )
                        except Exception as e:
                            logger.error(
                                "hook_emission_failed",
                                event="PROVIDER_STREAM_END",
                                error=str(e),
                                category="hooks",
                                exc_info=e,
                            )
                    else:
                        logger.debug(
                            "no_hook_manager_for_stream_end",
                            request_id=request_id,
                        )

                except httpx.TimeoutException as e:
                    logger.error(
                        "streaming_request_timeout",
                        url=self.url,
                        error=str(e),
                        exc_info=e,
                    )
                    async for error_chunk in _emit_error_sse(
                        {
                            "error": {
                                "type": "timeout_error",
                                "message": "Request timeout",
                            }
                        }
                    ):
                        yield error_chunk
                except httpx.ConnectError as e:
                    logger.error(
                        "streaming_connect_error",
                        url=self.url,
                        error=str(e),
                        exc_info=e,
                    )
                    async for error_chunk in _emit_error_sse(
                        {
                            "error": {
                                "type": "connection_error",
                                "message": "Connection failed",
                            }
                        }
                    ):
                        yield error_chunk
                except httpx.HTTPError as e:
                    logger.error(
                        "streaming_http_error", url=self.url, error=str(e), exc_info=e
                    )
                    async for error_chunk in _emit_error_sse(
                        {
                            "error": {
                                "type": "http_error",
                                "message": f"HTTP error: {str(e)}",
                            }
                        }
                    ):
                        yield error_chunk
                except Exception as e:
                    logger.error(
                        "streaming_request_unexpected_error",
                        url=self.url,
                        error=str(e),
                        exc_info=e,
                    )
                    async for error_chunk in _emit_error_sse(
                        {"error": {"type": "internal_server_error", "message": str(e)}}
                    ):
                        yield error_chunk

            # Create the actual streaming response with headers
            # Access logging now handled by hooks
            actual_response: Response
            if self.request_context:
                actual_response = StreamingResponse(
                    content=body_generator(),
                    status_code=response.status_code,
                    headers=final_headers,
                    media_type=self.media_type,
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

        # After the streaming context closes, optionally close the client we own
        if self._close_client_on_finish:
            with contextlib.suppress(Exception):
                await self.client.aclose()

    async def _process_sse_events(
        self, response: httpx.Response, adapter: Any
    ) -> AsyncGenerator[bytes, None]:
        """Parse and adapt SSE events from response stream.

        - Parse raw SSE bytes to JSON chunks
        - Optionally process raw chunks with metrics collector
        - Pass entire JSON stream through adapter (maintains state)
        - Serialize adapted chunks back to SSE format
        - Optionally process converted chunks with metrics collector
        """
        request_id = None
        if self.request_context and hasattr(self.request_context, "request_id"):
            request_id = self.request_context.request_id

        logger.debug(
            "sse_processing_pipeline_start",
            adapter_type=type(adapter).__name__,
            request_id=request_id,
            response_status=response.status_code,
            category="streaming_conversion",
        )

        # Create streaming pipeline:
        # 1. Parse raw SSE bytes to JSON chunks
        json_stream = self._parse_sse_to_json_stream(response.aiter_bytes())

        # 2. Pass entire JSON stream through adapter (maintains state)
        logger.debug(
            "sse_adapter_stream_calling",
            adapter_type=type(adapter).__name__,
            request_id=request_id,
            category="adapter_integration",
        )

        # Handle both legacy dict-based and new model-based adapters
        if hasattr(adapter, "convert_stream"):
            try:
                adapted_stream = await adapter.convert_stream(json_stream)  # type: ignore[attr-defined]
            except Exception as e:
                logger.warning(
                    "adapter_stream_failed_fallback_to_passthrough",
                    adapter_type=type(adapter).__name__,
                    error=str(e),
                    request_id=request_id,
                )
                adapted_stream = json_stream
        elif hasattr(adapter, "adapt_stream"):
            try:
                adapted_stream = adapter.adapt_stream(json_stream)
            except ValueError as e:
                # Fail fast for missing formatters - don't silently fall back
                if "No stream formatter available" in str(e):
                    logger.error(
                        "streaming_formatter_missing_failing_fast",
                        adapter_type=type(adapter).__name__,
                        error=str(e),
                        request_id=request_id,
                        category="streaming_conversion",
                    )
                    raise e
                else:
                    logger.warning(
                        "adapter_stream_failed_fallback_to_passthrough",
                        adapter_type=type(adapter).__name__,
                        error=str(e),
                        request_id=request_id,
                    )
                    # Fallback to passthrough for other errors
                    adapted_stream = json_stream
            except Exception as e:
                logger.warning(
                    "adapter_stream_failed_fallback_to_passthrough",
                    adapter_type=type(adapter).__name__,
                    error=str(e),
                    request_id=request_id,
                )
                # Fallback to passthrough for unexpected errors
                adapted_stream = json_stream
        else:
            # No adapter, passthrough
            adapted_stream = json_stream

        # 3. Serialize adapted chunks back to SSE format
        chunk_count = 0
        async for sse_bytes in self._serialize_json_to_sse_stream(adapted_stream):
            chunk_count += 1
            yield sse_bytes

        logger.debug(
            "sse_processing_pipeline_complete",
            adapter_type=type(adapter).__name__,
            request_id=request_id,
            total_processed_chunks=chunk_count,
            category="streaming_conversion",
        )

    async def _parse_sse_to_json_stream(
        self, raw_stream: AsyncIterator[bytes]
    ) -> AsyncIterator[dict[str, Any]]:
        """Parse raw SSE bytes stream into JSON chunks.

        Yields JSON objects extracted from SSE events without buffering
        the entire response.

        Args:
            raw_stream: Raw bytes stream from provider
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
                # Capture event type if present
                event_type = None
                for line in event_lines:
                    if line.startswith("event:"):
                        event_type = line[6:].strip()

                if data_lines:
                    data = "".join(data_lines)
                    if data == "[DONE]":
                        continue

                    try:
                        json_obj = json.loads(data)
                        # Preserve event type for downstream adapters (if missing)
                        if (
                            event_type
                            and isinstance(json_obj, dict)
                            and "type" not in json_obj
                        ):
                            json_obj["type"] = event_type
                        yield json_obj
                    except json.JSONDecodeError:
                        continue

    async def _serialize_json_to_sse_stream(
        self, json_stream: AsyncIterator[Any], include_done: bool = True
    ) -> AsyncGenerator[bytes, None]:
        """Serialize JSON chunks back to SSE format.

        Converts JSON objects to appropriate SSE event format:
        - For Anthropic format (has "type" field): event: {type}\ndata: {json}\n\n
        - For OpenAI format: data: {json}\n\n

        Args:
            json_stream: Stream of JSON objects after format conversion
        """
        from ccproxy.llms.streaming import AnthropicSSEFormatter

        formatter = AnthropicSSEFormatter()
        request_id = None
        if self.request_context and hasattr(self.request_context, "request_id"):
            request_id = self.request_context.request_id

        chunk_count = 0
        anthropic_chunks = 0
        openai_chunks = 0

        async for json_obj in json_stream:
            chunk_count += 1

            # Convert model to dict if needed
            if hasattr(json_obj, "model_dump"):
                json_obj = json_obj.model_dump()
            elif not isinstance(json_obj, dict):
                # Skip non-dict, non-model objects
                continue

            # Check if this is Anthropic or Response API style format (has "type" field)
            event_type = json_obj.get("type")
            if event_type:
                anthropic_chunks += 1
                # Use proper Anthropic SSE formatting with event: lines
                if event_type == "ping":
                    sse_event = formatter.format_ping()
                else:
                    sse_event = formatter.format_event(event_type, json_obj)
                sse_bytes = sse_event.encode("utf-8")

                logger.trace(
                    "sse_serialization_anthropic_format",
                    event_type=event_type,
                    chunk_number=chunk_count,
                    request_id=request_id,
                    category="sse_format",
                )
            else:
                openai_chunks += 1
                # Use standard OpenAI format (data: only)
                json_str = json.dumps(json_obj, ensure_ascii=False)
                sse_event = f"data: {json_str}\n\n"
                sse_bytes = sse_event.encode("utf-8")

                logger.trace(
                    "sse_serialization_openai_format",
                    chunk_number=chunk_count,
                    has_choices=bool(json_obj.get("choices")),
                    request_id=request_id,
                    category="sse_format",
                )

            yield sse_bytes

        logger.debug(
            "sse_serialization_complete",
            total_chunks=chunk_count,
            anthropic_chunks=anthropic_chunks,
            openai_chunks=openai_chunks,
            request_id=request_id,
            category="sse_format",
        )

        # Optionally send final [DONE] event (suppress for errors)
        if include_done:
            yield b"data: [DONE]\n\n"
