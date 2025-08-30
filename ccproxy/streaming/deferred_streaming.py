"""Deferred streaming response that preserves headers.

This implementation solves the header timing issue and supports SSE processing.
"""

import json
from collections.abc import AsyncGenerator, AsyncIterator
from datetime import datetime
from typing import TYPE_CHECKING, Any

import httpx
import structlog
from starlette.responses import Response, StreamingResponse

from ccproxy.hooks import HookEvent, HookManager
from ccproxy.hooks.base import HookContext


if TYPE_CHECKING:
    from ccproxy.adapters.base import APIAdapter
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

        # Create an async generator for the streaming content
        async def generate_content():
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

                try:
                    # Check for error status
                    if response.status_code >= 400:
                        error_body = await response.aread()
                        yield error_body
                        return

                    # Stream the response with optional SSE processing
                    if self.handler_config and self.handler_config.response_adapter:
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
        # Create streaming pipeline:
        # 1. Parse raw SSE bytes to JSON chunks
        json_stream = self._parse_sse_to_json_stream(response.aiter_bytes())

        # 2. Pass entire JSON stream through adapter (maintains state)
        adapted_stream = adapter.adapt_stream(json_stream)

        # 3. Serialize adapted chunks back to SSE format
        async for sse_bytes in self._serialize_json_to_sse_stream(adapted_stream):
            yield sse_bytes

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

                if data_lines:
                    data = "".join(data_lines)
                    if data == "[DONE]":
                        continue

                    try:
                        json_obj = json.loads(data)
                        yield json_obj
                    except json.JSONDecodeError:
                        continue

    async def _serialize_json_to_sse_stream(
        self, json_stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncGenerator[bytes, None]:
        """Serialize JSON chunks back to SSE format.

        Converts JSON objects to SSE event format:
        data: {json}\\n\\n

        Args:
            json_stream: Stream of JSON objects after format conversion
        """
        async for json_obj in json_stream:
            # Convert to SSE format
            json_str = json.dumps(json_obj, ensure_ascii=False)
            sse_event = f"data: {json_str}\n\n"
            sse_bytes = sse_event.encode("utf-8")
            yield sse_bytes

        # Send final [DONE] event
        yield b"data: [DONE]\n\n"
