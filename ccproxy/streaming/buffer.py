"""Streaming buffer service for converting streaming requests to non-streaming responses.

This service handles the pattern where a non-streaming request needs to be converted
internally to a streaming request, buffered, and then returned as a non-streaming response.
"""

import contextlib
import json
from datetime import datetime
from typing import TYPE_CHECKING, Any

import httpx
import structlog
from pydantic import ValidationError
from starlette.responses import Response

from ccproxy.core.plugins.hooks import HookEvent, HookManager
from ccproxy.core.plugins.hooks.base import HookContext
from ccproxy.llms.models import openai as openai_models
from ccproxy.llms.streaming.accumulators import ResponsesAccumulator, StreamAccumulator


if TYPE_CHECKING:
    from ccproxy.core.request_context import RequestContext
    from ccproxy.http.pool import HTTPPoolManager
    from ccproxy.services.handler_config import HandlerConfig
    from ccproxy.services.interfaces import IRequestTracer


logger = structlog.get_logger(__name__)


MAX_BODY_LOG_CHARS = 2048


def _stringify_payload(payload: Any) -> tuple[str | None, int, bool]:
    """Return a safe preview of request or response payloads."""

    if payload is None:
        return None, 0, False

    try:
        if isinstance(payload, bytes | bytearray | memoryview):
            text = bytes(payload).decode("utf-8", errors="replace")
        elif isinstance(payload, str):
            text = payload
        else:
            text = json.dumps(payload, ensure_ascii=False)
    except Exception:
        text = str(payload)

    length = len(text)
    truncated = length > MAX_BODY_LOG_CHARS
    preview = f"{text[:MAX_BODY_LOG_CHARS]}...[truncated]" if truncated else text
    return preview, length, truncated


class StreamingBufferService:
    """Service for handling stream-to-buffer conversion.

    This service orchestrates the conversion of non-streaming requests to streaming
    requests internally, buffers the entire stream response, and converts it back
    to a non-streaming JSON response while maintaining full observability.
    """

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        request_tracer: "IRequestTracer | None" = None,
        hook_manager: HookManager | None = None,
        http_pool_manager: "HTTPPoolManager | None" = None,
    ) -> None:
        """Initialize the streaming buffer service.

        Args:
            http_client: HTTP client for making requests
            request_tracer: Optional request tracer for observability
            hook_manager: Optional hook manager for event emission
            http_pool_manager: Optional HTTP pool manager for getting clients on demand
        """
        self.http_client = http_client
        self.request_tracer = request_tracer
        self.hook_manager = hook_manager
        self._http_pool_manager = http_pool_manager

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get HTTP client, either existing or from pool manager.

        Returns:
            HTTP client instance
        """
        # If we have a pool manager, get a fresh client from it
        if self._http_pool_manager is not None:
            return await self._http_pool_manager.get_client()

        # Fall back to existing client
        return self.http_client

    async def handle_buffered_streaming_request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes,
        handler_config: "HandlerConfig",
        request_context: "RequestContext",
        provider_name: str = "unknown",
    ) -> Response:
        """Main orchestration method for stream-to-buffer conversion.

        This method:
        1. Transforms the request to enable streaming
        2. Makes a streaming request to the provider
        3. Collects and buffers the entire stream
        4. Parses the buffered stream using SSE parser if available
        5. Returns a non-streaming response with proper headers and observability

        Args:
            method: HTTP method
            url: Target API URL
            headers: Request headers
            body: Request body
            handler_config: Handler configuration with SSE parser and transformers
            request_context: Request context for observability
            provider_name: Name of the provider for hook events

        Returns:
            Non-streaming Response with JSON content

        Raises:
            HTTPException: If streaming fails or parsing fails
        """
        try:
            request_preview, request_size, request_truncated = _stringify_payload(body)
            logger.info(
                "streaming_buffer_request_received",
                provider=provider_name,
                method=method,
                url=url,
                request_id=getattr(request_context, "request_id", None),
                body_preview=request_preview,
                body_size=request_size,
                body_truncated=request_truncated,
                category="streaming",
            )

            # Step 1: Transform request to enable streaming
            streaming_body = await self._transform_to_streaming_request(body)
            transformed_preview, transformed_size, transformed_truncated = (
                _stringify_payload(streaming_body)
            )
            logger.info(
                "streaming_buffer_request_transformed",
                provider=provider_name,
                method=method,
                url=url,
                request_id=getattr(request_context, "request_id", None),
                body_preview=transformed_preview,
                body_size=transformed_size,
                body_truncated=transformed_truncated,
                body_changed=streaming_body != body,
                category="streaming",
            )

            if handler_config.response_adapter:
                logger.info(
                    "streaming_buffer_response_adapter_detected",
                    provider=provider_name,
                    adapter_type=type(handler_config.response_adapter).__name__,
                    request_id=getattr(request_context, "request_id", None),
                    category="format",
                )

            # Step 2: Collect and parse the stream
            (
                final_data,
                status_code,
                response_headers,
            ) = await self._collect_and_parse_stream(
                method=method,
                url=url,
                headers=headers,
                body=streaming_body,
                handler_config=handler_config,
                request_context=request_context,
                provider_name=provider_name,
            )

            # Step 3: Build non-streaming response
            return await self._build_non_streaming_response(
                final_data=final_data,
                status_code=status_code,
                response_headers=response_headers,
                request_context=request_context,
                provider_name=provider_name,
            )

        except Exception as e:
            logger.error(
                "streaming_buffer_service_error",
                method=method,
                url=url,
                error=str(e),
                provider=provider_name,
                request_id=getattr(request_context, "request_id", None),
                exc_info=e,
            )
            # Emit error hook if hook manager is available
            if self.hook_manager:
                try:
                    error_context = HookContext(
                        event=HookEvent.PROVIDER_ERROR,
                        timestamp=datetime.now(),
                        provider=provider_name,
                        data={
                            "url": url,
                            "method": method,
                            "error": str(e),
                            "phase": "streaming_buffer_service",
                        },
                        metadata={
                            "request_id": getattr(request_context, "request_id", None),
                        },
                        error=e,
                    )
                    await self.hook_manager.emit_with_context(error_context)
                except Exception as hook_error:
                    logger.debug(
                        "hook_emission_failed",
                        event="PROVIDER_ERROR",
                        error=str(hook_error),
                        category="hooks",
                    )
            raise

    async def _transform_to_streaming_request(self, body: bytes) -> bytes:
        """Transform request body to enable streaming.

        Adds or modifies the 'stream' flag in the request body to enable streaming.

        Args:
            body: Original request body

        Returns:
            Modified request body with stream=true
        """
        if not body:
            # If no body, create minimal streaming request
            return json.dumps({"stream": True}).encode("utf-8")

        try:
            # Parse existing body
            data = json.loads(body)
        except json.JSONDecodeError:
            logger.warning(
                "failed_to_parse_request_body_for_streaming_transform",
                body_preview=body[:100].decode("utf-8", errors="ignore"),
            )
            # If we can't parse it, wrap it in a streaming request
            return json.dumps({"stream": True}).encode("utf-8")

        # Ensure stream flag is set to True
        if isinstance(data, dict):
            data["stream"] = True
        else:
            # If data is not a dict, wrap it
            data = {"stream": True, "original_data": data}

        return json.dumps(data).encode("utf-8")

    async def _collect_and_parse_stream(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes,
        handler_config: "HandlerConfig",
        request_context: "RequestContext",
        provider_name: str,
    ) -> tuple[dict[str, Any] | None, int, dict[str, str]]:
        """Collect streaming response and parse using SSE parser.

        Makes a streaming request, buffers all chunks, and applies the SSE parser
        from handler config to extract the final JSON response.

        Args:
            method: HTTP method
            url: Target URL
            headers: Request headers
            body: Request body with stream=true
            handler_config: Handler configuration with SSE parser
            request_context: Request context for observability
            provider_name: Provider name for hook events

        Returns:
            Tuple of (parsed_data, status_code, response_headers)
        """
        request_id = getattr(request_context, "request_id", None)

        # Prepare extensions for request ID tracking
        extensions = {}
        if request_id:
            extensions["request_id"] = request_id

        body_preview, body_size, body_truncated = _stringify_payload(body)
        logger.info(
            "streaming_buffer_upstream_request",
            provider=provider_name,
            method=method,
            url=url,
            request_id=request_id,
            body_preview=body_preview,
            body_size=body_size,
            body_truncated=body_truncated,
            category="streaming",
        )

        # Emit PROVIDER_STREAM_START hook
        if self.hook_manager:
            try:
                stream_start_context = HookContext(
                    event=HookEvent.PROVIDER_STREAM_START,
                    timestamp=datetime.now(),
                    provider=provider_name,
                    data={
                        "url": url,
                        "method": method,
                        "headers": dict(headers),
                        "request_id": request_id,
                        "buffered_mode": True,
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

        # Start streaming request and collect all chunks
        chunks: list[bytes] = []
        total_chunks = 0
        total_bytes = 0

        # Get HTTP client from pool manager if available for hook-enabled client
        http_client = await self._get_http_client()

        recent_buffer = bytearray()
        completion_detected = False

        async with http_client.stream(
            method=method,
            url=url,
            headers=headers,
            content=body,
            timeout=httpx.Timeout(300.0),
            extensions=extensions,
        ) as response:
            # Store response info
            status_code = response.status_code
            response_headers = dict(response.headers)

            # If error status, read error body and return it
            if status_code >= 400:
                error_body = await response.aread()
                error_preview, error_size, error_truncated = _stringify_payload(
                    error_body
                )
                logger.error(
                    "streaming_buffer_upstream_error",
                    provider=provider_name,
                    method=method,
                    url=url,
                    status_code=status_code,
                    body_preview=error_preview,
                    body_size=error_size,
                    body_truncated=error_truncated,
                    request_id=request_id,
                    category="streaming",
                )
                try:
                    error_data = json.loads(error_body)
                except json.JSONDecodeError:
                    error_data = {"error": error_body.decode("utf-8", errors="ignore")}
                return error_data, status_code, response_headers

            # Collect all stream chunks
            async for chunk in response.aiter_bytes():
                chunks.append(chunk)
                total_chunks += 1
                total_bytes += len(chunk)
                recent_buffer.extend(chunk)
                if len(recent_buffer) > 8192:
                    del recent_buffer[:-8192]

                # Emit PROVIDER_STREAM_CHUNK hook
                if self.hook_manager:
                    try:
                        chunk_context = HookContext(
                            event=HookEvent.PROVIDER_STREAM_CHUNK,
                            timestamp=datetime.now(),
                            provider=provider_name,
                            data={
                                "chunk": chunk,
                                "chunk_number": total_chunks,
                                "chunk_size": len(chunk),
                                "request_id": request_id,
                                "buffered_mode": True,
                            },
                            metadata={"request_id": request_id},
                        )
                        await self.hook_manager.emit_with_context(chunk_context)
                    except Exception as e:
                        logger.trace(
                            "hook_emission_failed",
                            event="PROVIDER_STREAM_CHUNK",
                            error=str(e),
                        )

                if not completion_detected and (
                    b"response.completed" in recent_buffer
                    or b"response.failed" in recent_buffer
                    or b"response.incomplete" in recent_buffer
                ):
                    completion_detected = True
                    logger.debug(
                        "streaming_buffer_completion_detected",
                        provider=provider_name,
                        request_id=request_id,
                        total_chunks=total_chunks,
                        total_bytes=total_bytes,
                        category="streaming",
                    )
                    break

        logger.info(
            "streaming_buffer_upstream_response",
            provider=provider_name,
            method=method,
            url=url,
            request_id=request_id,
            status_code=status_code,
            total_chunks=total_chunks,
            total_bytes=total_bytes,
            category="streaming",
        )

        # Emit PROVIDER_STREAM_END hook
        if self.hook_manager:
            try:
                stream_end_context = HookContext(
                    event=HookEvent.PROVIDER_STREAM_END,
                    timestamp=datetime.now(),
                    provider=provider_name,
                    data={
                        "url": url,
                        "method": method,
                        "request_id": request_id,
                        "total_chunks": total_chunks,
                        "total_bytes": total_bytes,
                        "buffered_mode": True,
                        "upstream_stream_text": b"".join(chunks).decode(
                            "utf-8", errors="replace"
                        ),
                    },
                    metadata={
                        "request_id": request_id,
                    },
                )
                await self.hook_manager.emit_with_context(stream_end_context)
            except Exception as e:
                logger.error(
                    "hook_emission_failed",
                    event="PROVIDER_STREAM_END",
                    error=str(e),
                    category="hooks",
                    exc_info=e,
                )

        # Update metrics if available
        if hasattr(request_context, "metrics"):
            request_context.metrics["stream_chunks"] = total_chunks
            request_context.metrics["stream_bytes"] = total_bytes

        # Parse the collected stream using SSE parser if available
        parsed_data = await self._parse_collected_stream(
            chunks=chunks,
            handler_config=handler_config,
            request_context=request_context,
        )

        if parsed_data is None:
            raise RuntimeError("Parsed streaming response is empty")

        return parsed_data, status_code, response_headers

    async def _parse_collected_stream(
        self,
        chunks: list[bytes],
        handler_config: "HandlerConfig",
        request_context: "RequestContext",
    ) -> dict[str, Any] | None:
        """Parse collected stream chunks using the configured SSE parser.

        Args:
            chunks: Collected stream chunks
            handler_config: Handler configuration with potential SSE parser
            request_context: Request context for logging

        Returns:
            Parsed final response data or None if parsing fails
        """
        if not chunks:
            logger.error("no_chunks_collected_for_parsing")
            raise RuntimeError("No streaming chunks were collected")

        # Combine all chunks into a single string
        full_content = b"".join(chunks).decode("utf-8", errors="replace")
        content_preview, content_size, content_truncated = _stringify_payload(
            full_content
        )
        logger.debug(
            "streaming_buffer_collected_content",
            request_id=getattr(request_context, "request_id", None),
            content_preview=content_preview,
            content_size=content_size,
            content_truncated=content_truncated,
            category="streaming",
        )

        stream_accumulator: StreamAccumulator | None = None
        accumulator_cls = getattr(request_context, "_tool_accumulator_class", None)
        if callable(accumulator_cls):
            try:
                stream_accumulator = accumulator_cls()
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.debug(
                    "stream_accumulator_init_failed",
                    error=str(exc),
                    request_id=getattr(request_context, "request_id", None),
                )
                stream_accumulator = None

        if stream_accumulator:
            self._accumulate_stream_events(
                full_content, stream_accumulator, request_context
            )

        # Attempt to reconstruct a Responses API payload from the SSE stream
        payloads = self._extract_sse_payloads(full_content)
        base_response: dict[str, Any] | None = None
        reasoning_signature: str | None = None
        for payload in payloads:
            if not isinstance(payload, dict):
                continue
            event_type = payload.get("type")
            if isinstance(event_type, str) and stream_accumulator is not None:
                with contextlib.suppress(Exception):
                    stream_accumulator.accumulate(event_type, payload)
            if event_type == "response.reasoning_summary_part.added":
                part = payload.get("part")
                if isinstance(part, dict):
                    signature = part.get("text") or part.get("signature")
                    if isinstance(signature, str):
                        reasoning_signature = signature
            if isinstance(payload.get("response"), dict):
                base_response = payload["response"]

        if base_response is None and payloads:
            # Fallback to first response created event
            for payload in payloads:
                resp = payload.get("response") if isinstance(payload, dict) else None
                if isinstance(resp, dict):
                    base_response = resp
                    break

        if base_response is not None:
            response_obj = dict(base_response)
            response_obj.setdefault("created_at", 0)
            response_obj.setdefault("status", "completed")
            response_obj.setdefault("model", response_obj.get("model") or "")
            response_obj.setdefault("output", response_obj.get("output") or {})
            response_obj.setdefault(
                "parallel_tool_calls", response_obj.get("parallel_tool_calls", False)
            )

            if reasoning_signature and isinstance(response_obj.get("reasoning"), dict):
                response_obj["reasoning"].setdefault("summary", [])

            accumulator_for_rebuild: ResponsesAccumulator | None = None
            if isinstance(stream_accumulator, ResponsesAccumulator):
                accumulator_for_rebuild = stream_accumulator
            else:
                accumulator_for_rebuild = ResponsesAccumulator()
                for payload in payloads:
                    if not isinstance(payload, dict):
                        continue
                    event_type = payload.get("type")
                    if isinstance(event_type, str):
                        with contextlib.suppress(Exception):
                            accumulator_for_rebuild.accumulate(event_type, payload)

            if accumulator_for_rebuild is not None:
                completed_payload = accumulator_for_rebuild.get_completed_response()
                logger.debug(
                    "streaming_buffer_accumulator_rebuild_attempt",
                    completed=bool(completed_payload),
                )
                if completed_payload is not None:
                    response_obj = completed_payload
                    return response_obj
                try:
                    response_obj = accumulator_for_rebuild.rebuild_response_object(
                        response_obj
                    )
                    logger.info(
                        "streaming_buffer_parser_strategy",
                        strategy="accumulator_rebuild",
                        request_id=getattr(request_context, "request_id", None),
                        category="streaming",
                    )
                    with contextlib.suppress(ValidationError):
                        typed_payload = openai_models.ResponseObject.model_validate(
                            response_obj
                        )
                        logger.debug(
                            "streaming_buffer_rebuilt_response",
                            response=typed_payload.model_dump(),
                            category="streaming",
                            request_id=getattr(request_context, "request_id", None),
                        )
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.debug(
                        "response_rebuild_failed",
                        error=str(exc),
                        request_id=getattr(request_context, "request_id", None),
                    )

            if not response_obj.get("usage"):
                usage = self._extract_usage_from_chunks(chunks)
                if usage:
                    response_obj["usage"] = {
                        "input_tokens": usage.get("input_tokens", 0),
                        "input_tokens_details": {"cached_tokens": 0},
                        "output_tokens": usage.get("output_tokens", 0),
                        "output_tokens_details": {"reasoning_tokens": 0},
                        "total_tokens": usage.get("total_tokens", 0),
                    }

            return response_obj

        # Try using the configured SSE parser first
        logger.debug(
            "parsing_collected_stream",
            content_preview=full_content[:200],
            request_id=getattr(request_context, "request_id", None),
        )

        if handler_config.sse_parser:
            try:
                parsed_data = handler_config.sse_parser(full_content)
                if parsed_data is not None:
                    logger.debug(
                        "sse_parser_success",
                        parsed_type=type(parsed_data).__name__,
                        request_id=getattr(request_context, "request_id", None),
                    )
                    logger.info(
                        "streaming_buffer_parser_strategy",
                        strategy="sse_parser",
                        request_id=getattr(request_context, "request_id", None),
                        category="streaming",
                    )

                    # Rebuild response with stream accumulator if available
                    if stream_accumulator and isinstance(parsed_data, dict):
                        try:
                            parsed_data = stream_accumulator.rebuild_response_object(
                                parsed_data
                            )
                            logger.debug(
                                "response_object_rebuilt",
                                request_id=getattr(request_context, "request_id", None),
                            )
                        except Exception as e:
                            logger.warning(
                                "response_rebuild_failed",
                                error=str(e),
                                request_id=getattr(request_context, "request_id", None),
                                exc_info=e,
                            )

                    return parsed_data
                else:
                    logger.warning(
                        "sse_parser_returned_none",
                        content_preview=full_content[:200],
                        request_id=getattr(request_context, "request_id", None),
                    )
            except Exception as e:
                logger.warning(
                    "sse_parser_failed",
                    error=str(e),
                    content_preview=full_content[:200],
                    request_id=getattr(request_context, "request_id", None),
                )

        # Fallback: try to parse as JSON if it's not SSE format
        try:
            parsed_json = json.loads(full_content.strip())
            if isinstance(parsed_json, dict):
                logger.info(
                    "streaming_buffer_parser_strategy",
                    strategy="direct_json",
                    request_id=getattr(request_context, "request_id", None),
                    category="streaming",
                )
                return parsed_json
            else:
                # If it's not a dict, wrap it
                logger.info(
                    "streaming_buffer_parser_strategy",
                    strategy="direct_json_wrapped",
                    request_id=getattr(request_context, "request_id", None),
                    category="streaming",
                )
                return {"data": parsed_json}
        except json.JSONDecodeError:
            pass

        # Fallback: try to extract from generic SSE format
        try:
            parsed_data = self._extract_from_generic_sse(full_content)
            if parsed_data is not None:
                logger.info(
                    "streaming_buffer_parser_strategy",
                    strategy="generic_sse",
                    request_id=getattr(request_context, "request_id", None),
                    category="streaming",
                )
                return parsed_data
        except Exception as e:
            logger.debug(
                "generic_sse_parsing_failed",
                error=str(e),
                request_id=getattr(request_context, "request_id", None),
            )

        logger.error(
            "stream_parsing_failed",
            content_preview=full_content[:200],
            request_id=getattr(request_context, "request_id", None),
            category="streaming",
        )
        raise RuntimeError("Failed to parse streaming response")

    @staticmethod
    def _extract_sse_payloads(content: str) -> list[dict[str, Any]]:
        """Extract JSON payloads from a raw SSE buffer."""

        payloads: list[dict[str, Any]] = []
        current: list[str] = []
        for line in content.splitlines():
            if line.startswith("data: "):
                current.append(line[6:])
            elif line.strip() == "" and current:
                payload = "".join(current)
                if payload and payload != "[DONE]":
                    with contextlib.suppress(json.JSONDecodeError):
                        payloads.append(json.loads(payload))
                current = []
        if current:
            payload = "".join(current)
            if payload and payload != "[DONE]":
                with contextlib.suppress(json.JSONDecodeError):
                    payloads.append(json.loads(payload))
        return payloads

    def _extract_from_generic_sse(self, content: str) -> dict[str, Any] | None:
        """Extract final JSON from generic SSE format.

        This is a fallback parser that tries to extract JSON from common SSE patterns.

        Args:
            content: Full SSE content

        Returns:
            Extracted JSON data or None if not found
        """
        lines = content.strip().split("\n")
        last_json_data = None

        for line in lines:
            line = line.strip()

            # Look for data lines
            if line.startswith("data: "):
                data_str = line[6:].strip()

                # Skip [DONE] markers
                if data_str == "[DONE]":
                    continue

                try:
                    json_data = json.loads(data_str)
                    # Keep track of the last valid JSON we find
                    last_json_data = json_data
                except json.JSONDecodeError:
                    continue

        if isinstance(last_json_data, dict) and "response" in last_json_data:
            response_payload = last_json_data["response"]
            if isinstance(response_payload, dict):
                return response_payload

        if isinstance(last_json_data, dict):
            return last_json_data

        return None

    @staticmethod
    def _accumulate_stream_events(
        full_content: str,
        accumulator: StreamAccumulator,
        request_context: "RequestContext",
    ) -> None:
        """Feed SSE events from the buffered content into the stream accumulator."""

        events = full_content.split("\n\n")
        for event in events:
            event = event.strip()
            if not event:
                continue

            event_name = ""
            data_lines: list[str] = []
            for raw_line in event.split("\n"):
                line = raw_line.strip()
                if line.startswith("event:"):
                    event_name = line[6:].strip()
                elif line.startswith("data:"):
                    payload = line[5:].lstrip()
                    if payload == "[DONE]":
                        data_lines = []
                        break
                    data_lines.append(payload)

            if not data_lines:
                continue

            try:
                event_data = json.loads("\n".join(data_lines))
            except json.JSONDecodeError:
                continue

            try:
                accumulator.accumulate(event_name, event_data)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.debug(
                    "tool_accumulator_accumulate_failed",
                    error=str(exc),
                    event_name=event_name,
                    request_id=getattr(request_context, "request_id", None),
                )

        try:
            # Store tool calls in request context metadata
            tool_calls = accumulator.get_complete_tool_calls()
            if tool_calls:
                existing = request_context.metadata.get("tool_calls")
                if isinstance(existing, list):
                    existing.extend(tool_calls)
                else:
                    request_context.metadata["tool_calls"] = tool_calls

            # Also store the accumulator itself for potential later use
            request_context.metadata["stream_accumulator"] = accumulator
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug(
                "tool_accumulator_finalize_failed",
                error=str(exc),
                request_id=getattr(request_context, "request_id", None),
            )

    def _extract_usage_from_chunks(self, chunks: list[bytes]) -> dict[str, int] | None:
        """Extract token usage from SSE chunks and normalize to Response API shape.

        Tries to find the last JSON object containing a "usage" field and returns a
        dict with keys: input_tokens, output_tokens, total_tokens.
        """
        last_usage: dict[str, Any] | None = None
        for chunk in chunks:
            try:
                text = chunk.decode("utf-8", errors="ignore")
            except Exception:
                continue
            for part in text.split("\n\n"):
                for line in part.splitlines():
                    line = line.strip()
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        continue
                    try:
                        obj = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    # Accept direct usage at top-level or nested
                    usage_obj = None
                    if isinstance(obj, dict) and "usage" in obj:
                        usage_obj = obj["usage"]
                    elif (
                        isinstance(obj, dict)
                        and "response" in obj
                        and isinstance(obj["response"], dict)
                    ):
                        # Some formats nest usage under response
                        usage_obj = obj["response"].get("usage")
                    if isinstance(usage_obj, dict):
                        last_usage = usage_obj

        if not isinstance(last_usage, dict):
            return None

        # Normalize keys
        input_tokens = None
        output_tokens = None
        total_tokens = None

        if "input_tokens" in last_usage or "output_tokens" in last_usage:
            input_tokens = int(last_usage.get("input_tokens", 0) or 0)
            output_tokens = int(last_usage.get("output_tokens", 0) or 0)
            total_tokens = int(
                last_usage.get("total_tokens", input_tokens + output_tokens)
            )
        elif "prompt_tokens" in last_usage or "completion_tokens" in last_usage:
            # Map OpenAI-style to Response API style
            input_tokens = int(last_usage.get("prompt_tokens", 0) or 0)
            output_tokens = int(last_usage.get("completion_tokens", 0) or 0)
            total_tokens = int(
                last_usage.get("total_tokens", input_tokens + output_tokens)
            )
        else:
            return None

        return {
            "input_tokens": input_tokens or 0,
            "output_tokens": output_tokens or 0,
            "total_tokens": total_tokens
            or ((input_tokens or 0) + (output_tokens or 0)),
        }

    async def _build_non_streaming_response(
        self,
        final_data: dict[str, Any] | None,
        status_code: int,
        response_headers: dict[str, str],
        request_context: "RequestContext",
        provider_name: str,
    ) -> Response:
        """Build the final non-streaming response.

        Creates a standard Response object with the parsed JSON data and appropriate headers.

        Args:
            final_data: Parsed response data
            status_code: HTTP status code from streaming response
            response_headers: Headers from streaming response
            request_context: Request context for request ID

        Returns:
            Non-streaming Response with JSON content
        """
        # Prepare response content
        if final_data is None:
            logger.error(
                "streaming_buffer_empty_final_data",
                provider=provider_name,
                request_id=getattr(request_context, "request_id", None),
                category="streaming",
            )
            raise RuntimeError("No data could be extracted from streaming response")

        response_content = json.dumps(final_data).encode("utf-8")
        response_preview, response_size, response_truncated = _stringify_payload(
            final_data
        )

        # Prepare response headers
        final_headers = {}

        # Copy relevant headers from streaming response
        for key, value in response_headers.items():
            # Skip streaming-specific headers and content-length
            if key.lower() not in {
                "transfer-encoding",
                "connection",
                "cache-control",
                "content-length",
            }:
                final_headers[key] = value

        # Set appropriate headers for JSON response
        # Note: Don't set Content-Length as the response may be wrapped by streaming middleware
        final_headers.update(
            {
                "Content-Type": "application/json",
            }
        )

        # Add request ID if available
        request_id = getattr(request_context, "request_id", None)
        if request_id:
            final_headers["X-Request-ID"] = request_id

        logger.debug(
            "non_streaming_response_built",
            status_code=status_code,
            content_length=len(response_content),
            data_keys=list(final_data.keys()) if isinstance(final_data, dict) else None,
            request_id=request_id,
        )

        logger.info(
            "streaming_buffer_response_ready",
            provider=provider_name,
            status_code=status_code,
            request_id=request_id,
            body_preview=response_preview,
            body_size=response_size,
            body_truncated=response_truncated,
            category="streaming",
        )

        # Create response - Starlette will automatically add Content-Length
        response = Response(
            content=response_content,
            status_code=status_code,
            headers=final_headers,
            media_type="application/json",
        )

        # Explicitly remove content-length header to avoid conflicts with middleware conversion
        # This follows the same pattern as the main branch for streaming response handling
        if "content-length" in response.headers:
            del response.headers["content-length"]
        if "Content-Length" in response.headers:
            del response.headers["Content-Length"]

        return response
