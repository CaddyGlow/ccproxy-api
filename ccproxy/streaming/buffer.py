"""Streaming buffer service for converting streaming requests to non-streaming responses.

This service handles the pattern where a non-streaming request needs to be converted
internally to a streaming request, buffered, and then returned as a non-streaming response.
"""

import json
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

import httpx
import structlog
from starlette.responses import Response

from ccproxy.core.plugins.hooks import HookEvent, HookManager
from ccproxy.core.plugins.hooks.base import HookContext


if TYPE_CHECKING:
    from ccproxy.core.request_context import RequestContext
    from ccproxy.http.pool import HTTPPoolManager
    from ccproxy.services.handler_config import HandlerConfig
    from ccproxy.services.interfaces import IRequestTracer


logger = structlog.get_logger(__name__)


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
            # Step 1: Transform request to enable streaming
            streaming_body = await self._transform_to_streaming_request(body)

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
                logger.warning(
                    "streaming_request_error_status",
                    status_code=status_code,
                    url=url,
                    error_body=error_body[:500].decode("utf-8", errors="ignore"),
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

        logger.debug(
            "stream_collection_completed",
            total_chunks=total_chunks,
            total_bytes=total_bytes,
            status_code=status_code,
            request_id=request_id,
        )

        # Parse the collected stream using SSE parser if available
        parsed_data = await self._parse_collected_stream(
            chunks=chunks,
            handler_config=handler_config,
            request_context=request_context,
        )

        # Attempt to extract usage tokens from collected SSE and merge into parsed data
        try:
            usage = self._extract_usage_from_chunks(chunks)
            if usage and isinstance(parsed_data, dict):
                # Only inject if missing or zero values
                existing = parsed_data.get("usage") or {}

                def _is_zero(v: Any) -> bool:
                    try:
                        return int(v) == 0
                    except Exception:
                        return False

                if not existing or (
                    _is_zero(existing.get("input_tokens", 0))
                    and _is_zero(existing.get("output_tokens", 0))
                ):
                    parsed_data["usage"] = usage
        except Exception as e:
            logger.debug(
                "usage_extraction_failed",
                error=str(e),
                request_id=getattr(request_context, "request_id", None),
            )

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
            logger.warning("no_chunks_collected_for_parsing")
            return None

        # Combine all chunks into a single string
        full_content = b"".join(chunks).decode("utf-8", errors="replace")

        # Try using the configured SSE parser first
        if handler_config.sse_parser:
            try:
                parsed_data = handler_config.sse_parser(full_content)
                if parsed_data is not None:
                    normalized_data = self._normalize_response_payload(parsed_data)
                    if isinstance(normalized_data, dict):
                        logger.debug(
                            "sse_parser_success",
                            parsed_keys=list(normalized_data.keys()),
                            request_id=getattr(request_context, "request_id", None),
                        )
                        return normalized_data
                    else:
                        logger.warning(
                            "sse_parser_normalized_to_non_dict",
                            type_received=type(normalized_data).__name__,
                            request_id=getattr(request_context, "request_id", None),
                        )
                        return None
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
                normalized_json = self._normalize_response_payload(parsed_json)
                if isinstance(normalized_json, dict):
                    return normalized_json
                else:
                    return {"data": parsed_json}
            else:
                # If it's not a dict, wrap it
                return {"data": parsed_json}
        except json.JSONDecodeError:
            pass

        # Fallback: try to extract from generic SSE format
        try:
            parsed_data = self._extract_from_generic_sse(full_content)
            if parsed_data is not None:
                normalized_data = self._normalize_response_payload(parsed_data)
                if isinstance(normalized_data, dict):
                    logger.debug(
                        "generic_sse_parsing_success",
                        request_id=getattr(request_context, "request_id", None),
                    )
                    return normalized_data
        except Exception as e:
            logger.debug(
                "generic_sse_parsing_failed",
                error=str(e),
                request_id=getattr(request_context, "request_id", None),
            )

        # If all parsing fails, return the raw content as error
        logger.warning(
            "stream_parsing_failed_returning_raw",
            content_preview=full_content[:200],
            request_id=getattr(request_context, "request_id", None),
        )

        return {
            "error": "Failed to parse streaming response",
            "raw_content": full_content[:1000],  # Truncate for safety
        }

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
                normalized_payload = self._normalize_response_payload(response_payload)
                if isinstance(normalized_payload, dict):
                    return normalized_payload

        normalized_data = self._normalize_response_payload(last_json_data)
        if isinstance(normalized_data, dict):
            return normalized_data

        return None

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

    def _normalize_response_payload(self, data: Any) -> Any:
        """Normalize Response API style payloads for downstream adapters.

        Ensures the structure conforms to `ResponseObject` expectations by
        filtering/transforming output items and filling required usage fields.
        """
        if not isinstance(data, dict):
            return data

        target = data
        if "response" in data and isinstance(data["response"], dict):
            target = data["response"]

        outputs = target.get("output")
        normalized_outputs: list[dict[str, Any]] = []
        if isinstance(outputs, list):
            for item in outputs:
                if not isinstance(item, dict):
                    continue

                item_type = item.get("type")
                if item_type == "message":
                    normalized_outputs.append(self._normalize_message_output(item))
                elif item_type == "reasoning":
                    summary = item.get("summary") or []
                    texts: list[str] = []
                    for part in summary:
                        if isinstance(part, dict):
                            text = part.get("text") or ""
                            if text:
                                texts.append(text)
                    if texts:
                        normalized_outputs.append(
                            {
                                "type": "message",
                                "id": item.get("id", "msg_reasoning"),
                                "status": item.get("status", "completed"),
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": " ".join(texts),
                                    }
                                ],
                            }
                        )

        if normalized_outputs:
            target["output"] = normalized_outputs
        elif isinstance(outputs, list) and outputs:
            # Fallback: ensure at least one assistant message exists
            target["output"] = [
                {
                    "type": "message",
                    "id": target.get("id", "msg_unnormalized"),
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "",
                        }
                    ],
                }
            ]

        # Ensure required top-level fields exist
        target.setdefault("object", "response")
        target.setdefault("status", "completed")
        target.setdefault("parallel_tool_calls", False)
        target.setdefault("created_at", int(time.time()))
        target.setdefault("id", data.get("id", target.get("id", "resp-buffered")))
        target.setdefault("model", data.get("model", target.get("model", "")))

        usage = target.get("usage")
        if isinstance(usage, dict):
            if "input_tokens" not in usage:
                usage["input_tokens"] = int(usage.get("prompt_tokens", 0) or 0)
            if "output_tokens" not in usage:
                usage["output_tokens"] = int(usage.get("completion_tokens", 0) or 0)
            usage.setdefault(
                "total_tokens",
                usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            )
            usage.setdefault("input_tokens_details", {"cached_tokens": 0})
            usage.setdefault("output_tokens_details", {"reasoning_tokens": 0})
        else:
            target.setdefault(
                "usage",
                {
                    "input_tokens": 0,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 0,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 0,
                },
            )

        return target

    def _normalize_message_output(self, item: dict[str, Any]) -> dict[str, Any]:
        """Normalize a message output item to Response API expectations."""
        normalized = dict(item)
        normalized["type"] = "message"
        normalized.setdefault("status", "completed")
        normalized.setdefault("role", "assistant")

        content = normalized.get("content")
        if isinstance(content, list):
            fixed_content = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "output_text":
                    text = part.get("text") or ""
                    fixed_content.append({"type": "output_text", "text": text})
                elif isinstance(part, str):
                    fixed_content.append({"type": "output_text", "text": part})
            normalized["content"] = fixed_content or [
                {"type": "output_text", "text": ""}
            ]
        elif isinstance(content, str):
            normalized["content"] = [{"type": "output_text", "text": content}]
        else:
            normalized["content"] = [{"type": "output_text", "text": ""}]

        normalized.setdefault("id", item.get("id", "msg_assistant"))
        return normalized

    async def _build_non_streaming_response(
        self,
        final_data: dict[str, Any] | None,
        status_code: int,
        response_headers: dict[str, str],
        request_context: "RequestContext",
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
            final_data = {"error": "No data could be extracted from streaming response"}
            status_code = status_code if status_code >= 400 else 500

        response_content = json.dumps(final_data).encode("utf-8")

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
