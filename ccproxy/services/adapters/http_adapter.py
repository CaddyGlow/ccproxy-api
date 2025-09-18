import contextlib
import json
from abc import abstractmethod
from typing import Any, Literal, cast
from urllib.parse import urlparse

import httpx
from fastapi import HTTPException, Request
from starlette.responses import JSONResponse, Response, StreamingResponse

from ccproxy.core.logging import get_plugin_logger
from ccproxy.models.provider import ProviderConfig
from ccproxy.services.adapters.base import BaseAdapter
from ccproxy.services.adapters.chain_composer import compose_from_chain
from ccproxy.services.handler_config import HandlerConfig
from ccproxy.streaming import DeferredStreaming
from ccproxy.streaming.handler import StreamingHandler
from ccproxy.utils.headers import extract_request_headers, filter_response_headers


logger = get_plugin_logger()


class BaseHTTPAdapter(BaseAdapter):
    """Simplified HTTP adapter with format chain support."""

    def __init__(
        self,
        config: ProviderConfig,
        auth_manager: Any,
        http_pool_manager: Any,
        streaming_handler: StreamingHandler | None = None,
        **kwargs: Any,
    ) -> None:
        # Call parent constructor to properly initialize config
        super().__init__(config=config, **kwargs)
        self.auth_manager = auth_manager
        self.http_pool_manager = http_pool_manager
        self.streaming_handler = streaming_handler
        self.format_registry = kwargs.get("format_registry")
        self.context = kwargs.get("context")

        logger.debug(
            "base_http_adapter_initialized",
            has_streaming_handler=streaming_handler is not None,
            has_format_registry=self.format_registry is not None,
        )

    async def handle_request(
        self, request: Request
    ) -> Response | StreamingResponse | DeferredStreaming:
        """Handle request with streaming detection and format chain support."""

        # Get context from middleware (already initialized)
        ctx = request.state.context

        # Step 1: Extract request data
        body = await request.body()
        headers = extract_request_headers(request)
        method = request.method
        endpoint = ctx.metadata.get("endpoint", "")

        # Extra debug breadcrumbs to confirm code path and detection inputs
        logger.debug(
            "http_adapter_handle_request_entry",
            endpoint=endpoint,
            method=method,
            content_type=headers.get("content-type"),
            has_streaming_handler=bool(self.streaming_handler),
            category="stream_detection",
        )

        # Step 2: Early streaming detection
        if self.streaming_handler:
            logger.debug(
                "checking_should_stream",
                endpoint=endpoint,
                has_streaming_handler=True,
                content_type=headers.get("content-type"),
                category="stream_detection",
            )
            # Detect streaming via Accept header and/or body flag stream:true
            body_wants_stream = False
            try:
                parsed = json.loads(body.decode()) if body else {}
                body_wants_stream = bool(parsed.get("stream", False))
            except Exception:
                body_wants_stream = False
            header_wants_stream = self.streaming_handler.should_stream_response(headers)
            logger.debug(
                "should_stream_results",
                body_wants_stream=body_wants_stream,
                header_wants_stream=header_wants_stream,
                endpoint=endpoint,
                category="stream_detection",
            )
            if body_wants_stream or header_wants_stream:
                logger.debug(
                    "streaming_request_detected",
                    endpoint=endpoint,
                    detected_via=(
                        "content_type_sse"
                        if header_wants_stream
                        else "body_stream_flag"
                    ),
                    category="stream_detection",
                )
                return await self.handle_streaming(request, endpoint)
            else:
                logger.debug(
                    "not_streaming_request",
                    endpoint=endpoint,
                    category="stream_detection",
                )

        # Step 3: Execute format chain if specified (non-streaming)
        request_payload: dict[str, Any] | None = None
        if ctx.format_chain and len(ctx.format_chain) > 1:
            try:
                request_payload = self._decode_json_body(body, context="request")
            except ValueError as exc:
                logger.error(
                    "format_chain_request_parse_failed",
                    error=str(exc),
                    endpoint=endpoint,
                    category="transform",
                )
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": {
                            "type": "invalid_request_error",
                            "message": "Failed to parse request body for format conversion",
                            "details": str(exc),
                        }
                    },
                )

            try:
                logger.debug(
                    "format_chain_request_about_to_convert",
                    chain=ctx.format_chain,
                    endpoint=endpoint,
                    category="transform",
                )
                request_payload = await self._apply_format_chain(
                    data=request_payload,
                    format_chain=ctx.format_chain,
                    stage="request",
                )
                body = self._encode_json_body(request_payload)
                logger.trace(
                    "format_chain_request_converted",
                    from_format=ctx.format_chain[0],
                    to_format=ctx.format_chain[-1],
                    keys=list(request_payload.keys()),
                    size_bytes=len(body),
                    category="transform",
                )
            except Exception as e:
                logger.error(
                    "format_chain_request_failed",
                    error=str(e),
                    endpoint=endpoint,
                    exc_info=e,
                    category="transform",
                )
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": {
                            "type": "invalid_request_error",
                            "message": "Failed to convert request using format chain",
                            "details": str(e),
                        }
                    },
                )
        # Step 4: Provider-specific preparation
        prepared_body, prepared_headers = await self.prepare_provider_request(
            body, headers, endpoint
        )
        with contextlib.suppress(Exception):
            logger.trace(
                "provider_request_prepared",
                endpoint=endpoint,
                header_keys=list(prepared_headers.keys()),
                body_size=len(prepared_body or b""),
                category="http",
            )

        # Step 5: Execute HTTP request
        target_url = await self.get_target_url(endpoint)
        provider_response = await self._execute_http_request(
            method,
            target_url,
            prepared_headers,
            prepared_body,
        )
        logger.trace(
            "provider_response_received",
            status_code=getattr(provider_response, "status_code", None),
            content_type=getattr(provider_response, "headers", {}).get(
                "content-type", None
            ),
            category="http",
        )

        # Step 6: Provider-specific response processing
        response = await self.process_provider_response(provider_response, endpoint)

        # filter out hop-by-hop headers
        headers = filter_response_headers(dict(provider_response.headers))

        # Step 7: Format the response
        if isinstance(response, StreamingResponse):
            logger.debug("process_provider_response_streaming")
            return await self._convert_streaming_response(
                response, ctx.format_chain, ctx
            )
        elif isinstance(response, Response):
            logger.debug("process_provider_response")
            if ctx.format_chain and len(ctx.format_chain) > 1:
                stage: Literal["response", "error"] = (
                    "error" if provider_response.status_code >= 400 else "response"
                )
                try:
                    payload = self._decode_json_body(
                        cast(bytes, response.body), context=stage
                    )
                except ValueError as exc:
                    logger.error(
                        "format_chain_response_parse_failed",
                        error=str(exc),
                        endpoint=endpoint,
                        stage=stage,
                        category="transform",
                    )
                    return response

                try:
                    payload = await self._apply_format_chain(
                        data=payload,
                        format_chain=ctx.format_chain,
                        stage=stage,
                    )
                    body_bytes = self._encode_json_body(payload)
                    return Response(
                        content=body_bytes,
                        status_code=provider_response.status_code,
                        headers=headers,
                        media_type=provider_response.headers.get(
                            "content-type", "application/json"
                        ),
                    )
                except Exception as e:
                    logger.error(
                        "format_chain_response_failed",
                        error=str(e),
                        endpoint=endpoint,
                        stage=stage,
                        exc_info=e,
                        category="transform",
                    )
                    # Return proper error instead of potentially malformed response
                    return JSONResponse(
                        status_code=500,
                        content={
                            "error": {
                                "type": "internal_server_error",
                                "message": "Failed to convert response format",
                                "details": str(e),
                            }
                        },
                    )
            else:
                logger.debug("format_chain_skipped", reason="no forward chain")
                return response
        else:
            logger.warning(
                "unexpected_provider_response_type", type=type(response).__name__
            )
        return Response(
            content=provider_response.content,
            status_code=provider_response.status_code,
            headers=headers,
            media_type=headers.get("content-type", "application/json"),
        )
        # raise ValueError(
        #     "process_provider_response must return httpx.Response for non-streaming",
        # )

    async def handle_streaming(
        self, request: Request, endpoint: str, **kwargs: Any
    ) -> StreamingResponse | DeferredStreaming:
        """Handle a streaming request using StreamingHandler with format chain support."""

        logger.debug("handle_streaming_called", endpoint=endpoint)

        if not self.streaming_handler:
            logger.error("streaming_handler_missing")
            # Fallback to regular request handling
            response = await self.handle_request(request)
            if isinstance(response, StreamingResponse | DeferredStreaming):
                return response
            else:
                logger.warning("non_streaming_fallback", endpoint=endpoint)
                return response  # type: ignore[return-value]

        # Get context from middleware
        ctx = request.state.context

        # Extract request data
        body = await request.body()
        headers = extract_request_headers(request)

        # Step 1: Execute request-side format chain if specified (streaming)
        if ctx.format_chain and len(ctx.format_chain) > 1:
            try:
                stream_payload = self._decode_json_body(body, context="stream_request")
                stream_payload = await self._apply_format_chain(
                    data=stream_payload,
                    format_chain=ctx.format_chain,
                    stage="request",
                )
                body = self._encode_json_body(stream_payload)
                logger.trace(
                    "format_chain_stream_request_converted",
                    from_format=ctx.format_chain[0],
                    to_format=ctx.format_chain[-1],
                    keys=list(stream_payload.keys()),
                    size_bytes=len(body),
                    category="transform",
                )
            except Exception as e:
                logger.error(
                    "format_chain_stream_request_failed",
                    error=str(e),
                    endpoint=endpoint,
                    exc_info=e,
                    category="transform",
                )
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": {
                            "type": "invalid_request_error",
                            "message": "Failed to convert streaming request using format chain",
                            "details": str(e),
                        }
                    },
                )

        # Step 2: Provider-specific preparation (add auth headers, etc.)
        prepared_body, prepared_headers = await self.prepare_provider_request(
            body, headers, endpoint
        )

        # Get format adapter for streaming if format chain exists
        # Important: Do NOT reverse the chain. Adapters are defined for the
        # declared flow and handle response/streaming internally.
        streaming_format_adapter = None
        if ctx.format_chain and self.format_registry:
            # For streaming responses, we need to reverse the format chain direction
            # Request: client_format → provider_format
            # Stream Response: provider_format → client_format
            from_format = ctx.format_chain[-1]  # provider format (e.g., "anthropic")
            to_format = ctx.format_chain[
                0
            ]  # client format (e.g., "openai.chat_completions")
            streaming_format_adapter = self.format_registry.get_if_exists(
                from_format, to_format
            )

            logger.debug(
                "streaming_adapter_lookup",
                format_chain=ctx.format_chain,
                from_format=from_format,
                to_format=to_format,
                adapter_found=streaming_format_adapter is not None,
                adapter_type=type(streaming_format_adapter).__name__
                if streaming_format_adapter
                else None,
            )

        # Build handler config for streaming with a composed format adapter derived from chain
        # Import here to avoid circular imports
        composed_adapter = (
            compose_from_chain(registry=self.format_registry, chain=ctx.format_chain)
            if self.format_registry and ctx.format_chain
            else streaming_format_adapter
        )

        handler_config = HandlerConfig(
            supports_streaming=True,
            request_transformer=None,
            response_adapter=composed_adapter,  # use composed adapter when available
            format_context=None,
        )

        # Get target URL for proper client pool management
        target_url = await self.get_target_url(endpoint)

        # Get HTTP client from pool manager with base URL for hook integration
        parsed_url = urlparse(target_url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"

        # Delegate to StreamingHandler - no format chain needed since adapter is in config
        return await self.streaming_handler.handle_streaming_request(
            method=request.method,
            url=target_url,
            headers=prepared_headers,  # Use prepared headers with auth
            body=prepared_body,  # Use prepared body
            handler_config=handler_config,
            request_context=ctx,
            client=await self.http_pool_manager.get_client(base_url=base_url),
        )

    async def _convert_streaming_response(
        self, response: StreamingResponse, format_chain: list[str], ctx: Any
    ) -> StreamingResponse:
        """Convert streaming response through reverse format chain."""
        # For now, disable reverse format chain for streaming responses
        # This complex conversion should be handled by the existing format adapter system
        # TODO: Implement proper streaming format conversion
        logger.debug(
            "reverse_streaming_format_chain_disabled",
            reason="complex_sse_parsing_disabled",
            format_chain=format_chain,
        )
        return response

    @abstractmethod
    async def prepare_provider_request(
        self, body: bytes, headers: dict[str, str], endpoint: str
    ) -> tuple[bytes, dict[str, str]]:
        """Provider prepares request. Headers have lowercase keys."""
        pass

    @abstractmethod
    async def process_provider_response(
        self, response: httpx.Response, endpoint: str
    ) -> Response | StreamingResponse:
        """Provider processes response."""
        pass

    @abstractmethod
    async def get_target_url(self, endpoint: str) -> str:
        """Get target URL for this provider."""
        pass

    async def _apply_format_chain(
        self,
        *,
        data: dict[str, Any],
        format_chain: list[str],
        stage: Literal["request", "response", "error"],
    ) -> dict[str, Any]:
        if not self.format_registry:
            raise RuntimeError("Format registry is not configured")

        pairs = self._build_chain_pairs(format_chain, stage)
        current = data
        for step_index, (from_format, to_format) in enumerate(pairs, start=1):
            adapter = self.format_registry.get(from_format, to_format)
            logger.debug(
                "format_chain_step_start",
                from_format=from_format,
                to_format=to_format,
                stage=stage,
                step=step_index,
            )

            if stage == "request":
                current = await adapter.convert_request(current)
            elif stage == "response":
                current = await adapter.convert_response(current)
            elif stage == "error":
                current = await adapter.convert_error(current)
            else:  # pragma: no cover - defensive
                raise ValueError(f"Unsupported format chain stage: {stage}")

            logger.debug(
                "format_chain_step_completed",
                from_format=from_format,
                to_format=to_format,
                stage=stage,
                step=step_index,
            )

        return current

    def _build_chain_pairs(
        self, format_chain: list[str], stage: Literal["request", "response", "error"]
    ) -> list[tuple[str, str]]:
        if len(format_chain) < 2:
            return []

        if stage == "response":
            pairs = [
                (format_chain[i + 1], format_chain[i])
                for i in range(len(format_chain) - 1)
            ]
            pairs.reverse()
            return pairs

        return [
            (format_chain[i], format_chain[i + 1]) for i in range(len(format_chain) - 1)
        ]

    def _decode_json_body(self, body: bytes, *, context: str) -> dict[str, Any]:
        if not body:
            return {}

        try:
            parsed = json.loads(body.decode())
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:  # pragma: no cover
            raise ValueError(f"{context} body is not valid JSON: {exc}") from exc

        if not isinstance(parsed, dict):
            raise ValueError(
                f"{context} body must be a JSON object, got {type(parsed).__name__}"
            )

        return parsed

    def _encode_json_body(self, data: dict[str, Any]) -> bytes:
        try:
            return json.dumps(data).encode()
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError(f"Failed to serialize format chain output: {exc}") from exc

    async def _execute_http_request(
        self, method: str, url: str, headers: dict[str, str], body: bytes
    ) -> httpx.Response:
        """Execute HTTP request."""
        # Convert to canonical headers for HTTP
        canonical_headers = headers

        # Get HTTP client
        client = await self.http_pool_manager.get_client()

        # Execute
        response: httpx.Response = await client.request(
            method=method,
            url=url,
            headers=canonical_headers,
            content=body,
            timeout=120.0,
        )
        return response

    async def cleanup(self) -> None:
        """Cleanup resources."""
        logger.debug("adapter_cleanup_completed")
