"""Claude SDK adapter implementation using delegation pattern."""

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, cast

import httpx
from fastapi import HTTPException, Request
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response, StreamingResponse

from ccproxy.config.utils import OPENAI_CHAT_COMPLETIONS_PATH
from ccproxy.core.logging import get_plugin_logger
from ccproxy.core.request_context import RequestContext
from ccproxy.llms.streaming import OpenAIStreamProcessor
from ccproxy.services.adapters.chain_composer import compose_from_chain
from ccproxy.services.adapters.format_adapter import FormatAdapterProtocol
from ccproxy.services.adapters.http_adapter import BaseHTTPAdapter
from ccproxy.streaming import DeferredStreaming
from ccproxy.streaming.sse import serialize_json_to_sse_stream


if TYPE_CHECKING:
    from ccproxy.services.interfaces import IMetricsCollector

from .auth import NoOpAuthManager
from .config import ClaudeSDKSettings
from .handler import ClaudeSDKHandler
from .manager import SessionManager
from .models import MessageResponse


logger = get_plugin_logger()


class ClaudeSDKAdapter(BaseHTTPAdapter):
    """Claude SDK adapter implementation using delegation pattern.

    This adapter integrates with the application request lifecycle,
    following the same pattern as claude_api and codex plugins.
    """

    def __init__(
        self,
        config: ClaudeSDKSettings,
        # Optional dependencies
        session_manager: SessionManager | None = None,
        metrics: "IMetricsCollector | None" = None,
        hook_manager: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Claude SDK adapter with explicit dependencies.

        Args:
            config: SDK configuration settings
            session_manager: Optional session manager for session handling
            metrics: Optional metrics collector
            hook_manager: Optional hook manager for emitting events
        """
        # Initialize BaseHTTPAdapter with dummy auth_manager and http_pool_manager
        # since ClaudeSDK doesn't use external HTTP
        super().__init__(
            config=config, auth_manager=None, http_pool_manager=None, **kwargs
        )
        self.metrics = metrics
        self.hook_manager = hook_manager

        # Generate or set default session ID
        self._runtime_default_session_id = None
        if (
            config.auto_generate_default_session
            and config.sdk_session_pool
            and config.sdk_session_pool.enabled
        ):
            # Generate a random session ID for this runtime
            self._runtime_default_session_id = f"auto-{uuid.uuid4().hex[:12]}"
            logger.debug(
                "auto_generated_session",
                session_id=self._runtime_default_session_id,
                lifetime="runtime",
            )
        elif config.default_session_id:
            self._runtime_default_session_id = config.default_session_id
            logger.debug(
                "using_configured_default_session",
                session_id=self._runtime_default_session_id,
            )

        # Use provided session_manager or create if needed and enabled
        if (
            session_manager is None
            and config.sdk_session_pool
            and config.sdk_session_pool.enabled
        ):
            session_manager = SessionManager(config=config)
            logger.debug(
                "adapter_session_pool_enabled",
                session_ttl=config.sdk_session_pool.session_ttl,
                max_sessions=config.sdk_session_pool.max_sessions,
                has_default_session=bool(self._runtime_default_session_id),
                auto_generated=config.auto_generate_default_session,
            )

        self.session_manager = session_manager
        self.handler: ClaudeSDKHandler | None = ClaudeSDKHandler(
            config=config,
            session_manager=session_manager,
            hook_manager=hook_manager,
        )
        self.auth_manager = NoOpAuthManager()
        self._detection_service: Any | None = None
        self._initialized = False
        self._format_adapter_cache: dict[tuple[str, ...], FormatAdapterProtocol] = {}

    async def initialize(self) -> None:
        """Initialize the adapter and start session manager if needed."""
        if not self._initialized:
            if self.session_manager:
                await self.session_manager.start()
                logger.debug("session_manager_started")
            self._initialized = True

    def set_detection_service(self, detection_service: Any) -> None:
        """Set the detection service.

        Args:
            detection_service: Claude CLI detection service
        """
        self._detection_service = detection_service

    def _resolve_format_adapter(
        self, format_chain: list[str]
    ) -> FormatAdapterProtocol | None:
        """Return a composed format adapter for the provided chain."""

        if not self.format_registry or len(format_chain) < 2:
            return None

        key = tuple(format_chain)
        adapter = self._format_adapter_cache.get(key)
        if adapter is not None:
            return adapter

        adapter = compose_from_chain(
            registry=self.format_registry,
            chain=format_chain,
            name=f"claude_sdk_adapter_{'__'.join(format_chain)}",
        )
        self._format_adapter_cache[key] = adapter
        return adapter

    @staticmethod
    async def _single_payload_stream(
        payload: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        yield payload

    async def handle_request(
        self, request: Request
    ) -> Response | StreamingResponse | DeferredStreaming:
        # Ensure adapter is initialized
        await self.initialize()

        # Extract endpoint from request URL
        endpoint = request.url.path
        method = request.method

        # Parse request body
        body = await request.body()
        if not body:
            raise HTTPException(status_code=400, detail="Request body is required")

        try:
            request_data = json.loads(body)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid JSON: {str(e)}"
            ) from e

        request_context: RequestContext | None = RequestContext.get_current()
        if not request_context:
            raise HTTPException(
                status_code=500,
                detail=(
                    "RequestContext not available - plugin must be invoked within the "
                    "application request lifecycle"
                ),
            )

        self._ensure_tool_accumulator(request_context)

        format_chain = list(getattr(request_context, "format_chain", []) or [])
        try:
            format_adapter = self._resolve_format_adapter(format_chain)
        except Exception as exc:  # pragma: no cover - defensive logging in production
            logger.error(
                "format_adapter_resolution_failed",
                error=str(exc),
                format_chain=format_chain,
                endpoint=endpoint,
                category="format",
                exc_info=exc,
            )
            raise HTTPException(
                status_code=500,
                detail="Failed to prepare format adapter for Claude SDK request",
            ) from exc

        if format_adapter:
            try:
                request_data = await format_adapter.convert_request(request_data)
            except Exception as exc:
                logger.error(
                    "format_request_conversion_failed",
                    error=str(exc),
                    format_chain=format_chain,
                    endpoint=endpoint,
                    category="format",
                    exc_info=exc,
                )
                raise HTTPException(
                    status_code=400,
                    detail="Failed to convert request payload for Claude SDK",
                ) from exc

        # Check if format conversion is needed (OpenAI to Anthropic)
        # The endpoint will contain the path after the prefix, e.g., "/v1/chat/completions"
        needs_conversion = bool(format_adapter) or endpoint.endswith(
            OPENAI_CHAT_COMPLETIONS_PATH
        )

        # Extract parameters for SDK handler
        messages = request_data.get("messages", [])
        model = request_data.get("model", "claude-3-opus-20240229")
        temperature = request_data.get("temperature")
        max_tokens = request_data.get("max_tokens")
        stream = request_data.get("stream", False)

        # Get session_id from multiple sources (in priority order):
        # 1. URL path (stored in request.state by the route handler)
        # 2. Query parameters
        # 3. Request body
        # 4. Default from config (if session pool is enabled)
        session_id = getattr(request.state, "session_id", None)
        source = "path" if session_id else None

        if not session_id and request.query_params:
            session_id = request.query_params.get("session_id")
            source = "query" if session_id else None

        if not session_id:
            session_id = request_data.get("session_id")
            source = "body" if session_id else None

        if (
            not session_id
            and self._runtime_default_session_id
            and self.config.sdk_session_pool
            and self.config.sdk_session_pool.enabled
        ):
            # Use runtime default session_id (either configured or auto-generated)
            session_id = self._runtime_default_session_id
            source = (
                "default"
                if not self.config.auto_generate_default_session
                else "auto-generated"
            )

        # Log session_id source for debugging
        if session_id:
            logger.debug(
                "session_id_extracted",
                session_id=session_id,
                source=source,
                has_default_configured=bool(self.config.default_session_id),
                auto_generate_enabled=self.config.auto_generate_default_session,
                runtime_default=self._runtime_default_session_id,
                session_pool_enabled=bool(
                    self.config.sdk_session_pool
                    and self.config.sdk_session_pool.enabled
                ),
            )

        # Update context with claude_sdk specific metadata
        request_context.metadata.update(
            {
                "provider": "claude_sdk",
                "service_type": "claude_sdk",
                "endpoint": endpoint.rstrip("/").split("/")[-1]
                if endpoint
                else "messages",
                "model": model,
                "stream": stream,
            }
        )

        logger.info(
            "plugin_request",
            plugin="claude_sdk",
            endpoint=endpoint,
            model=model,
            is_streaming=stream,
            needs_conversion=needs_conversion,
            session_id=session_id,
            target_url=f"claude-sdk://{session_id}"
            if session_id
            else "claude-sdk://direct",
        )

        try:
            # Call handler directly to create completion
            if not self.handler:
                raise HTTPException(status_code=503, detail="Handler not initialized")

            result = await self.handler.create_completion(
                request_context=request_context,
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                session_id=session_id,
                **{
                    k: v
                    for k, v in request_data.items()
                    if k
                    not in [
                        "messages",
                        "model",
                        "temperature",
                        "max_tokens",
                        "stream",
                        "session_id",
                    ]
                },
            )

            if stream:
                # Return streaming response
                stream_result = cast(AsyncIterator[dict[str, Any]], result)

                if format_adapter:
                    logger.debug(
                        "format_stream_adapter_applied",
                        format_chain=format_chain,
                        endpoint=endpoint,
                        category="format",
                    )
                    try:
                        converted_stream = format_adapter.convert_stream(stream_result)
                    except Exception as exc:
                        logger.error(
                            "format_stream_conversion_failed",
                            error=str(exc),
                            format_chain=format_chain,
                            endpoint=endpoint,
                            category="format",
                            exc_info=exc,
                        )
                        raise HTTPException(
                            status_code=500,
                            detail="Failed to convert Claude SDK streaming payload",
                        ) from exc

                    async def adapted_stream_generator() -> AsyncIterator[bytes]:
                        """Generate SSE stream from converted OpenAI-style chunks."""

                        try:
                            async for sse_chunk in serialize_json_to_sse_stream(
                                converted_stream,
                                include_done=bool(
                                    format_chain
                                    and format_chain[0].startswith("openai.")
                                ),
                                request_context=request_context,
                            ):
                                yield sse_chunk
                        except asyncio.CancelledError as exc:
                            logger.warning(
                                "streaming_cancelled",
                                error=str(exc),
                                exc_info=exc,
                                category="streaming",
                            )
                            raise
                        except httpx.TimeoutException as exc:
                            logger.error(
                                "streaming_timeout",
                                error=str(exc),
                                exc_info=exc,
                                category="streaming",
                            )
                            error_stream = serialize_json_to_sse_stream(
                                self._single_payload_stream(
                                    {"error": "Request timed out"}
                                ),
                                include_done=False,
                                request_context=request_context,
                            )
                            async for error_chunk in error_stream:
                                yield error_chunk
                        except httpx.HTTPError as exc:
                            logger.error(
                                "streaming_http_error",
                                error=str(exc),
                                status_code=getattr(exc.response, "status_code", None)
                                if hasattr(exc, "response")
                                else None,
                                exc_info=exc,
                                category="streaming",
                            )
                            error_stream = serialize_json_to_sse_stream(
                                self._single_payload_stream(
                                    {"error": f"HTTP error: {exc}"}
                                ),
                                include_done=False,
                                request_context=request_context,
                            )
                            async for error_chunk in error_stream:
                                yield error_chunk
                        except Exception as exc:  # pragma: no cover - defensive
                            logger.error(
                                "streaming_unexpected_error",
                                error=str(exc),
                                exc_info=exc,
                                category="streaming",
                            )
                            error_stream = serialize_json_to_sse_stream(
                                self._single_payload_stream({"error": str(exc)}),
                                include_done=False,
                                request_context=request_context,
                            )
                            async for error_chunk in error_stream:
                                yield error_chunk

                    return StreamingResponse(
                        content=adapted_stream_generator(),
                        media_type="text/event-stream",
                        headers={
                            "Cache-Control": "no-cache",
                            "Connection": "keep-alive",
                            "X-Claude-SDK-Response": "true",
                        },
                    )

                logger.debug(
                    "format_stream_adapter_not_used",
                    reason="no_format_adapter" if not format_adapter else "fallback",
                    format_chain=format_chain,
                    endpoint=endpoint,
                    category="format",
                )

                async def stream_generator() -> AsyncIterator[bytes]:
                    """Handle passthrough or OpenAI-format streaming."""

                    try:
                        if needs_conversion:
                            processor = OpenAIStreamProcessor(
                                model=model,
                                enable_usage=True,
                                enable_tool_calls=True,
                                output_format="sse",
                            )

                            async for sse_chunk in processor.process_stream(
                                stream_result
                            ):
                                if isinstance(sse_chunk, bytes):
                                    yield sse_chunk
                                else:
                                    yield str(sse_chunk).encode()
                        else:
                            async for chunk in serialize_json_to_sse_stream(
                                stream_result,
                                include_done=True,
                                request_context=request_context,
                            ):
                                if isinstance(chunk, bytes):
                                    yield chunk
                                else:
                                    yield str(chunk).encode()
                    except asyncio.CancelledError as exc:
                        logger.warning(
                            "streaming_cancelled",
                            error=str(exc),
                            exc_info=exc,
                            category="streaming",
                        )
                        raise
                    except httpx.TimeoutException as exc:
                        logger.error(
                            "streaming_timeout",
                            error=str(exc),
                            exc_info=exc,
                            category="streaming",
                        )
                        async for error_chunk in serialize_json_to_sse_stream(
                            self._single_payload_stream({"error": "Request timed out"}),
                            include_done=False,
                            request_context=request_context,
                        ):
                            yield error_chunk
                    except httpx.HTTPError as exc:
                        logger.error(
                            "streaming_http_error",
                            error=str(exc),
                            status_code=getattr(exc.response, "status_code", None)
                            if hasattr(exc, "response")
                            else None,
                            exc_info=exc,
                            category="streaming",
                        )
                        async for error_chunk in serialize_json_to_sse_stream(
                            self._single_payload_stream(
                                {"error": f"HTTP error: {exc}"}
                            ),
                            include_done=False,
                            request_context=request_context,
                        ):
                            yield error_chunk
                    except Exception as exc:
                        logger.error(
                            "streaming_unexpected_error",
                            error=str(exc),
                            exc_info=exc,
                            category="streaming",
                        )
                        async for error_chunk in serialize_json_to_sse_stream(
                            self._single_payload_stream({"error": str(exc)}),
                            include_done=False,
                            request_context=request_context,
                        ):
                            yield error_chunk

                return StreamingResponse(
                    content=stream_generator(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Claude-SDK-Response": "true",
                    },
                )
            else:
                # Convert MessageResponse to dict for JSON response
                if isinstance(result, MessageResponse):
                    response_data = result.model_dump()
                else:
                    # This shouldn't happen when stream=False, but handle it
                    response_data = cast(dict[str, Any], result)

                # Convert to OpenAI format if needed
                if format_adapter:
                    try:
                        response_data = await format_adapter.convert_response(
                            response_data
                        )
                    except Exception as exc:
                        logger.error(
                            "format_response_conversion_failed",
                            error=str(exc),
                            format_chain=format_chain,
                            endpoint=endpoint,
                            category="format",
                            exc_info=exc,
                        )
                        raise HTTPException(
                            status_code=500,
                            detail="Failed to convert Claude SDK response payload",
                        ) from exc

                return Response(
                    content=json.dumps(response_data),
                    media_type="application/json",
                    headers={
                        "X-Claude-SDK-Response": "true",
                    },
                )

        except httpx.TimeoutException as e:
            logger.error(
                "request_timeout",
                error=str(e),
                exc_info=e,
                category="http",
            )
            raise HTTPException(status_code=408, detail="Request timed out") from e
        except httpx.HTTPError as e:
            logger.error(
                "http_error",
                error=str(e),
                status_code=getattr(e.response, "status_code", None)
                if hasattr(e, "response")
                else None,
                exc_info=e,
                category="http",
            )
            raise HTTPException(status_code=502, detail=f"HTTP error: {e}") from e
        except asyncio.CancelledError as e:
            logger.warning(
                "request_cancelled",
                error=str(e),
                exc_info=e,
            )
            raise
        except Exception as e:
            logger.error(
                "request_handling_failed",
                error=str(e),
                exc_info=e,
            )
            raise HTTPException(
                status_code=500, detail=f"SDK request failed: {str(e)}"
            ) from e

    async def handle_streaming(
        self, request: Request, endpoint: str, **kwargs: Any
    ) -> StreamingResponse:
        """Handle a streaming request through Claude SDK.

        This is a convenience method that ensures stream=true and delegates
        to handle_request which handles both streaming and non-streaming.

        Args:
            request: FastAPI request object
            endpoint: Target endpoint path
            **kwargs: Additional arguments

        Returns:
            Streaming response from Claude SDK
        """
        if not self._initialized:
            await self.initialize()

        # Parse and modify request to ensure stream=true
        body = await request.body()
        if not body:
            request_data = {"stream": True}
        else:
            try:
                request_data = json.loads(body)
            except json.JSONDecodeError:
                request_data = {"stream": True}

        # Force streaming
        request_data["stream"] = True
        modified_body = json.dumps(request_data).encode()

        # Create modified request with stream=true
        modified_scope = {
            **request.scope,
            "_body": modified_body,
        }

        modified_request = StarletteRequest(
            scope=modified_scope,
            receive=request.receive,
        )
        modified_request._body = modified_body

        # Delegate to handle_request which will handle streaming
        result = await self.handle_request(modified_request)

        # Ensure we return a streaming response
        if not isinstance(result, StreamingResponse):
            # This shouldn't happen since we forced stream=true, but handle it gracefully
            logger.warning(
                "unexpected_response_type",
                expected="StreamingResponse",
                actual=type(result).__name__,
            )
            return StreamingResponse(
                iter([result.body if hasattr(result, "body") else b""]),
                media_type="text/event-stream",
                headers={"X-Claude-SDK-Response": "true"},
            )

        return result

    async def cleanup(self) -> None:
        """Cleanup resources when shutting down."""
        try:
            # Shutdown session manager first
            if self.session_manager:
                await self.session_manager.shutdown()
                self.session_manager = None

            # Close handler
            if self.handler:
                await self.handler.close()
                self.handler = None

            # Clear references to prevent memory leaks
            self._detection_service = None

            # Mark as not initialized
            self._initialized = False

            logger.debug("adapter_cleanup_completed")

        except Exception as e:
            logger.error(
                "adapter_cleanup_failed",
                error=str(e),
                exc_info=e,
            )

    async def close(self) -> None:
        """Compatibility method - delegates to cleanup()."""
        await self.cleanup()

    # BaseHTTPAdapter abstract method implementations
    # Note: ClaudeSDK doesn't use external HTTP, so these methods are minimal implementations

    async def prepare_provider_request(
        self, body: bytes, headers: dict[str, str], endpoint: str
    ) -> tuple[bytes, dict[str, str]]:
        """Prepare request for ClaudeSDK (minimal implementation).

        ClaudeSDK uses the local Claude SDK rather than making HTTP requests,
        so this just passes through the body and headers.
        """
        return body, headers

    async def process_provider_response(
        self, response: "httpx.Response", endpoint: str
    ) -> Response | StreamingResponse:
        """Process response from ClaudeSDK (minimal implementation).

        ClaudeSDK handles response processing in handle_request method,
        so this should not be called in normal operation.
        """
        # This shouldn't be called for ClaudeSDK, but provide a fallback
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=dict(response.headers),
        )

    async def get_target_url(self, endpoint: str) -> str:
        """Get target URL for ClaudeSDK (minimal implementation).

        ClaudeSDK uses local SDK rather than HTTP URLs,
        so this returns a placeholder URL.
        """
        return f"claude-sdk://local/{endpoint.lstrip('/')}"
