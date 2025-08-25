"""Claude SDK adapter implementation using delegation pattern."""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any, cast

import httpx
from fastapi import HTTPException, Request
from starlette.responses import Response, StreamingResponse

from ccproxy.core.logging import get_plugin_logger
from ccproxy.observability.streaming_response import StreamingResponseWithLogging
from ccproxy.services.adapters.base import BaseAdapter

from .auth import NoOpAuthManager
from .config import ClaudeSDKSettings
from .format_adapter import ClaudeSDKFormatAdapter
from .handler import ClaudeSDKHandler
from .manager import SessionManager
from .transformers.request import ClaudeSDKRequestTransformer
from .transformers.response import ClaudeSDKResponseTransformer


logger = get_plugin_logger()


class ClaudeSDKAdapter(BaseAdapter):
    """Claude SDK adapter implementation using delegation pattern.

    This adapter delegates to ProxyService for request handling,
    following the same pattern as claude_api and codex plugins.
    """

    def __init__(
        self, config: ClaudeSDKSettings, proxy_service: Any | None = None
    ) -> None:
        """Initialize the Claude SDK adapter."""
        import uuid

        self.logger = get_plugin_logger()
        self.config = config
        self.proxy_service = proxy_service

        # Generate or set default session ID
        self._runtime_default_session_id = None
        if (
            config.auto_generate_default_session
            and config.sdk_session_pool
            and config.sdk_session_pool.enabled
        ):
            # Generate a random session ID for this runtime
            self._runtime_default_session_id = f"auto-{uuid.uuid4().hex[:12]}"
            self.logger.debug(
                "auto_generated_session",
                session_id=self._runtime_default_session_id,
                lifetime="runtime",
            )
        elif config.default_session_id:
            self._runtime_default_session_id = config.default_session_id
            self.logger.debug(
                "using_configured_default_session",
                session_id=self._runtime_default_session_id,
            )

        # Initialize SessionManager if session pool is enabled
        session_manager = None
        if config.sdk_session_pool and config.sdk_session_pool.enabled:
            session_manager = SessionManager(config=config)
            self.logger.debug(
                "adapter_session_pool_enabled",
                session_ttl=config.sdk_session_pool.session_ttl,
                max_sessions=config.sdk_session_pool.max_sessions,
                has_default_session=bool(self._runtime_default_session_id),
                auto_generated=config.auto_generate_default_session,
            )

        self.session_manager = session_manager
        self.handler: ClaudeSDKHandler | None = ClaudeSDKHandler(
            config=config, session_manager=session_manager
        )
        self.format_adapter = ClaudeSDKFormatAdapter()
        self.request_transformer = ClaudeSDKRequestTransformer()
        # Initialize response transformer with CORS settings
        cors_settings = (
            getattr(proxy_service.config, "cors", None) if proxy_service else None
        )
        self.response_transformer = ClaudeSDKResponseTransformer(cors_settings)
        self.auth_manager = NoOpAuthManager()
        self._detection_service: Any | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the adapter and start session manager if needed."""
        if not self._initialized:
            if self.session_manager:
                await self.session_manager.start()
                self.logger.info("session_manager_started")
            self._initialized = True

    def set_detection_service(self, detection_service: Any) -> None:
        """Set the detection service.

        Args:
            detection_service: Claude CLI detection service
        """
        self._detection_service = detection_service

    async def handle_request(
        self, request: Request, endpoint: str, method: str, **kwargs: Any
    ) -> Response | StreamingResponse:
        # Ensure adapter is initialized
        await self.initialize()

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

        # Check if format conversion is needed (OpenAI to Anthropic)
        # The endpoint will contain the path after the prefix, e.g., "/v1/chat/completions"
        from ccproxy.config.constants import OPENAI_CHAT_COMPLETIONS_PATH

        needs_conversion = endpoint.endswith(OPENAI_CHAT_COMPLETIONS_PATH)
        if needs_conversion and self.format_adapter:
            request_data = await self.format_adapter.adapt_request(request_data)

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
            self.logger.debug(
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

        # Get RequestContext - it must exist when called via ProxyService
        from ccproxy.observability.context import RequestContext

        request_context: RequestContext | None = RequestContext.get_current()
        if not request_context:
            raise HTTPException(
                status_code=500,
                detail="RequestContext not available - plugin must be called via ProxyService",
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

        self.logger.debug(
            "plugin_request",
            plugin="claude_sdk",
            endpoint=endpoint,
            target_url=f"claude-sdk://{session_id}"
            if session_id
            else "claude-sdk://direct",
            model=model,
            is_streaming=stream,
            needs_conversion=needs_conversion,
            session_id=session_id,
            category="http",
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
                async def stream_generator() -> AsyncIterator[bytes]:
                    """Generate SSE stream from handler's async iterator."""
                    try:
                        if needs_conversion:
                            # Use OpenAIStreamProcessor to convert Claude SSE to OpenAI SSE format
                            from ccproxy.adapters.openai.streaming import (
                                OpenAIStreamProcessor,
                            )

                            # Create processor with SSE output format
                            processor = OpenAIStreamProcessor(
                                model=model,
                                enable_usage=True,
                                enable_tool_calls=True,
                                output_format="sse",  # Generate SSE strings
                            )

                            # Process the stream and yield SSE formatted chunks
                            # Cast to AsyncIterator since we know stream=True
                            stream_result = cast(AsyncIterator[dict[str, Any]], result)
                            async for sse_chunk in processor.process_stream(
                                stream_result
                            ):
                                # sse_chunk is already a formatted SSE string when output_format="sse"
                                if isinstance(sse_chunk, str):
                                    yield sse_chunk.encode()
                                else:
                                    # Should not happen, but handle gracefully
                                    yield str(sse_chunk).encode()
                        else:
                            # Pass through Claude SSE format as-is
                            # Cast to AsyncIterator since we know stream=True
                            stream_result = cast(AsyncIterator[dict[str, Any]], result)
                            async for chunk in stream_result:
                                import json

                                data = json.dumps(chunk)
                                yield f"data: {data}\n\n".encode()
                    except asyncio.CancelledError as e:
                        self.logger.warning(
                            "streaming_cancelled",
                            error=str(e),
                            exc_info=e,
                            category="streaming",
                        )
                        raise
                    except httpx.TimeoutException as e:
                        self.logger.error(
                            "streaming_timeout",
                            error=str(e),
                            exc_info=e,
                            category="streaming",
                        )
                        error_chunk = {"error": "Request timed out"}
                        yield f"data: {json.dumps(error_chunk)}\n\n".encode()
                    except httpx.HTTPError as e:
                        self.logger.error(
                            "streaming_http_error",
                            error=str(e),
                            status_code=getattr(e.response, "status_code", None)
                            if hasattr(e, "response")
                            else None,
                            exc_info=e,
                            category="streaming",
                        )
                        error_chunk = {"error": f"HTTP error: {e}"}
                        yield f"data: {json.dumps(error_chunk)}\n\n".encode()
                    except Exception as e:
                        self.logger.error(
                            "streaming_unexpected_error",
                            error=str(e),
                            exc_info=e,
                            category="streaming",
                        )
                        error_chunk = {"error": str(e)}
                        yield f"data: {json.dumps(error_chunk)}\n\n".encode()
                        # Don't add extra [DONE] here as OpenAIStreamProcessor already adds it

                # Get metrics from proxy_service if available
                metrics = None
                if self.proxy_service and hasattr(self.proxy_service, "metrics"):
                    metrics = self.proxy_service.metrics

                # Use StreamingResponseWithLogging for automatic access logging
                return StreamingResponseWithLogging(
                    content=stream_generator(),
                    request_context=request_context,
                    metrics=metrics,
                    status_code=200,
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Claude-SDK-Response": "true",
                    },
                )
            else:
                # Convert MessageResponse to dict for JSON response
                from .models import MessageResponse

                if isinstance(result, MessageResponse):
                    response_data = result.model_dump()
                else:
                    # This shouldn't happen when stream=False, but handle it
                    response_data = cast(dict[str, Any], result)

                # Convert to OpenAI format if needed
                if needs_conversion and self.format_adapter:
                    response_data = await self.format_adapter.adapt_response(
                        response_data
                    )

                return Response(
                    content=json.dumps(response_data),
                    media_type="application/json",
                    headers={
                        "X-Claude-SDK-Response": "true",
                    },
                )

        except httpx.TimeoutException as e:
            self.logger.error(
                "request_timeout",
                error=str(e),
                exc_info=e,
                category="http",
            )
            raise HTTPException(status_code=408, detail="Request timed out") from e
        except httpx.HTTPError as e:
            self.logger.error(
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
            self.logger.warning(
                "request_cancelled",
                error=str(e),
                exc_info=e,
            )
            raise
        except Exception as e:
            self.logger.error(
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

        from starlette.requests import Request as StarletteRequest

        modified_request = StarletteRequest(
            scope=modified_scope,
            receive=request.receive,
        )
        modified_request._body = modified_body

        # Delegate to handle_request which will handle streaming
        result = await self.handle_request(modified_request, endpoint, "POST", **kwargs)

        # Ensure we return a streaming response
        if not isinstance(result, StreamingResponse):
            # This shouldn't happen since we forced stream=true, but handle it gracefully
            self.logger.warning(
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
            self.proxy_service = None
            self._detection_service = None

            # Mark as not initialized
            self._initialized = False

            self.logger.debug("adapter_cleanup_completed")

        except Exception as e:
            self.logger.error(
                "adapter_cleanup_failed",
                error=str(e),
                exc_info=e,
            )

    async def close(self) -> None:
        """Compatibility method - delegates to cleanup()."""
        await self.cleanup()
