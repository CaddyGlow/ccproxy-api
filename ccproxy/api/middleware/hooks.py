"""Hooks middleware for request lifecycle management."""

import time
from datetime import datetime
from typing import Any, cast

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import StreamingResponse

from ccproxy.api.middleware.streaming_hooks import StreamingResponseWithHooks
from ccproxy.core.logging import TraceBoundLogger, get_logger
from ccproxy.core.plugins.hooks import HookEvent, HookManager
from ccproxy.core.plugins.hooks.base import HookContext
from ccproxy.utils.headers import (
    extract_request_headers,
    extract_response_headers,
)


logger: TraceBoundLogger = get_logger()


class HooksMiddleware(BaseHTTPMiddleware):
    """Middleware that emits hook lifecycle events for requests.

    This middleware wraps the entire request-response cycle and emits:
    - REQUEST_STARTED before processing request
    - REQUEST_COMPLETED on successful response
    - REQUEST_FAILED on error

    It maintains RequestContext compatibility and provides centralized
    hook emission for both regular and streaming responses.
    """

    def __init__(self, app: Any, hook_manager: HookManager | None = None) -> None:
        """Initialize the hooks middleware.

        Args:
            app: ASGI application
            hook_manager: Hook manager for emitting events
        """
        super().__init__(app)
        self.hook_manager = hook_manager

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        """Dispatch the request with hook emission.

        Args:
            request: The incoming request
            call_next: The next middleware/handler in the chain

        Returns:
            The response from downstream handlers
        """
        # Get hook manager from app state if not set during init
        hook_manager = self.hook_manager
        if not hook_manager and hasattr(request.app.state, "hook_manager"):
            hook_manager = request.app.state.hook_manager

        # Skip hook emission if no hook manager available
        if not hook_manager:
            return cast(Response, await call_next(request))

        # Extract request_id from ASGI scope extensions
        request_id = getattr(request.state, "request_id", None)
        if not request_id:
            # Fallback to headers or generate one
            request_id = request.headers.get(
                "X-Request-ID", f"req-{int(time.time() * 1000)}"
            )

        # Get or create RequestContext
        from ccproxy.core.request_context import RequestContext

        request_context = RequestContext.get_current()
        if not request_context:
            # Create minimal context if none exists
            start_time_perf = time.perf_counter()
            request_context = RequestContext(
                request_id=request_id,
                start_time=start_time_perf,
                logger=logger,
            )

        # Wall-clock time for human-readable timestamps
        start_time = time.time()

        # Create hook context for the request
        logger.debug("headers_on_request_start", headers=dict(request.headers))
        hook_context = HookContext(
            event=HookEvent.REQUEST_STARTED,  # Will be overridden in emit calls
            timestamp=datetime.fromtimestamp(start_time),
            data={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                # Extract headers using utility function
                "headers": extract_request_headers(request),
            },
            metadata=getattr(request_context, "metadata", {}),
            request=request,
        )

        try:
            # Emit REQUEST_STARTED before processing
            await hook_manager.emit_with_context(hook_context)

            # Capture and emit HTTP_REQUEST hook with body
            await self._emit_http_request_hook(hook_manager, request, hook_context)

            # Process the request
            response = cast(Response, await call_next(request))

            # Update hook context with response information
            end_time = time.time()
            response_hook_context = HookContext(
                event=HookEvent.REQUEST_COMPLETED,  # Will be overridden in emit calls
                timestamp=datetime.fromtimestamp(start_time),
                data={
                    "request_id": request_id,
                    "method": request.method,
                    "url": str(request.url),
                    "headers": extract_request_headers(request),
                    "response_status": getattr(response, "status_code", 200),
                    # Response headers preserved via extract_response_headers
                    "response_headers": extract_response_headers(response),
                    "duration": end_time - start_time,
                },
                metadata=getattr(request_context, "metadata", {}),
                request=request,
                response=response,
            )

            # Handle streaming responses specially
            # Check if it's a streaming response (including middleware wrapped streaming responses)
            is_streaming = (
                isinstance(response, StreamingResponse)
                or type(response).__name__ == "_StreamingResponse"
            )
            logger.debug(
                "hooks_middleware_checking_response_type",
                response_type=type(response).__name__,
                response_class=str(type(response)),
                is_streaming=is_streaming,
                request_id=request_id,
            )
            if is_streaming:
                # For streaming responses, wrap with hook emission on completion
                # Don't emit REQUEST_COMPLETED here - it will be emitted when streaming actually completes

                logger.debug(
                    "hooks_middleware_wrapping_streaming_response",
                    request_id=request_id,
                    method=request.method,
                    url=str(request.url),
                    status_code=getattr(response, "status_code", 200),
                    duration=end_time - start_time,
                    response_type="streaming",
                    category="hooks",
                )

                # Wrap the streaming response to emit hooks on completion
                request_data = {
                    "method": request.method,
                    "url": str(request.url),
                    "headers": extract_request_headers(request),
                }

                # Include RequestContext metadata if available
                request_metadata: dict[str, Any] = {}
                if request_context:
                    request_metadata = getattr(request_context, "metadata", {})

                response_stream = cast(StreamingResponse, response)

                # Coerce body iterator to AsyncGenerator[bytes]
                async def _coerce_bytes() -> Any:
                    async for chunk in response_stream.body_iterator:
                        if isinstance(chunk, bytes):
                            yield chunk
                        elif isinstance(chunk, memoryview):
                            yield bytes(chunk)
                        else:
                            yield str(chunk).encode("utf-8", errors="replace")

                wrapped_response = StreamingResponseWithHooks(
                    content=_coerce_bytes(),
                    hook_manager=hook_manager,
                    request_id=request_id,
                    request_data=request_data,
                    request_metadata=request_metadata,
                    start_time=start_time,
                    status_code=response_stream.status_code,
                    headers=dict(response_stream.headers),
                    media_type=response_stream.media_type,
                )

                return wrapped_response
            else:
                # For regular responses, emit HTTP_RESPONSE and REQUEST_COMPLETED
                await self._emit_http_response_hook(
                    hook_manager, request, response, hook_context
                )
                await hook_manager.emit_with_context(response_hook_context)

                logger.debug(
                    "hooks_middleware_request_completed",
                    request_id=request_id,
                    method=request.method,
                    url=str(request.url),
                    status_code=getattr(response, "status_code", 200),
                    duration=end_time - start_time,
                    response_type="regular",
                    category="hooks",
                )

            return response

        except Exception as e:
            # Update hook context with error information
            end_time = time.time()
            error_hook_context = HookContext(
                event=HookEvent.REQUEST_FAILED,  # Will be overridden in emit calls
                timestamp=datetime.fromtimestamp(start_time),
                data={
                    "request_id": request_id,
                    "method": request.method,
                    "url": str(request.url),
                    "headers": extract_request_headers(request),
                    "duration": end_time - start_time,
                },
                metadata=getattr(request_context, "metadata", {}),
                request=request,
                error=e,
            )

            # Emit REQUEST_FAILED on error
            try:
                await hook_manager.emit_with_context(error_hook_context)
            except Exception as hook_error:
                logger.error(
                    "hooks_middleware_hook_emission_failed",
                    request_id=request_id,
                    original_error=str(e),
                    hook_error=str(hook_error),
                    category="hooks",
                )

            logger.debug(
                "hooks_middleware_request_failed",
                request_id=request_id,
                method=request.method,
                url=str(request.url),
                error=str(e),
                duration=end_time - start_time,
                category="hooks",
            )

            # Re-raise the original exception
            raise

    async def _emit_http_request_hook(
        self, hook_manager: HookManager, request: Request, base_context: HookContext
    ) -> None:
        """Emit HTTP_REQUEST hook with request body capture.

        Args:
            hook_manager: Hook manager for emitting events
            request: FastAPI request object
            base_context: Base hook context for request metadata
        """
        try:
            # Capture request body - this may be empty for GET requests
            request_body = await self._capture_request_body(request)

            # Build HTTP request context
            http_request_context = {
                "request_id": base_context.data.get("request_id"),
                "method": request.method,
                "url": str(request.url),
                "headers": extract_request_headers(request),
                "is_client_request": True,  # Distinguish from provider requests
            }

            # Add body information if available - pass raw data to let formatters handle conversion
            if request_body:
                http_request_context["body"] = request_body
                # Set content type for formatters to use
                content_type = request.headers.get("content-type", "")
                http_request_context["is_json"] = "application/json" in content_type

            # Emit HTTP_REQUEST hook
            await hook_manager.emit(HookEvent.HTTP_REQUEST, http_request_context)

        except Exception as e:
            logger.debug(
                "http_request_hook_emission_failed",
                error=str(e),
                request_id=base_context.data.get("request_id"),
                method=request.method,
                category="hooks",
            )

    async def _emit_http_response_hook(
        self,
        hook_manager: HookManager,
        request: Request,
        response: Response,
        base_context: HookContext,
    ) -> None:
        """Emit HTTP_RESPONSE hook with response body capture.

        Args:
            hook_manager: Hook manager for emitting events
            request: FastAPI request object
            response: FastAPI response object
            base_context: Base hook context for request metadata
        """
        try:
            # Build HTTP response context
            http_response_context = {
                "request_id": base_context.data.get("request_id"),
                "method": request.method,
                "url": str(request.url),
                "headers": extract_request_headers(request),
                "status_code": getattr(response, "status_code", 200),
                "response_headers": dict(getattr(response, "headers", {})),
                "is_client_response": True,  # Distinguish from provider responses
            }

            # Capture response body for non-streaming responses
            response_body = await self._capture_response_body(response)
            if response_body is not None:
                http_response_context["response_body"] = response_body

            # Emit HTTP_RESPONSE hook
            await hook_manager.emit(HookEvent.HTTP_RESPONSE, http_response_context)

        except Exception as e:
            logger.debug(
                "http_response_hook_emission_failed",
                error=str(e),
                request_id=base_context.data.get("request_id"),
                status_code=getattr(response, "status_code", 200),
                category="hooks",
            )

    async def _capture_request_body(self, request: Request) -> bytes:
        """Capture request body, handling caching for multiple reads.

        Args:
            request: FastAPI request object

        Returns:
            Request body as bytes
        """
        try:
            # Check if body is already cached
            if hasattr(request.state, "cached_body"):
                return cast(bytes, request.state.cached_body)

            # Read and cache body for future use
            body = await request.body()
            request.state.cached_body = body
            return body

        except Exception as e:
            logger.debug(
                "request_body_capture_failed",
                error=str(e),
                method=request.method,
                url=str(request.url),
            )
            return b""

    async def _capture_response_body(self, response: Response) -> bytes | None:
        """Capture response body for non-streaming responses.

        Args:
            response: FastAPI response object

        Returns:
            Response body as raw bytes or None if unavailable
        """
        try:
            # For regular Response objects, try to get body
            if hasattr(response, "body") and response.body:
                body_data = response.body
                logger.debug(
                    "response_body_capture_debug",
                    body_type=type(body_data).__name__,
                    body_size=len(body_data)
                    if hasattr(body_data, "__len__")
                    else "no_len",
                    has_body_attr=hasattr(response, "body"),
                    body_truthy=bool(response.body),
                )
                # Ensure return type is bytes
                if isinstance(body_data, memoryview):
                    return body_data.tobytes()
                return body_data

            logger.debug(
                "response_body_capture_none",
                has_body_attr=hasattr(response, "body"),
                body_truthy=bool(getattr(response, "body", None)),
                response_type=type(response).__name__,
            )
            return None

        except Exception as e:
            logger.debug(
                "response_body_capture_failed",
                error=str(e),
                status_code=getattr(response, "status_code", 200),
            )
            return None


def create_hooks_middleware(
    hook_manager: HookManager | None = None,
) -> type[HooksMiddleware]:
    """Create a hooks middleware class with the provided hook manager.

    Args:
        hook_manager: Hook manager for emitting events

    Returns:
        HooksMiddleware class configured with the hook manager
    """

    class ConfiguredHooksMiddleware(HooksMiddleware):
        def __init__(self, app: Any) -> None:
            super().__init__(app, hook_manager)

    return ConfiguredHooksMiddleware
