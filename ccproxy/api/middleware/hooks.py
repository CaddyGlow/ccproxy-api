"""Hooks middleware for request lifecycle management."""

import time
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import StreamingResponse

from ccproxy.api.middleware.streaming_hooks import StreamingResponseWithHooks
from ccproxy.core.logging import TraceBoundLogger, get_logger
from ccproxy.hooks import HookEvent, HookManager
from ccproxy.hooks.base import HookContext


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
            return await call_next(request)

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
            request_context = RequestContext(request_id=request_id)

        start_time = time.time()

        # Create hook context for the request
        from datetime import datetime

        hook_context = HookContext(
            event=HookEvent.REQUEST_STARTED,  # Will be overridden in emit calls
            timestamp=datetime.fromtimestamp(start_time),
            data={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "headers": dict(request.headers),
            },
            metadata=getattr(request_context, "metadata", {}),
            request=request,
        )

        try:
            # Emit REQUEST_STARTED before processing
            await hook_manager.emit_with_context(hook_context)

            # Process the request
            response = await call_next(request)

            # Update hook context with response information
            end_time = time.time()
            response_hook_context = HookContext(
                event=HookEvent.REQUEST_COMPLETED,  # Will be overridden in emit calls
                timestamp=datetime.fromtimestamp(start_time),
                data={
                    "request_id": request_id,
                    "method": request.method,
                    "url": str(request.url),
                    "headers": dict(request.headers),
                    "response_status": getattr(response, "status_code", 200),
                    "response_headers": dict(getattr(response, "headers", {})),
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
                    "headers": dict(request.headers),
                }

                # Include RequestContext metadata if available
                request_metadata = {}
                if request_context:
                    request_metadata = getattr(request_context, "metadata", {})

                wrapped_response = StreamingResponseWithHooks(
                    content=response.body_iterator,
                    hook_manager=hook_manager,
                    request_id=request_id,
                    request_data=request_data,
                    request_metadata=request_metadata,
                    start_time=start_time,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=response.media_type,
                )

                return wrapped_response
            else:
                # For regular responses, emit REQUEST_COMPLETED
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
                    "headers": dict(request.headers),
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
