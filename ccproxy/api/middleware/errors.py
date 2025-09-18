"""Error handling middleware for CCProxy API Server."""

import traceback
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from ccproxy.core.constants import (
    FORMAT_ANTHROPIC_MESSAGES,
    FORMAT_OPENAI_CHAT,
    FORMAT_OPENAI_RESPONSES,
)
from ccproxy.core.errors import (
    AuthenticationError,
    ClaudeProxyError,
    DockerError,
    MiddlewareError,
    ModelNotFoundError,
    NotFoundError,
    PermissionError,
    ProxyAuthenticationError,
    ProxyConnectionError,
    ProxyError,
    ProxyTimeoutError,
    RateLimitError,
    ServiceUnavailableError,
    TimeoutError,
    TransformationError,
    ValidationError,
)
from ccproxy.core.logging import get_logger
from ccproxy.llms.models import anthropic as anthropic_models
from ccproxy.llms.models import openai as openai_models


logger = get_logger(__name__)


def _detect_format_from_path(path: str) -> str | None:
    """Detect the expected format from the request path.

    Args:
        path: Request URL path

    Returns:
        Detected format or None if cannot determine
    """
    if "/chat/completions" in path:
        return FORMAT_OPENAI_CHAT
    elif "/messages" in path:
        return FORMAT_ANTHROPIC_MESSAGES
    elif "/responses" in path:
        return FORMAT_OPENAI_RESPONSES
    return None


def _get_format_aware_error_content(
    error_type: str, message: str, status_code: int, base_format: str | None
) -> dict[str, Any]:
    """Create format-aware error response content using proper models.

    Args:
        error_type: Type of error for logging
        message: Error message
        status_code: HTTP status code
        base_format: Base format from format_chain[0]

    Returns:
        Formatted error response content using proper models
    """
    # Default CCProxy format
    default_content = {
        "error": {
            "type": error_type,
            "message": message,
        }
    }

    try:
        if base_format in {FORMAT_OPENAI_CHAT, FORMAT_OPENAI_RESPONSES}:
            # Use OpenAI error model
            error_detail = openai_models.ErrorDetail(
                message=message,
                type=error_type,
                code=error_type
                if base_format == FORMAT_OPENAI_RESPONSES
                else str(status_code),
                param=None,
            )
            error_response = openai_models.ErrorResponse(error=error_detail)
            return error_response.model_dump()

        elif base_format == FORMAT_ANTHROPIC_MESSAGES:
            # Use Anthropic error model
            # APIError has a fixed type field, so create a generic ErrorDetail instead
            api_error = anthropic_models.ErrorDetail(message=message)
            # Anthropic error format has 'type': 'error' at top level
            return {"type": "error", "error": api_error.model_dump()}

    except Exception as e:
        # Log the error but don't fail - fallback to default format
        logger.warning(
            "format_aware_error_creation_failed",
            base_format=base_format,
            error_type=error_type,
            fallback_reason=str(e),
            category="middleware",
        )

    # Fallback to default format
    return default_content


def setup_error_handlers(app: FastAPI) -> None:
    """Setup error handlers for the FastAPI application.

    Args:
        app: FastAPI application instance
    """
    logger.debug("error_handlers_setup_start", category="lifecycle")

    # Metrics are now handled by the metrics plugin via hooks
    metrics = None

    # Define error type mappings with status codes and error types
    ERROR_MAPPINGS: dict[type[Exception], tuple[int | None, str]] = {
        ClaudeProxyError: (None, "claude_proxy_error"),  # Uses exc.status_code
        ValidationError: (400, "validation_error"),
        AuthenticationError: (401, "authentication_error"),
        ProxyAuthenticationError: (401, "proxy_authentication_error"),
        PermissionError: (403, "permission_error"),
        NotFoundError: (404, "not_found_error"),
        ModelNotFoundError: (404, "model_not_found_error"),
        TimeoutError: (408, "timeout_error"),
        RateLimitError: (429, "rate_limit_error"),
        ProxyError: (500, "proxy_error"),
        TransformationError: (500, "transformation_error"),
        MiddlewareError: (500, "middleware_error"),
        DockerError: (500, "docker_error"),
        ProxyConnectionError: (502, "proxy_connection_error"),
        ServiceUnavailableError: (503, "service_unavailable_error"),
        ProxyTimeoutError: (504, "proxy_timeout_error"),
    }

    async def unified_error_handler(
        request: Request,
        exc: Exception,
        status_code: int | None = None,
        error_type: str | None = None,
        include_client_info: bool = False,
    ) -> JSONResponse:
        """Unified error handler for all exception types.

        Args:
            request: The incoming request
            exc: The exception that was raised
            status_code: HTTP status code to return
            error_type: Type of error for logging and response
            include_client_info: Whether to include client IP in logs
        """
        # Get status code from exception if it has one
        if status_code is None:
            status_code = getattr(exc, "status_code", 500)

        # Determine error type if not provided
        if error_type is None:
            error_type = getattr(exc, "error_type", "unknown_error")

        # Get request ID from request state or headers
        request_id = getattr(request.state, "request_id", None) or request.headers.get(
            "x-request-id"
        )

        # Store status code in request state for access logging
        if hasattr(request.state, "context") and hasattr(
            request.state.context, "metadata"
        ):
            request.state.context.metadata["status_code"] = status_code

        # Build log kwargs
        log_kwargs = {
            "error_type": error_type,
            "error_message": str(exc),
            "status_code": status_code,
            "request_method": request.method,
            "request_url": str(request.url.path),
        }

        # Add client info if needed (for auth errors)
        if include_client_info and request.client:
            log_kwargs["client_ip"] = request.client.host
            if error_type in ("authentication_error", "proxy_authentication_error"):
                log_kwargs["user_agent"] = request.headers.get("user-agent", "unknown")

        # Log the error
        logger.error(
            f"{error_type.replace('_', ' ').title()}",
            **log_kwargs,
            category="middleware",
        )

        # Record error in metrics
        if metrics:
            metrics.record_error(
                error_type=error_type,
                endpoint=str(request.url.path),
                model=None,
                service_type="middleware",
            )

        # Prepare headers with x-request-id if available
        headers = {}
        if request_id:
            headers["x-request-id"] = request_id

        # Detect format from request context for format-aware error responses
        base_format = None
        try:
            if hasattr(request.state, "context") and hasattr(
                request.state.context, "format_chain"
            ):
                format_chain = request.state.context.format_chain
                if format_chain and len(format_chain) > 0:
                    base_format = format_chain[
                        0
                    ]  # First format is the client's expected format
                    logger.debug(
                        "format_aware_error_detected",
                        base_format=base_format,
                        format_chain=format_chain,
                        category="middleware",
                    )
        except Exception as e:
            logger.debug("format_detection_failed", error=str(e), category="middleware")

        # Get format-aware error content
        error_content = _get_format_aware_error_content(
            error_type=error_type,
            message=str(exc),
            status_code=status_code,
            base_format=base_format,
        )

        # Return JSON response with format-aware content
        return JSONResponse(
            status_code=status_code,
            content=error_content,
            headers=headers,
        )

    # Register specific error handlers using the unified handler
    for exc_class, (status, err_type) in ERROR_MAPPINGS.items():
        # Determine if this error type should include client info
        include_client = err_type in (
            "authentication_error",
            "proxy_authentication_error",
            "permission_error",
            "rate_limit_error",
        )

        # Create a closure to capture the specific error configuration
        def make_handler(
            status_code: int | None, error_type: str, include_client_info: bool
        ) -> Callable[[Request, Exception], Awaitable[JSONResponse]]:
            async def handler(request: Request, exc: Exception) -> JSONResponse:
                return await unified_error_handler(
                    request, exc, status_code, error_type, include_client_info
                )

            return handler

        # Register the handler
        app.exception_handler(exc_class)(make_handler(status, err_type, include_client))

    # FastAPI validation errors
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Handle FastAPI request validation errors with format awareness."""
        # Get request ID from request state or headers
        request_id = getattr(request.state, "request_id", None) or request.headers.get(
            "x-request-id"
        )

        # Try to get format from request context (set by middleware)
        base_format = None
        try:
            if hasattr(request.state, "context") and hasattr(
                request.state.context, "format_chain"
            ):
                format_chain = request.state.context.format_chain
                if format_chain and len(format_chain) > 0:
                    base_format = format_chain[0]
        except Exception:
            pass  # Fallback to path detection if needed

        # Fallback: detect format from path if context isn't available
        if base_format is None:
            base_format = _detect_format_from_path(str(request.url.path))

        # Create a readable error message from validation errors
        error_details = []
        for error in exc.errors():
            loc = " -> ".join(str(x) for x in error["loc"])
            error_details.append(f"{loc}: {error['msg']}")

        error_message = "; ".join(error_details)

        # Log the validation error
        logger.warning(
            "Request validation error",
            error_type="validation_error",
            error_message=error_message,
            status_code=422,
            request_method=request.method,
            request_url=str(request.url.path),
            base_format=base_format,
            category="middleware",
        )

        # Prepare headers with x-request-id if available
        headers = {}
        if request_id:
            headers["x-request-id"] = request_id

        # Get format-aware error content
        error_content = _get_format_aware_error_content(
            error_type="validation_error",
            message=error_message,
            status_code=422,
            base_format=base_format,
        )

        return JSONResponse(
            status_code=422,
            content=error_content,
            headers=headers,
        )

    # Standard HTTP exceptions
    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request, exc: HTTPException
    ) -> JSONResponse:
        """Handle HTTP exceptions."""
        # Get request ID from request state or headers
        request_id = getattr(request.state, "request_id", None) or request.headers.get(
            "x-request-id"
        )

        # Store status code in request state for access logging
        if hasattr(request.state, "context") and hasattr(
            request.state.context, "metadata"
        ):
            request.state.context.metadata["status_code"] = exc.status_code

        # Don't log stack trace for expected errors (404, 401)
        if exc.status_code in (404, 401):
            log_func = logger.debug if exc.status_code == 404 else logger.warning

            log_func(
                f"HTTP {exc.status_code} error",
                error_type=f"http_{exc.status_code}",
                error_message=exc.detail,
                status_code=exc.status_code,
                request_method=request.method,
                request_url=str(request.url.path),
                category="middleware",
            )
        else:
            # Log with basic stack trace (no local variables)
            stack_trace = traceback.format_exc(limit=5)  # Limit to 5 frames

            logger.error(
                "HTTP exception",
                error_type="http_error",
                error_message=exc.detail,
                status_code=exc.status_code,
                request_method=request.method,
                request_url=str(request.url.path),
                stack_trace=stack_trace,
                category="middleware",
            )

        # Record error in metrics
        if metrics:
            if exc.status_code == 404:
                error_type = "http_404"
            elif exc.status_code == 401:
                error_type = "http_401"
            else:
                error_type = "http_error"
            metrics.record_error(
                error_type=error_type,
                endpoint=str(request.url.path),
                model=None,
                service_type="middleware",
            )

        # Prepare headers with x-request-id if available
        headers = {}
        if request_id:
            headers["x-request-id"] = request_id

        # Detect format from request context for format-aware error responses
        base_format = None
        try:
            if hasattr(request.state, "context") and hasattr(
                request.state.context, "format_chain"
            ):
                format_chain = request.state.context.format_chain
                if format_chain and len(format_chain) > 0:
                    base_format = format_chain[0]
        except Exception:
            pass  # Ignore format detection errors

        # Determine error type for format-aware response
        if exc.status_code == 404:
            error_type = "not_found"
        elif exc.status_code == 401:
            error_type = "authentication_error"
        else:
            error_type = "http_error"

        # Get format-aware error content
        error_content = _get_format_aware_error_content(
            error_type=error_type,
            message=exc.detail,
            status_code=exc.status_code,
            base_format=base_format,
        )

        return JSONResponse(
            status_code=exc.status_code,
            content=error_content,
            headers=headers,
        )

    @app.exception_handler(StarletteHTTPException)
    async def starlette_http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        """Handle Starlette HTTP exceptions."""
        # Get request ID from request state or headers
        request_id = getattr(request.state, "request_id", None) or request.headers.get(
            "x-request-id"
        )

        # Don't log stack trace for 404 errors as they're expected
        if exc.status_code == 404:
            logger.debug(
                "Starlette HTTP 404 error",
                error_type="starlette_http_404",
                error_message=exc.detail,
                status_code=404,
                request_method=request.method,
                request_url=str(request.url.path),
                category="middleware",
            )
        else:
            logger.error(
                "Starlette HTTP exception",
                error_type="starlette_http_error",
                error_message=exc.detail,
                status_code=exc.status_code,
                request_method=request.method,
                request_url=str(request.url.path),
                category="middleware",
            )

        # Record error in metrics
        if metrics:
            error_type = (
                "starlette_http_404"
                if exc.status_code == 404
                else "starlette_http_error"
            )
            metrics.record_error(
                error_type=error_type,
                endpoint=str(request.url.path),
                model=None,
                service_type="middleware",
            )

        # Prepare headers with x-request-id if available
        headers = {}
        if request_id:
            headers["x-request-id"] = request_id

        # Detect format from request context for format-aware error responses
        base_format = None
        try:
            if hasattr(request.state, "context") and hasattr(
                request.state.context, "format_chain"
            ):
                format_chain = request.state.context.format_chain
                if format_chain and len(format_chain) > 0:
                    base_format = format_chain[0]
        except Exception:
            pass  # Ignore format detection errors

        # Determine error type for format-aware response
        if exc.status_code == 404:
            error_type = "not_found"
        else:
            error_type = "http_error"

        # Get format-aware error content
        error_content = _get_format_aware_error_content(
            error_type=error_type,
            message=exc.detail,
            status_code=exc.status_code,
            base_format=base_format,
        )

        return JSONResponse(
            status_code=exc.status_code,
            content=error_content,
            headers=headers,
        )

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Handle all other unhandled exceptions."""
        # Get request ID from request state or headers
        request_id = getattr(request.state, "request_id", None) or request.headers.get(
            "x-request-id"
        )

        # Store status code in request state for access logging
        if hasattr(request.state, "context") and hasattr(
            request.state.context, "metadata"
        ):
            request.state.context.metadata["status_code"] = 500

        logger.error(
            "Unhandled exception",
            error_type="unhandled_exception",
            error_message=str(exc),
            status_code=500,
            request_method=request.method,
            request_url=str(request.url.path),
            exc_info=True,
            category="middleware",
        )

        # Record error in metrics
        if metrics:
            metrics.record_error(
                error_type="unhandled_exception",
                endpoint=str(request.url.path),
                model=None,
                service_type="middleware",
            )

        # Prepare headers with x-request-id if available
        headers = {}
        if request_id:
            headers["x-request-id"] = request_id

        # Detect format from request context for format-aware error responses
        base_format = None
        try:
            if hasattr(request.state, "context") and hasattr(
                request.state.context, "format_chain"
            ):
                format_chain = request.state.context.format_chain
                if format_chain and len(format_chain) > 0:
                    base_format = format_chain[0]
        except Exception:
            pass  # Ignore format detection errors

        # Get format-aware error content for internal server error
        error_content = _get_format_aware_error_content(
            error_type="internal_server_error",
            message="An internal server error occurred",
            status_code=500,
            base_format=base_format,
        )

        return JSONResponse(
            status_code=500,
            content=error_content,
            headers=headers,
        )

    logger.debug("error_handlers_setup_completed", category="lifecycle")
