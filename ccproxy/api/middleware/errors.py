"""Error handling middleware for CCProxy API Server."""

from collections.abc import Awaitable, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from structlog import get_logger

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
from ccproxy.observability.metrics import get_metrics


logger = get_logger(__name__)


def setup_error_handlers(app: FastAPI) -> None:
    """Setup error handlers for the FastAPI application.

    Args:
        app: FastAPI application instance
    """
    logger.debug("error_handlers_setup_start", category="lifecycle")

    # Get metrics instance for error recording
    try:
        metrics = get_metrics()
        logger.debug("error_handlers_metrics_loaded", category="lifecycle")
    except ImportError as e:
        logger.warning(
            "error_handlers_metrics_import_failed",
            error=str(e),
            exc_info=e,
            category="lifecycle",
        )
        metrics = None
    except (AttributeError, TypeError) as e:
        logger.warning(
            "error_handlers_metrics_unavailable",
            error=str(e),
            exc_info=e,
            category="lifecycle",
        )
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

        # Return JSON response
        return JSONResponse(
            status_code=status_code,
            content={
                "error": {
                    "type": error_type,
                    "message": str(exc),
                }
            },
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

    # Standard HTTP exceptions
    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request, exc: HTTPException
    ) -> JSONResponse:
        """Handle HTTP exceptions."""
        # Store status code in request state for access logging
        if hasattr(request.state, "context") and hasattr(
            request.state.context, "metadata"
        ):
            request.state.context.metadata["status_code"] = exc.status_code

        # Don't log stack trace for expected errors (404, 401)
        if exc.status_code in (404, 401):
            log_level = "debug" if exc.status_code == 404 else "warning"
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
            import traceback

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

        # TODO: Add when in prod hide details in response
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "type": "http_error",
                    "message": exc.detail,
                }
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def starlette_http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        """Handle Starlette HTTP exceptions."""
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
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "type": "http_error",
                    "message": exc.detail,
                }
            },
        )

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Handle all other unhandled exceptions."""
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
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "type": "internal_server_error",
                    "message": "An internal server error occurred",
                }
            },
        )

    logger.debug("error_handlers_setup_completed", category="lifecycle")
