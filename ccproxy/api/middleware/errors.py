"""Error handling middleware for CCProxy API Server."""

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
    logger.debug("error_handlers_setup_start")

    # Get metrics instance for error recording
    try:
        metrics = get_metrics()
        logger.debug("error_handlers_metrics_loaded")
    except Exception as e:
        logger.warning("error_handlers_metrics_unavailable", error=str(e))
        metrics = None

    @app.exception_handler(ClaudeProxyError)
    async def claude_proxy_error_handler(
        request: Request, exc: ClaudeProxyError
    ) -> JSONResponse:
        """Handle Claude proxy specific errors."""
        # Store status code in request state for access logging
        if hasattr(request.state, "context") and hasattr(
            request.state.context, "metadata"
        ):
            request.state.context.metadata["status_code"] = exc.status_code

        logger.error(
            "Claude proxy error",
            error_type="claude_proxy_error",
            error_message=str(exc),
            status_code=exc.status_code,
            request_method=request.method,
            request_url=str(request.url.path),
        )

        # Record error in metrics
        if metrics:
            metrics.record_error(
                error_type="claude_proxy_error",
                endpoint=str(request.url.path),
                model=None,
                service_type="middleware",
            )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "type": exc.error_type,
                    "message": str(exc),
                }
            },
        )

    @app.exception_handler(ValidationError)
    async def validation_error_handler(
        request: Request, exc: ValidationError
    ) -> JSONResponse:
        """Handle validation errors."""
        # Store status code in request state for access logging
        if hasattr(request.state, "context") and hasattr(
            request.state.context, "metadata"
        ):
            request.state.context.metadata["status_code"] = 400

        logger.error(
            "Validation error",
            error_type="validation_error",
            error_message=str(exc),
            status_code=400,
            request_method=request.method,
            request_url=str(request.url.path),
        )

        # Record error in metrics
        if metrics:
            metrics.record_error(
                error_type="validation_error",
                endpoint=str(request.url.path),
                model=None,
                service_type="middleware",
            )
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "type": "validation_error",
                    "message": str(exc),
                }
            },
        )

    @app.exception_handler(AuthenticationError)
    async def authentication_error_handler(
        request: Request, exc: AuthenticationError
    ) -> JSONResponse:
        """Handle authentication errors."""
        logger.error(
            "Authentication error",
            error_type="authentication_error",
            error_message=str(exc),
            status_code=401,
            request_method=request.method,
            request_url=str(request.url.path),
            client_ip=request.client.host if request.client else "unknown",
            user_agent=request.headers.get("user-agent", "unknown"),
        )

        # Record error in metrics
        if metrics:
            metrics.record_error(
                error_type="authentication_error",
                endpoint=str(request.url.path),
                model=None,
                service_type="middleware",
            )
        return JSONResponse(
            status_code=401,
            content={
                "error": {
                    "type": "authentication_error",
                    "message": str(exc),
                }
            },
        )

    @app.exception_handler(PermissionError)
    async def permission_error_handler(
        request: Request, exc: PermissionError
    ) -> JSONResponse:
        """Handle permission errors."""
        logger.error(
            "Permission error",
            error_type="permission_error",
            error_message=str(exc),
            status_code=403,
            request_method=request.method,
            request_url=str(request.url.path),
            client_ip=request.client.host if request.client else "unknown",
        )

        # Record error in metrics
        if metrics:
            metrics.record_error(
                error_type="permission_error",
                endpoint=str(request.url.path),
                model=None,
                service_type="middleware",
            )
        return JSONResponse(
            status_code=403,
            content={
                "error": {
                    "type": "permission_error",
                    "message": str(exc),
                }
            },
        )

    @app.exception_handler(NotFoundError)
    async def not_found_error_handler(
        request: Request, exc: NotFoundError
    ) -> JSONResponse:
        """Handle not found errors."""
        logger.error(
            "Not found error",
            error_type="not_found_error",
            error_message=str(exc),
            status_code=404,
            request_method=request.method,
            request_url=str(request.url.path),
        )

        # Record error in metrics
        if metrics:
            metrics.record_error(
                error_type="not_found_error",
                endpoint=str(request.url.path),
                model=None,
                service_type="middleware",
            )
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "type": "not_found_error",
                    "message": str(exc),
                }
            },
        )

    @app.exception_handler(RateLimitError)
    async def rate_limit_error_handler(
        request: Request, exc: RateLimitError
    ) -> JSONResponse:
        """Handle rate limit errors."""
        logger.error(
            "Rate limit error",
            error_type="rate_limit_error",
            error_message=str(exc),
            status_code=429,
            request_method=request.method,
            request_url=str(request.url.path),
            client_ip=request.client.host if request.client else "unknown",
        )

        # Record error in metrics
        if metrics:
            metrics.record_error(
                error_type="rate_limit_error",
                endpoint=str(request.url.path),
                model=None,
                service_type="middleware",
            )
        return JSONResponse(
            status_code=429,
            content={
                "error": {
                    "type": "rate_limit_error",
                    "message": str(exc),
                }
            },
        )

    @app.exception_handler(ModelNotFoundError)
    async def model_not_found_error_handler(
        request: Request, exc: ModelNotFoundError
    ) -> JSONResponse:
        """Handle model not found errors."""
        logger.error(
            "Model not found error",
            error_type="model_not_found_error",
            error_message=str(exc),
            status_code=404,
            request_method=request.method,
            request_url=str(request.url.path),
        )

        # Record error in metrics
        if metrics:
            metrics.record_error(
                error_type="model_not_found_error",
                endpoint=str(request.url.path),
                model=None,
                service_type="middleware",
            )
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "type": "model_not_found_error",
                    "message": str(exc),
                }
            },
        )

    @app.exception_handler(TimeoutError)
    async def timeout_error_handler(
        request: Request, exc: TimeoutError
    ) -> JSONResponse:
        """Handle timeout errors."""
        logger.error(
            "Timeout error",
            error_type="timeout_error",
            error_message=str(exc),
            status_code=408,
            request_method=request.method,
            request_url=str(request.url.path),
        )

        # Record error in metrics
        if metrics:
            metrics.record_error(
                error_type="timeout_error",
                endpoint=str(request.url.path),
                model=None,
                service_type="middleware",
            )
        return JSONResponse(
            status_code=408,
            content={
                "error": {
                    "type": "timeout_error",
                    "message": str(exc),
                }
            },
        )

    @app.exception_handler(ServiceUnavailableError)
    async def service_unavailable_error_handler(
        request: Request, exc: ServiceUnavailableError
    ) -> JSONResponse:
        """Handle service unavailable errors."""
        logger.error(
            "Service unavailable error",
            error_type="service_unavailable_error",
            error_message=str(exc),
            status_code=503,
            request_method=request.method,
            request_url=str(request.url.path),
        )

        # Record error in metrics
        if metrics:
            metrics.record_error(
                error_type="service_unavailable_error",
                endpoint=str(request.url.path),
                model=None,
                service_type="middleware",
            )
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "type": "service_unavailable_error",
                    "message": str(exc),
                }
            },
        )

    @app.exception_handler(DockerError)
    async def docker_error_handler(request: Request, exc: DockerError) -> JSONResponse:
        """Handle Docker errors."""
        logger.error(
            "Docker error",
            error_type="docker_error",
            error_message=str(exc),
            status_code=500,
            request_method=request.method,
            request_url=str(request.url.path),
        )

        # Record error in metrics
        if metrics:
            metrics.record_error(
                error_type="docker_error",
                endpoint=str(request.url.path),
                model=None,
                service_type="middleware",
            )
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "type": "docker_error",
                    "message": str(exc),
                }
            },
        )

    # Core proxy errors
    @app.exception_handler(ProxyError)
    async def proxy_error_handler(request: Request, exc: ProxyError) -> JSONResponse:
        """Handle proxy errors."""
        logger.error(
            "Proxy error",
            error_type="proxy_error",
            error_message=str(exc),
            status_code=500,
            request_method=request.method,
            request_url=str(request.url.path),
        )

        # Record error in metrics
        if metrics:
            metrics.record_error(
                error_type="proxy_error",
                endpoint=str(request.url.path),
                model=None,
                service_type="middleware",
            )
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "type": "proxy_error",
                    "message": str(exc),
                }
            },
        )

    @app.exception_handler(TransformationError)
    async def transformation_error_handler(
        request: Request, exc: TransformationError
    ) -> JSONResponse:
        """Handle transformation errors."""
        logger.error(
            "Transformation error",
            error_type="transformation_error",
            error_message=str(exc),
            status_code=500,
            request_method=request.method,
            request_url=str(request.url.path),
        )

        # Record error in metrics
        if metrics:
            metrics.record_error(
                error_type="transformation_error",
                endpoint=str(request.url.path),
                model=None,
                service_type="middleware",
            )
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "type": "transformation_error",
                    "message": str(exc),
                }
            },
        )

    @app.exception_handler(MiddlewareError)
    async def middleware_error_handler(
        request: Request, exc: MiddlewareError
    ) -> JSONResponse:
        """Handle middleware errors."""
        logger.error(
            "Middleware error",
            error_type="middleware_error",
            error_message=str(exc),
            status_code=500,
            request_method=request.method,
            request_url=str(request.url.path),
        )

        # Record error in metrics
        if metrics:
            metrics.record_error(
                error_type="middleware_error",
                endpoint=str(request.url.path),
                model=None,
                service_type="middleware",
            )
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "type": "middleware_error",
                    "message": str(exc),
                }
            },
        )

    @app.exception_handler(ProxyConnectionError)
    async def proxy_connection_error_handler(
        request: Request, exc: ProxyConnectionError
    ) -> JSONResponse:
        """Handle proxy connection errors."""
        logger.error(
            "Proxy connection error",
            error_type="proxy_connection_error",
            error_message=str(exc),
            status_code=502,
            request_method=request.method,
            request_url=str(request.url.path),
        )

        # Record error in metrics
        if metrics:
            metrics.record_error(
                error_type="proxy_connection_error",
                endpoint=str(request.url.path),
                model=None,
                service_type="middleware",
            )
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "type": "proxy_connection_error",
                    "message": str(exc),
                }
            },
        )

    @app.exception_handler(ProxyTimeoutError)
    async def proxy_timeout_error_handler(
        request: Request, exc: ProxyTimeoutError
    ) -> JSONResponse:
        """Handle proxy timeout errors."""
        logger.error(
            "Proxy timeout error",
            error_type="proxy_timeout_error",
            error_message=str(exc),
            status_code=504,
            request_method=request.method,
            request_url=str(request.url.path),
        )

        # Record error in metrics
        if metrics:
            metrics.record_error(
                error_type="proxy_timeout_error",
                endpoint=str(request.url.path),
                model=None,
                service_type="middleware",
            )
        return JSONResponse(
            status_code=504,
            content={
                "error": {
                    "type": "proxy_timeout_error",
                    "message": str(exc),
                }
            },
        )

    @app.exception_handler(ProxyAuthenticationError)
    async def proxy_authentication_error_handler(
        request: Request, exc: ProxyAuthenticationError
    ) -> JSONResponse:
        """Handle proxy authentication errors."""
        logger.error(
            "Proxy authentication error",
            error_type="proxy_authentication_error",
            error_message=str(exc),
            status_code=401,
            request_method=request.method,
            request_url=str(request.url.path),
            client_ip=request.client.host if request.client else "unknown",
        )

        # Record error in metrics
        if metrics:
            metrics.record_error(
                error_type="proxy_authentication_error",
                endpoint=str(request.url.path),
                model=None,
                service_type="middleware",
            )
        return JSONResponse(
            status_code=401,
            content={
                "error": {
                    "type": "proxy_authentication_error",
                    "message": str(exc),
                }
            },
        )

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

        # Don't log stack trace for 404 errors as they're expected
        if exc.status_code == 404:
            logger.debug(
                "HTTP 404 error",
                error_type="http_404",
                error_message=exc.detail,
                status_code=404,
                request_method=request.method,
                request_url=str(request.url.path),
            )
        else:
            # Log with basic stack trace (no local variables)
            stack_trace = None
            # For structlog, we can always include traceback since structlog handles filtering
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
            )

        # Record error in metrics
        if metrics:
            error_type = "http_404" if exc.status_code == 404 else "http_error"
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
            )
        else:
            logger.error(
                "Starlette HTTP exception",
                error_type="starlette_http_error",
                error_message=exc.detail,
                status_code=exc.status_code,
                request_method=request.method,
                request_url=str(request.url.path),
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

    logger.debug("error_handlers_setup_completed")
