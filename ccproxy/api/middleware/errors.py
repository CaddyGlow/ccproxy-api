"""Error handling middleware for Claude Code Proxy API Server."""

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

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


logger = get_logger(__name__)


def setup_error_handlers(app: FastAPI) -> None:
    """Setup error handlers for the FastAPI application.

    Args:
        app: FastAPI application instance
    """
    logger.debug("Setting up error handlers")

    @app.exception_handler(ClaudeProxyError)
    async def claude_proxy_error_handler(
        request: Request, exc: ClaudeProxyError
    ) -> JSONResponse:
        """Handle Claude proxy specific errors."""
        logger.error(f"Claude proxy error: {exc}")
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
        logger.error(f"Validation error: {exc}")
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
        logger.error(f"Authentication error: {exc}")
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
        logger.error(f"Permission error: {exc}")
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
        logger.error(f"Not found error: {exc}")
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
        logger.error(f"Rate limit error: {exc}")
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
        logger.error(f"Model not found error: {exc}")
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
        logger.error(f"Timeout error: {exc}")
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
        logger.error(f"Service unavailable error: {exc}")
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
        logger.error(f"Docker error: {exc}")
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
        logger.error(f"Proxy error: {exc}")
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
        logger.error(f"Transformation error: {exc}")
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
        logger.error(f"Middleware error: {exc}")
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
        logger.error(f"Proxy connection error: {exc}")
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
        logger.error(f"Proxy timeout error: {exc}")
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
        logger.error(f"Proxy authentication error: {exc}")
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
        logger.error(
            f"HTTP exception: {exc.status_code} - {exc.detail}",
            exc_info=logger.isEnabledFor(logging.DEBUG),
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
        logger.error(
            f"Starlette HTTP exception: {exc.status_code} - {exc.detail}",
            exc_info=logger.isEnabledFor(logging.DEBUG),
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
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "type": "internal_server_error",
                    "message": "An internal server error occurred",
                }
            },
        )

    logger.info("Error handlers configured successfully")
