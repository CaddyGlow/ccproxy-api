"""Main FastAPI application for Claude Proxy API Server."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from ccproxy.config.settings import Settings, get_settings
from ccproxy.utils.logging import get_logger, setup_rich_logging


# Get settings first to determine log level
settings = get_settings()

# Configure rich logging with settings
setup_rich_logging(level=settings.log_level)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Claude Code Proxy API Server...")
    settings = get_settings()
    logger.info(f"Server configured for host: {settings.host}, port: {settings.port}")

    # Log Claude CLI configuration
    if settings.claude_cli_path:
        logger.info(f"Claude CLI configured at: {settings.claude_cli_path}")
    else:
        logger.info("Claude CLI path: Auto-detect at runtime")
        logger.info("Auto-detection will search the following locations:")
        for path in settings.get_searched_paths():
            logger.info(f"  - {path}")

    # Initialize metrics storage if enabled
    if settings.metrics_enabled:
        from pathlib import Path

        from ccproxy.metrics.sync_storage import get_sync_metrics_storage

        # Create metrics database directory if it doesn't exist
        metrics_db_path = Path(settings.metrics_db_path)
        metrics_db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize storage
        storage = get_sync_metrics_storage(f"sqlite:///{metrics_db_path}")
        storage.initialize()
        logger.info(f"Metrics storage initialized at: {metrics_db_path}")

        # Store storage instance for later use
        app.state.metrics_storage = storage

    yield

    # Shutdown
    logger.info("Shutting down Claude Proxy API Server...")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if settings is None:
        settings = get_settings()

    app = FastAPI(
        title="Claude Proxy API Server",
        description="High-performance API server providing Anthropic-compatible interface for Claude AI models",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_credentials,
        allow_methods=settings.cors_methods,
        allow_headers=settings.cors_headers,
        allow_origin_regex=settings.cors_origin_regex,
        expose_headers=settings.cors_expose_headers,
        max_age=settings.cors_max_age,
    )

    # Add metrics middleware (if enabled)
    if settings.metrics_enabled:
        from ccproxy.metrics.middleware import MetricsMiddleware

        app.add_middleware(MetricsMiddleware)
        logger.info("Metrics collection enabled")

    # Health check endpoint
    @app.get("/health")
    async def health_check() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "healthy", "service": "claude-proxy"}

    # Include API routes
    from ccproxy.routers import (
        anthropic_router,
        create_reverse_proxy_router,
        metrics_router,
        oauth_router,
        openai_router,
    )

    # Claude Code SDK endpoints (local execution) - using standard Anthropic format
    app.include_router(anthropic_router, prefix=f"{settings.claude_code_prefix}/v1")
    app.include_router(openai_router, prefix=f"{settings.claude_code_prefix}/openai/v1")

    # OAuth authentication endpoints
    app.include_router(oauth_router)

    # Metrics endpoints (if enabled) - MUST be before reverse proxy
    if settings.metrics_enabled:
        app.include_router(metrics_router, prefix="/metrics")
        logger.info("Metrics API endpoints enabled")

        # Mount static files for dashboard CSS and JS
        from pathlib import Path

        static_path = Path(__file__).parent / "static"
        if static_path.exists():
            app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
            logger.info(f"Static files mounted from: {static_path}")

    # Reverse proxy endpoints with different modes
    app.include_router(create_reverse_proxy_router("minimal"), prefix="/min")
    app.include_router(create_reverse_proxy_router("full"), prefix="/api")

    # Default reverse proxy for root path - MUST be last as it has empty prefix
    app.include_router(
        create_reverse_proxy_router(settings.default_proxy_mode), prefix=""
    )

    # Legacy compatibility - old paths redirect to new ones
    # app.include_router(anthropic.router, prefix="/v1")
    # app.include_router(openai.router, prefix="/openai/v1")

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Any, exc: Exception) -> JSONResponse:
        """Global exception handler."""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "type": "internal_server_error",
                    "message": "Internal server error occurred",
                }
            },
        )

    return app


# Create the app instance for production use
app = create_app()
