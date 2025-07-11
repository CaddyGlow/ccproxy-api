"""FastAPI application factory for Claude Code Proxy API Server."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from ccproxy.api.middleware.cors import setup_cors_middleware
from ccproxy.api.middleware.errors import setup_error_handlers
from ccproxy.api.routes.anthropic import router as anthropic_router
from ccproxy.api.routes.claude import router as claude_router
from ccproxy.api.routes.health import router as health_router
from ccproxy.api.routes.metrics import router as metrics_router
from ccproxy.api.routes.openai import router as openai_router
from ccproxy.api.routes.proxy import router as proxy_router
from ccproxy.config.settings import Settings, get_settings
from ccproxy.core.logging import get_logger, setup_rich_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    settings = get_settings()
    logger = get_logger(__name__)

    # Startup
    logger.info("Starting Claude Code Proxy API Server...")
    logger.info(
        f"Server configured for host: {settings.server.host}, port: {settings.server.port}"
    )

    # Log Claude CLI configuration
    if settings.claude.cli_path:
        logger.info(f"Claude CLI configured at: {settings.claude.cli_path}")
    else:
        logger.info("Claude CLI path: Auto-detect at runtime")
        logger.info("Auto-detection will search the following locations:")
        for path in settings.claude.get_searched_paths():
            logger.info(f"  - {path}")

    yield

    # Shutdown
    logger.info("Shutting down Claude Code Proxy API Server...")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Optional settings override. If None, uses get_settings().

    Returns:
        Configured FastAPI application instance.
    """
    if settings is None:
        settings = get_settings()

    # Configure rich logging with settings
    setup_rich_logging(level=settings.server.log_level)

    app = FastAPI(
        title="Claude Code Proxy API Server",
        description="High-performance API server providing Anthropic and OpenAI-compatible interfaces for Claude AI models",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Setup middleware
    setup_cors_middleware(app, settings)
    setup_error_handlers(app)

    # Include health and metrics routers (keep as they are)
    app.include_router(health_router, tags=["health"])
    app.include_router(metrics_router, prefix="/metrics", tags=["metrics"])

    # New /sdk/ routes for Claude SDK endpoints
    app.include_router(claude_router, prefix="/sdk", tags=["claude-sdk"])

    # New /api/ routes for proxy endpoints (includes OpenAI-compatible /v1/chat/completions)
    app.include_router(proxy_router, prefix="/api", tags=["proxy-api"])

    # Legacy OpenAI-compatible API endpoints (both standard and prefixed paths)
    # These will be simplified or removed in favor of /sdk/ and /api/ routes
    app.include_router(openai_router, prefix="/v1", tags=["openai-standard"])
    app.include_router(openai_router, prefix="/openai/v1", tags=["openai"])

    # Legacy Anthropic API endpoints (for non-OpenAI paths under /v1)
    # These will be simplified or removed in favor of /sdk/ and /api/ routes
    app.include_router(anthropic_router, prefix="/v1", tags=["anthropic"])

    return app


def get_app() -> FastAPI:
    """Get the FastAPI application instance.

    Returns:
        FastAPI application instance.
    """
    return create_app()
