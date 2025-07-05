"""Main FastAPI application for Claude Proxy API Server."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from claude_code_proxy.config.settings import get_settings
from claude_code_proxy.services.pool_manager import pool_manager


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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

    # Initialize connection pool if enabled
    logger.info("[STARTUP] Configuring connection pool manager...")
    pool_manager.configure(settings)
    if settings.pool_settings.enabled:
        logger.info(
            f"[STARTUP] Initializing Claude connection pool (min={settings.pool_settings.min_size}, "
            f"max={settings.pool_settings.max_size})"
        )
        await pool_manager.initialize()
        logger.info("[STARTUP] Claude connection pool initialized successfully")
    else:
        logger.info("[STARTUP] Claude connection pooling is disabled")

    yield

    # Shutdown
    logger.info("Shutting down Claude Proxy API Server...")

    # Shutdown connection pool
    if pool_manager.is_enabled:
        logger.info("[SHUTDOWN] Shutting down Claude connection pool...")
        await pool_manager.shutdown()
        logger.info("[SHUTDOWN] Claude connection pool shut down successfully")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
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
        allow_origins=["*"],  # Configure as needed for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health check endpoint
    @app.get("/health")
    async def health_check() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "healthy", "service": "claude-proxy"}

    # Pool statistics endpoint
    @app.get("/pool/stats")
    async def pool_stats() -> dict[str, Any]:
        """Get connection pool statistics."""
        return {
            "pool_enabled": pool_manager.is_enabled,
            "stats": pool_manager.get_stats() if pool_manager.is_enabled else None,
        }

    # Include API routes
    from claude_code_proxy.api.openai import chat_router, models_router
    from claude_code_proxy.api.v1 import chat, messages

    # Anthropic-compatible endpoints
    app.include_router(chat.router, prefix="/v1")
    app.include_router(messages.router, prefix="/v1")

    # OpenAI-compatible endpoints
    app.include_router(chat_router, prefix="/openai/v1")
    app.include_router(models_router, prefix="/openai/v1")

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


# Create the app instance
app = create_app()
