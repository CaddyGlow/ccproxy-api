"""Main FastAPI application for Claude Proxy API Server."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from claude_proxy.config.settings import get_settings


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
    logger.info("Starting Claude Proxy API Server...")
    settings = get_settings()
    logger.info(f"Server configured for host: {settings.host}, port: {settings.port}")

    # Configure secure Claude SDK with privilege dropping
    from claude_proxy.utils.secure_claude_sdk import configure_secure_claude_sdk
    configure_secure_claude_sdk(
        user=settings.claude_user,
        group=settings.claude_group,
        working_directory=settings.claude_working_directory,
    )
    logger.info(f"Configured secure Claude SDK with user: {settings.claude_user}, group: {settings.claude_group}, cwd: {settings.claude_working_directory}")

    yield

    # Shutdown
    logger.info("Shutting down Claude Proxy API Server...")


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

    # Include API routes
    from claude_proxy.api.openai import chat_router, models_router
    from claude_proxy.api.v1 import chat

    # Anthropic-compatible endpoints
    app.include_router(chat.router, prefix="/v1")

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
