"""FastAPI application factory for Claude Code Proxy API Server."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from ccproxy.api.middleware.cors import setup_cors_middleware
from ccproxy.api.middleware.errors import setup_error_handlers
from ccproxy.api.middleware.logging import AccessLogMiddleware
from ccproxy.api.routes.claude import router as claude_router
from ccproxy.api.routes.health import router as health_router
from ccproxy.api.routes.metrics import router as metrics_router
from ccproxy.api.routes.proxy import router as proxy_router
from ccproxy.config.settings import Settings, get_settings
from ccproxy.core.logging import get_logger
from ccproxy.observability.config import configure_observability
from ccproxy.observability.pipeline import get_pipeline
from ccproxy.observability.scheduler import get_scheduler, stop_scheduler


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

    # Configure observability system (structlog + pipeline + scheduler)
    try:
        # Determine format based on log level - Rich for DEBUG, JSON for production
        format_type = "rich" if settings.server.log_level.upper() == "DEBUG" else "json"
        show_path = settings.server.log_level.upper() == "DEBUG"

        # Configure Rich logging to reduce stack trace verbosity
        from ccproxy.core.logging import setup_rich_logging

        setup_rich_logging(
            level=settings.server.log_level,
            show_path=show_path,
            show_time=True,
            verbose_tracebacks=settings.server.log_level.upper() == "DEBUG",
        )

        # configure_observability(
        #     format_type=format_type,
        #     level=settings.server.log_level,
        #     show_path=show_path,
        #     show_time=True,
        # )
        pipeline = await get_pipeline()
        scheduler = await get_scheduler()
        logger.info("Observability system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize observability system: {e}")
        # Continue startup even if observability fails (graceful degradation)

    yield

    # Shutdown
    logger.info("Shutting down Claude Code Proxy API Server...")

    # Stop observability system
    try:
        if "pipeline" in locals():
            await pipeline.stop()
        if "scheduler" in locals():
            await scheduler.stop()
        # Also stop global scheduler
        await stop_scheduler()
        logger.info("Observability system stopped")
    except Exception as e:
        logger.error(f"Error stopping observability system: {e}")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Optional settings override. If None, uses get_settings().

    Returns:
        Configured FastAPI application instance.
    """
    if settings is None:
        settings = get_settings()

    app = FastAPI(
        title="Claude Code Proxy API Server",
        description="High-performance API server providing Anthropic and OpenAI-compatible interfaces for Claude AI models",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Setup middleware
    setup_cors_middleware(app, settings)
    setup_error_handlers(app)

    # Add custom access log middleware
    app.add_middleware(AccessLogMiddleware)

    # Include health router (always enabled)
    app.include_router(health_router, tags=["health"])

    # Include metrics router only if enabled
    if settings.observability.metrics_enabled:
        app.include_router(metrics_router, tags=["metrics"])

    # Include OAuth router for authentication flows
    from ccproxy.auth.oauth.routes import router as oauth_router

    app.include_router(oauth_router, prefix="/oauth", tags=["oauth"])

    # New /sdk/ routes for Claude SDK endpoints
    app.include_router(claude_router, prefix="/sdk", tags=["claude-sdk"])

    # New /api/ routes for proxy endpoints (includes OpenAI-compatible /v1/chat/completions)
    app.include_router(proxy_router, prefix="/api", tags=["proxy-api"])

    # Mount static files for dashboard SPA
    from pathlib import Path

    # Get the path to the dashboard static files
    current_file = Path(__file__)
    project_root = (
        current_file.parent.parent.parent
    )  # ccproxy/api/app.py -> project root
    dashboard_static_path = project_root / "ccproxy" / "static" / "dashboard"

    # Mount dashboard static files if they exist
    if dashboard_static_path.exists():
        # Mount the _app directory for SvelteKit assets at the correct base path
        app_path = dashboard_static_path / "_app"
        if app_path.exists():
            app.mount(
                "/metrics/dashboard/_app",
                StaticFiles(directory=str(app_path)),
                name="dashboard-assets",
            )

        # Mount favicon.svg at root level
        favicon_path = dashboard_static_path / "favicon.svg"
        if favicon_path.exists():
            # For single files, we'll handle this in the dashboard route or add a specific route
            pass

    return app


def get_app() -> FastAPI:
    """Get the FastAPI application instance.

    Returns:
        FastAPI application instance.
    """
    return create_app()
