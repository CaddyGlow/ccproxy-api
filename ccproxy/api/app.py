"""FastAPI application factory for Claude Code Proxy API Server."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from structlog import get_logger

from ccproxy import __version__
from ccproxy.api.middleware.cors import setup_cors_middleware
from ccproxy.api.middleware.errors import setup_error_handlers
from ccproxy.api.middleware.logging import AccessLogMiddleware
from ccproxy.api.middleware.request_id import RequestIDMiddleware
from ccproxy.api.middleware.server_header import ServerHeaderMiddleware
from ccproxy.api.routes.claude import router as claude_router
from ccproxy.api.routes.health import router as health_router
from ccproxy.api.routes.metrics import (
    dashboard_router,
    logs_router,
    prometheus_router,
)
from ccproxy.api.routes.proxy import router as proxy_router
from ccproxy.auth.oauth.routes import router as oauth_router
from ccproxy.config.settings import Settings, get_settings
from ccproxy.core.logging import setup_logging
from ccproxy.observability.storage.duckdb_simple import SimpleDuckDBStorage
from ccproxy.scheduler.manager import start_scheduler, stop_scheduler


logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    settings = get_settings()

    # Startup
    logger.info(
        "server_start",
        host=settings.server.host,
        port=settings.server.port,
        url=f"http://{settings.server.host}:{settings.server.port}",
    )
    logger.debug(
        "server_configured", host=settings.server.host, port=settings.server.port
    )

    # Log Claude CLI configuration
    if settings.claude.cli_path:
        logger.debug("claude_cli_configured", cli_path=settings.claude.cli_path)
    else:
        logger.debug("claude_cli_auto_detect")
        logger.debug(
            "claude_cli_search_paths", paths=settings.claude.get_searched_paths()
        )

    # Start scheduler system
    try:
        scheduler = await start_scheduler(settings)
        app.state.scheduler = scheduler
        logger.debug("scheduler_initialized")
    except Exception as e:
        logger.error("scheduler_initialization_failed", error=str(e))
        # Continue startup even if scheduler fails (graceful degradation)

    # Initialize log storage if needed and backend is duckdb
    if (
        settings.observability.needs_storage_backend
        and settings.observability.log_storage_backend == "duckdb"
    ):
        try:
            storage = SimpleDuckDBStorage(
                database_path=settings.observability.duckdb_path
            )
            await storage.initialize()
            app.state.log_storage = storage
            logger.debug(
                "log_storage_initialized",
                backend="duckdb",
                path=str(settings.observability.duckdb_path),
                collection_enabled=settings.observability.logs_collection_enabled,
            )
        except Exception as e:
            logger.error("log_storage_initialization_failed", error=str(e))
            # Continue without log storage (graceful degradation)

    yield

    # Shutdown
    logger.debug("server_stop")

    # Stop scheduler system
    try:
        scheduler = getattr(app.state, "scheduler", None)
        await stop_scheduler(scheduler)
        logger.debug("scheduler_stopped")
    except Exception as e:
        logger.error("scheduler_stop_failed", error=str(e))

    # Close log storage if initialized
    if hasattr(app.state, "log_storage") and app.state.log_storage:
        try:
            await app.state.log_storage.close()
            logger.debug("log_storage_closed")
        except Exception as e:
            logger.error("log_storage_close_failed", error=str(e))


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Optional settings override. If None, uses get_settings().

    Returns:
        Configured FastAPI application instance.
    """
    if settings is None:
        settings = get_settings()

    # Configure logging based on settings BEFORE any module uses logger
    # This is needed for reload mode where the app is re-imported
    import logging

    import structlog

    from ccproxy.config.settings import config_manager

    # Only configure if not already configured or if no file handler exists
    root_logger = logging.getLogger()
    has_file_handler = any(
        isinstance(h, logging.FileHandler) for h in root_logger.handlers
    )

    if not structlog.is_configured() or not has_file_handler:
        # Only setup logging if not already configured with file handler
        # Always use console output
        json_logs = False
        # Don't override file logging if it was already configured
        if not has_file_handler:
            setup_logging(json_logs=json_logs, log_level=settings.server.log_level)

    app = FastAPI(
        title="Claude Code Proxy API Server",
        description="High-performance API server providing Anthropic and OpenAI-compatible interfaces for Claude AI models",
        version=__version__,
        lifespan=lifespan,
    )

    # Setup middleware
    setup_cors_middleware(app, settings)
    setup_error_handlers(app)

    # Add custom access log middleware first (will run second due to middleware order)
    app.add_middleware(AccessLogMiddleware)

    # Add request ID middleware second (will run first to initialize context)
    app.add_middleware(RequestIDMiddleware)

    # Add server header middleware (for non-proxy routes)
    # You can customize the server name here
    app.add_middleware(ServerHeaderMiddleware, server_name="uvicorn")

    # Include health router (always enabled)
    app.include_router(health_router, tags=["health"])

    # Include observability routers with granular controls
    if settings.observability.metrics_endpoint_enabled:
        app.include_router(prometheus_router, tags=["metrics"])

    if settings.observability.logs_endpoints_enabled:
        app.include_router(logs_router, tags=["logs"])

    if settings.observability.dashboard_enabled:
        app.include_router(dashboard_router, tags=["dashboard"])

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
                "/dashboard/_app",
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
