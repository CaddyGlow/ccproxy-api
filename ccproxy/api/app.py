"""FastAPI application factory for CCProxy API Server."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
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
from ccproxy.api.routes.confirmations import router as confirmations_router
from ccproxy.api.routes.health import router as health_router
from ccproxy.api.routes.mcp import setup_mcp
from ccproxy.api.routes.metrics import (
    dashboard_router,
    logs_router,
    prometheus_router,
)
from ccproxy.api.routes.proxy import router as proxy_router
from ccproxy.api.services.confirmation_service import get_confirmation_service
from ccproxy.api.ui.terminal_confirmation_handler import TerminalConfirmationHandler
from ccproxy.auth.credentials_adapter import CredentialsAuthManager
from ccproxy.auth.exceptions import CredentialsNotFoundError
from ccproxy.auth.oauth.routes import router as oauth_router
from ccproxy.config.settings import Settings, get_settings
from ccproxy.core.logging import setup_logging
from ccproxy.observability import get_metrics
from ccproxy.observability.storage.duckdb_simple import SimpleDuckDBStorage
from ccproxy.scheduler.manager import start_scheduler, stop_scheduler
from ccproxy.services.claude_sdk_service import ClaudeSDKService
from ccproxy.services.credentials import CredentialsManager


logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    settings = get_settings()

    # Store settings in app state for reuse in dependencies
    app.state.settings = settings

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

    # Validate authentication token at startup
    try:
        credentials_manager = CredentialsManager()
        validation = await credentials_manager.validate()

        if validation.valid and not validation.expired:
            credentials = validation.credentials
            oauth_token = credentials.claude_ai_oauth if credentials else None

            if oauth_token and oauth_token.expires_at_datetime:
                hours_until_expiry = int(
                    (
                        oauth_token.expires_at_datetime - datetime.now(UTC)
                    ).total_seconds()
                    / 3600
                )
                logger.info(
                    "auth_token_valid",
                    expires_in_hours=hours_until_expiry,
                    subscription_type=oauth_token.subscription_type,
                    credentials_path=str(validation.path) if validation.path else None,
                )
            else:
                logger.info("auth_token_valid", credentials_path=str(validation.path))
        elif validation.expired:
            logger.warning(
                "auth_token_expired",
                message="Authentication token has expired. Please run 'ccproxy auth login' to refresh.",
                credentials_path=str(validation.path) if validation.path else None,
            )
        else:
            logger.warning(
                "auth_token_invalid",
                message="Authentication token is invalid. Please run 'ccproxy auth login'.",
                credentials_path=str(validation.path) if validation.path else None,
            )
    except CredentialsNotFoundError:
        logger.warning(
            "auth_token_not_found",
            message="No authentication credentials found. Please run 'ccproxy auth login' to authenticate.",
            searched_paths=settings.auth.storage.storage_paths,
        )
    except Exception as e:
        logger.error(
            "auth_token_validation_error",
            error=str(e),
            message="Failed to validate authentication token. The server will continue without authentication.",
        )

    # Validate Claude binary at startup
    claude_path, found_in_path = settings.claude.find_claude_cli()
    if claude_path:
        logger.info(
            "claude_binary_found",
            path=claude_path,
            found_in_path=found_in_path,
            message=f"Claude CLI binary found at: {claude_path}",
        )
    else:
        searched_paths = settings.claude.get_searched_paths()
        logger.warning(
            "claude_binary_not_found",
            message="Claude CLI binary not found. Please install Claude CLI to use SDK features.",
            searched_paths=searched_paths,
            install_command="npm install -g @anthropic-ai/claude-code",
        )

    # Initialize ClaudeSDKService and store in app state
    try:
        # Create auth manager with settings
        auth_manager = CredentialsAuthManager()

        # Get global metrics instance
        metrics = get_metrics()

        # Create ClaudeSDKService instance
        claude_service = ClaudeSDKService(
            auth_manager=auth_manager,
            metrics=metrics,
            settings=settings,
        )

        # Store in app state for reuse in dependencies
        app.state.claude_service = claude_service
        logger.debug("claude_sdk_service_initialized")
    except Exception as e:
        logger.error("claude_sdk_service_initialization_failed", error=str(e))
        # Continue startup even if ClaudeSDKService fails (graceful degradation)

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

    # Initialize confirmation service
    try:
        logger.info("Initializing confirmation service...")
        confirmation_service = get_confirmation_service()

        # Only connect terminal handler if not using external handler
        if settings.server.use_terminal_confirmation_handler:
            terminal_handler = TerminalConfirmationHandler()

            # TODO: Terminal handler should subscribe to events from the service
            # instead of trying to set a handler directly
            # The service uses an event-based architecture, not direct handlers

            logger.info(
                "confirmation_handler_configured",
                handler_type="terminal",
                message="Connected terminal handler to confirmation service",
            )
            app.state.terminal_handler = terminal_handler
        else:
            logger.info(
                "confirmation_handler_configured",
                handler_type="external_sse",
                message="Terminal confirmation handler disabled - use 'ccproxy confirmation-handler connect' to handle confirmations",
            )
            logger.warning(
                "confirmation_handler_required",
                message="No confirmation handler active. Start external handler with: ccproxy confirmation-handler connect",
            )

        # Start the confirmation service
        await confirmation_service.start()

        # Store references in app state
        app.state.confirmation_service = confirmation_service

        logger.info(
            "confirmation_service_initialized",
            timeout_seconds=confirmation_service._timeout_seconds,
            terminal_handler_enabled=settings.server.use_terminal_confirmation_handler,
        )
    except Exception as e:
        logger.error("confirmation_service_initialization_failed", error=str(e))
        # Continue without confirmation service (API will work but without prompts)

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

    # Stop confirmation service
    if hasattr(app.state, "confirmation_service") and app.state.confirmation_service:
        try:
            await app.state.confirmation_service.stop()
            logger.debug("confirmation_service_stopped")
        except Exception as e:
            logger.error("confirmation_service_stop_failed", error=str(e))

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
        title="CCProxy API Server",
        description="High-performance API server providing Anthropic and OpenAI-compatible interfaces for Claude AI models",
        version=__version__,
        lifespan=lifespan,
    )

    # Setup MCP BEFORE adding any middleware
    # This is necessary because MCP uses Server-Sent Events (SSE) which don't work well
    # with certain middleware like BaseHTTPMiddleware
    setup_mcp(app)

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

    # Confirmation endpoints for SSE streaming and responses
    app.include_router(confirmations_router, tags=["confirmations"])

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
