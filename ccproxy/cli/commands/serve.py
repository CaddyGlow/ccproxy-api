"""Serve command for CCProxy API server - consolidates server-related commands."""

from pathlib import Path
from typing import Annotated, Any

import typer
import uvicorn
from click import get_current_context
from rich.console import Console
from rich.syntax import Syntax

from ccproxy.cli.helpers import get_rich_toolkit
from ccproxy.config.settings import ConfigurationError, Settings
from ccproxy.core.logging import get_logger, setup_logging

from ..options.security_options import validate_auth_token
from ..options.server_options import (
    validate_log_level,
    validate_port,
)


def get_config_path_from_context() -> Path | None:
    """Get config path from typer context if available."""
    try:
        ctx = get_current_context()
        if ctx and ctx.obj and "config_path" in ctx.obj:
            config_path = ctx.obj["config_path"]
            return config_path if config_path is None else Path(config_path)
    except RuntimeError:
        pass
    return None


def _show_api_usage_info(toolkit: Any, settings: Settings) -> None:
    """Show API usage information when auth token is configured."""

    toolkit.print_title("API Client Configuration", tag="config")

    anthropic_base_url = f"http://{settings.server.host}:{settings.server.port}/claude"
    openai_base_url = f"http://{settings.server.host}:{settings.server.port}/codex"

    toolkit.print("Environment Variables for API Clients:", tag="info")
    toolkit.print_line()

    console = Console()

    auth_token = "YOUR_AUTH_TOKEN" if settings.security.auth_token else "NOT_SET"
    exports = f"""export ANTHROPIC_API_KEY={auth_token}
export ANTHROPIC_BASE_URL={anthropic_base_url}
export OPENAI_API_KEY={auth_token}
export OPENAI_BASE_URL={openai_base_url}"""

    console.print(Syntax(exports, "bash", theme="monokai", background_color="default"))
    toolkit.print_line()


def _run_local_server(settings: Settings) -> None:
    """Run the server locally."""
    # in_docker = is_running_in_docker()
    toolkit = get_rich_toolkit()
    logger = get_logger(__name__)

    if settings.security.auth_token:
        _show_api_usage_info(toolkit, settings)

    logger.debug(
        "server_starting",
        host=settings.server.host,
        port=settings.server.port,
        url=f"http://{settings.server.host}:{settings.server.port}",
    )

    reload_includes = None
    if settings.server.reload:
        reload_includes = ["ccproxy", "pyproject.toml", "uv.lock", "plugins"]

    # container = create_service_container(settings)

    uvicorn.run(
        # app=create_app(container),
        app="ccproxy.api.app:create_app",
        factory=True,
        host=settings.server.host,
        port=settings.server.port,
        reload=settings.server.reload,
        workers=settings.server.workers,
        log_config=None,
        access_log=False,
        server_header=False,
        date_header=False,
        reload_includes=reload_includes,
    )


def api(
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Path to configuration file (TOML, JSON, or YAML)",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            rich_help_panel="Configuration",
        ),
    ] = None,
    port: Annotated[
        int | None,
        typer.Option(
            "--port",
            "-p",
            help="Port to run the server on",
            callback=validate_port,
            rich_help_panel="Server Settings",
        ),
    ] = None,
    host: Annotated[
        str | None,
        typer.Option(
            "--host",
            "-h",
            help="Host to bind the server to",
            rich_help_panel="Server Settings",
        ),
    ] = None,
    reload: Annotated[
        bool | None,
        typer.Option(
            "--reload/--no-reload",
            help="Enable auto-reload for development",
            rich_help_panel="Server Settings",
        ),
    ] = None,
    log_level: Annotated[
        str | None,
        typer.Option(
            "--log-level",
            help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Use WARNING for minimal output.",
            callback=validate_log_level,
            rich_help_panel="Server Settings",
        ),
    ] = None,
    log_file: Annotated[
        str | None,
        typer.Option(
            "--log-file",
            help="Path to JSON log file. If specified, logs will be written to this file in JSON format",
            rich_help_panel="Server Settings",
        ),
    ] = None,
    auth_token: Annotated[
        str | None,
        typer.Option(
            "--auth-token",
            help="Bearer token for API authentication",
            callback=validate_auth_token,
            rich_help_panel="Security Settings",
        ),
    ] = None,
    enable_plugin: Annotated[
        list[str] | None,
        typer.Option(
            "--enable-plugin",
            help="Enable a plugin by name (repeatable)",
            rich_help_panel="Plugin Settings",
        ),
    ] = None,
    disable_plugin: Annotated[
        list[str] | None,
        typer.Option(
            "--disable-plugin",
            help="Disable a plugin by name (repeatable)",
            rich_help_panel="Plugin Settings",
        ),
    ] = None,
    # Removed unused flags: plugin_setting, no_network_calls,
    # disable_version_check, disable_pricing_updates
) -> None:
    """Start the CCProxy API server."""
    try:
        if config is None:
            config = get_config_path_from_context()

        # Base CLI context; plugin-injected args merged below
        cli_context = {
            "port": port,
            "host": host,
            "reload": reload,
            "log_level": log_level,
            "log_file": log_file,
            "auth_token": auth_token,
            "enabled_plugins": enable_plugin,
            "disabled_plugins": disable_plugin,
        }

        # Merge plugin-provided CLI args via helper
        try:
            from ccproxy.cli.helpers import get_plugin_cli_args

            plugin_args = get_plugin_cli_args()
            if plugin_args:
                cli_context.update(plugin_args)
        except Exception:
            pass

        # Pass CLI context to settings creation
        settings = Settings.from_config(config_path=config, cli_context=cli_context)

        setup_logging(
            json_logs=settings.logging.format == "json",
            log_level_name=settings.logging.level,
            log_file=settings.logging.file,
        )

        logger = get_logger(__name__)

        logger.debug(
            "configuration_loaded",
            host=settings.server.host,
            port=settings.server.port,
            log_level=settings.logging.level,
            log_file=settings.logging.file,
            auth_enabled=bool(settings.security.auth_token),
            duckdb_enabled=bool(
                (settings.plugins.get("duckdb_storage") or {}).get("enabled", False)
            ),
        )

        _run_local_server(settings)

    except ConfigurationError as e:
        toolkit = get_rich_toolkit()
        toolkit.print(f"Configuration error: {e}", tag="error")
        raise typer.Exit(1) from e
    except OSError as e:
        toolkit = get_rich_toolkit()
        toolkit.print(
            f"Server startup failed (port/permission issue): {e}", tag="error"
        )
        raise typer.Exit(1) from e
    except ImportError as e:
        toolkit = get_rich_toolkit()
        toolkit.print(f"Import error during server startup: {e}", tag="error")
        raise typer.Exit(1) from e
    except Exception as e:
        toolkit = get_rich_toolkit()
        toolkit.print(f"Error starting server: {e}", tag="error")
        raise typer.Exit(1) from e
