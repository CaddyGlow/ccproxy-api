"""FastAPI development commands."""

import logging
from typing import Optional

import typer
import uvicorn

from claude_code_proxy.config.settings import ConfigurationError, config_manager
from claude_code_proxy.utils.cli import get_rich_toolkit, get_uvicorn_log_config
from claude_code_proxy.utils.helper import get_root_package_name
from claude_code_proxy.utils.logging import get_logger

from .api import get_config_path_from_context


# Create FastAPI sub-application
app = typer.Typer(help="FastAPI development commands")

# Logger will be configured by configuration manager
logger = get_logger(__name__)


def _run(
    command: str,
    host: str,
    port: int,
    reload: bool,
    app: str = f"{get_root_package_name()}:app",
    workers: int | None = None,
) -> None:
    """Run FastAPI with centralized configuration management."""
    try:
        # Load settings and configure logging
        settings = config_manager.load_settings(
            config_path=get_config_path_from_context()
        )
        config_manager.setup_logging()

        logger.info(f"Starting {command} server...")
        # logger.info(f"Server will run at http://{host}:{port}")

        # Run uvicorn with our already configured logging
        uvicorn.run(
            app=app,
            host=host,
            port=port,
            reload=reload,
            workers=workers,
            log_config=get_uvicorn_log_config(),
        )
    except ConfigurationError as e:
        toolkit = get_rich_toolkit()
        toolkit.print(f"Configuration error: {e}", tag="error")
        raise typer.Exit(1) from e
    except Exception as e:
        toolkit = get_rich_toolkit()
        toolkit.print(f"Error starting {command} server: {e}", tag="error")
        raise typer.Exit(1) from e


@app.command()
def run(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    # workers: int = typer.Option(1, "--workers", help="Number of worker processes"),
    reload: bool = typer.Option(
        False, "--reload/--no-reload", help="Enable auto-reload"
    ),
) -> None:
    """Run a FastAPI app in production mode."""
    _run("production", host, port, reload)


@app.command()
def dev(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(
        True, "--reload/--no-reload", help="Enable auto-reload"
    ),
) -> None:
    """Run a FastAPI app in development mode."""
    _run("development", host, port, reload)
