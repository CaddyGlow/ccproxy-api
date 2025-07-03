"""Main entry point for Claude Proxy API Server."""

import logging
from typing import Optional

import typer
import uvicorn

from claude_proxy.config.settings import get_settings


app = typer.Typer()
logger = logging.getLogger(__name__)


@app.command()
def serve(
    host: str = typer.Option(
        None,
        "--host",
        "-h",
        help="Host to bind the server to",
    ),
    port: int = typer.Option(
        None,
        "--port",
        "-p",
        help="Port to bind the server to",
    ),
    reload: bool = typer.Option(
        False,
        "--reload",
        "-r",
        help="Enable auto-reload for development",
    ),
    log_level: str = typer.Option(
        None,
        "--log-level",
        "-l",
        help="Logging level",
    ),
) -> None:
    """
    Start the Claude Proxy API Server.

    This command starts the FastAPI server with uvicorn, providing
    an Anthropic-compatible API interface for Claude AI models.

    Args:
        host: Host to bind the server to (overrides config)
        port: Port to bind the server to (overrides config)
        reload: Enable auto-reload for development
        log_level: Logging level (overrides config)
    """
    try:
        # Load settings
        settings = get_settings()

        # Override settings with command line args
        server_host = host if host is not None else settings.host
        server_port = port if port is not None else settings.port
        server_log_level = (
            log_level if log_level is not None else settings.log_level.lower()
        )

        # Configure logging
        logging.basicConfig(
            level=getattr(logging, server_log_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        typer.echo(f"Starting Claude Proxy API Server on {server_host}:{server_port}")
        typer.echo(f"Log level: {server_log_level}")

        if reload:
            typer.echo("Auto-reload enabled for development")

        # Start the server
        uvicorn.run(
            "claude_proxy.main:app",
            host=server_host,
            port=server_port,
            log_level=server_log_level,
            reload=reload,
        )

    except Exception as e:
        typer.echo(f"Error starting server: {e}", err=True)
        raise typer.Exit(1) from e


@app.command()
def config() -> None:
    """Show current configuration."""
    try:
        settings = get_settings()
        typer.echo("Current Configuration:")
        typer.echo(f"  Host: {settings.host}")
        typer.echo(f"  Port: {settings.port}")
        typer.echo(f"  Log Level: {settings.log_level}")
        typer.echo(f"  API Key: {'***' if settings.anthropic_api_key else 'NOT SET'}")
        typer.echo(f"  Claude CLI Path: {settings.claude_cli_path or 'Auto-detect'}")
        typer.echo(f"  Workers: {settings.workers}")
        typer.echo(f"  Reload: {settings.reload}")

    except Exception as e:
        typer.echo(f"Error loading configuration: {e}", err=True)
        raise typer.Exit(1) from e


if __name__ == "__main__":
    app()
