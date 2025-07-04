"""Main entry point for Claude Proxy API Server."""

import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
import uvicorn

from claude_code_proxy.config.settings import get_settings
from claude_code_proxy.utils.claude_wrapper import create_claude_wrapper


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

        typer.echo(
            f"Starting Claude Code Proxy API Server on {server_host}:{server_port}"
        )
        typer.echo(f"Log level: {server_log_level}")

        # Log Claude CLI configuration
        if settings.claude_cli_path:
            typer.echo(f"Claude CLI path: {settings.claude_cli_path}")
        else:
            typer.echo("Claude CLI path: Auto-detect")
            typer.echo("Auto-detection will search:")
            for path in settings.get_searched_paths():
                typer.echo(f"  - {path}")

        if reload:
            typer.echo("Auto-reload enabled for development")

        # Start the server
        uvicorn.run(
            "claude_code_proxy.main:app",
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
        typer.echo(f"  Claude CLI Path: {settings.claude_cli_path or 'Auto-detect'}")
        typer.echo(f"  Workers: {settings.workers}")
        typer.echo(f"  Reload: {settings.reload}")

    except Exception as e:
        typer.echo(f"Error loading configuration: {e}", err=True)
        raise typer.Exit(1) from e


@app.command()
def claude(
    args: list[str] = typer.Argument(
        help="Arguments to pass to claude CLI (e.g. --version, doctor, config)"
    ),
    timeout: int = typer.Option(
        30,
        "--timeout",
        "-t",
        help="Command timeout in seconds",
    ),
) -> None:
    """
    Execute claude CLI commands directly.

    This is a simple pass-through to the claude CLI executable
    found by the settings system.

    Examples:
        python main.py claude -- --version
        python main.py claude -- doctor
        python main.py claude -- config
    """
    try:
        # Load settings to find claude path
        settings = get_settings()

        # Get claude path
        claude_path = settings.claude_cli_path
        if not claude_path:
            typer.echo("Error: Claude CLI not found.", err=True)
            typer.echo(
                "Please install Claude CLI or configure claude_cli_path.", err=True
            )
            raise typer.Exit(1)

        # Resolve to absolute path
        if not Path(claude_path).is_absolute():
            claude_path = str(Path(claude_path).resolve())

        typer.echo(f"Executing: {claude_path} {' '.join(args)}")
        typer.echo("")

        # Execute command directly
        full_cmd = [claude_path] + args

        try:
            result = subprocess.run(
                full_cmd,
                timeout=timeout,
                cwd=Path.cwd(),
                env=None,  # Use current environment
            )

            # Exit with same code as claude
            if result.returncode != 0:
                raise typer.Exit(result.returncode)

        except subprocess.TimeoutExpired as e:
            typer.echo(f"Command timed out after {timeout} seconds", err=True)
            raise typer.Exit(1) from e
        except KeyboardInterrupt as e:
            typer.echo("Command interrupted by user", err=True)
            raise typer.Exit(130) from e

    except Exception as e:
        typer.echo(f"Error executing claude command: {e}", err=True)
        raise typer.Exit(1) from e


if __name__ == "__main__":
    app()
