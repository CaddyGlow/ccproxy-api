"""Main entry point for Claude Proxy API Server."""

import json
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

import fastapi_cli.discover
import typer
import uvicorn
from fastapi_cli.cli import app as fastapi_app
from fastapi_cli.exceptions import FastAPICLIException

from claude_code_proxy.config.settings import get_settings
from claude_code_proxy.utils.docker_builder import DockerCommandBuilder
from claude_code_proxy.utils.helper import get_package_dir


app = typer.Typer(rich_markup_mode="rich")
logger = logging.getLogger(__name__)
# Remove the fastapi callback to avoid the warning
fastapi_app.callback()(lambda: None)
app.add_typer(fastapi_app)


def get_default_path_hook() -> Path:
    potential_paths = (get_package_dir() / "claude_code_proxy" / "main.py",)

    for full_path in potential_paths:
        path = Path(full_path)
        if path.is_file():
            return path

    raise FastAPICLIException(
        "Could not find a default file to run, please provide an explicit path"
    )


fastapi_cli.discover.get_default_path = get_default_path_hook


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
    docker: bool = typer.Option(
        False,
        "--docker",
        "-d",
        help="Run claude command from docker image instead of local CLI",
    ),
    docker_image: str | None = typer.Option(
        None,
        "--docker-image",
        help="Docker image to use (overrides config)",
    ),
    docker_env: list[str] = typer.Option(
        [],
        "--docker-env",
        help="Environment variables to pass to Docker (KEY=VALUE format, can be used multiple times)",
    ),
    docker_volume: list[str] = typer.Option(
        [],
        "--docker-volume",
        help="Volume mounts to add (host:container[:options] format, can be used multiple times)",
    ),
    docker_arg: list[str] = typer.Option(
        [],
        "--docker-arg",
        help="Additional Docker run arguments (can be used multiple times)",
    ),
    docker_home: str | None = typer.Option(
        None,
        "--docker-home",
        help="Home directory inside Docker container (overrides config)",
    ),
    docker_workspace: str | None = typer.Option(
        None,
        "--docker-workspace",
        help="Workspace directory inside Docker container (overrides config)",
    ),
) -> None:
    """
    Execute claude CLI commands directly.

    This is a simple pass-through to the claude CLI executable
    found by the settings system or run from docker image.

    Examples:
        ccproxy claude -- --version
        ccproxy claude -- doctor
        ccproxy claude -- config
        ccproxy claude --docker -- --version
        ccproxy claude --docker --docker-image custom:latest -- --version
        ccproxy claude --docker --docker-env API_KEY=sk-... --docker-volume ./data:/data -- chat
    """
    try:
        if docker:
            # Load settings to get Docker configuration
            settings = get_settings()

            # Build Docker command with settings and CLI overrides
            docker_cmd = DockerCommandBuilder.from_settings_and_overrides(
                settings.docker_settings,
                args,
                docker_image=docker_image,
                docker_env=docker_env,
                docker_volume=docker_volume,
                docker_arg=docker_arg,
                docker_home=docker_home,
                docker_workspace=docker_workspace,
            )

            typer.echo(f"Executing: {' '.join(docker_cmd)}")
            typer.echo("")

            full_cmd = docker_cmd
        else:
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
            # Use os.execvp to replace current process with claude
            # This hands over full control to claude, including signal handling
            os.execvp(full_cmd[0], full_cmd)
        except OSError as e:
            typer.echo(f"Failed to execute command: {e}", err=True)
            raise typer.Exit(1) from e

    except Exception as e:
        typer.echo(f"Error executing claude command: {e}", err=True)
        raise typer.Exit(1) from e


if __name__ == "__main__":
    app()
