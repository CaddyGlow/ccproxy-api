"""Main entry point for Claude Proxy API Server."""

import logging
import os
import secrets
from pathlib import Path
from typing import Optional

import fastapi_cli.discover
import typer
import uvicorn
from click import get_current_context
from fastapi_cli.cli import _run
from fastapi_cli.cli import app as fastapi_app
from fastapi_cli.exceptions import FastAPICLIException

from claude_code_proxy._version import __version__
from claude_code_proxy.commands.config import app as config_app
from claude_code_proxy.config.settings import get_settings
from claude_code_proxy.utils.docker_builder import DockerCommandBuilder
from claude_code_proxy.utils.helper import get_package_dir


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        typer.echo(f"claude-code-proxy-api {__version__}")
        raise typer.Exit()


app = typer.Typer(
    rich_markup_mode="rich",
    add_completion=False,
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)
logger = logging.getLogger(__name__)


# Add global --version option
@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file (TOML, JSON, or YAML)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Claude Code Proxy API Server - Anthropic and OpenAI compatible interface for Claude."""
    # Store config path in context for use by commands
    try:
        ctx = get_current_context()
        ctx.ensure_object(dict)
        ctx.obj["config_path"] = config
    except RuntimeError:
        # No active click context (e.g., in tests)
        pass


# Remove the fastapi callback to avoid the warning
# fastapi_app.callback()(lambda: None)
fastapi_app.callback()(None)  # type: ignore[type-var]
# Register fastapi app with typer
app.add_typer(fastapi_app)

# Register config command
app.add_typer(config_app)


def get_default_path_hook() -> Path:
    app_entry_path = get_package_dir() / "claude_code_proxy" / "main.py"
    if app_entry_path.is_file():
        return app_entry_path

    raise FastAPICLIException(
        "Could not find a default file to run, please provide an explicit path"
    )


def get_config_path_from_context() -> Path | None:
    """Get config path from typer context if available."""
    try:
        ctx = get_current_context()
        if ctx and ctx.obj and "config_path" in ctx.obj:
            config_path = ctx.obj["config_path"]
            return config_path if config_path is None else Path(config_path)
    except RuntimeError:
        # No active click context (e.g., in tests)
        pass
    return None


fastapi_cli.discover.get_default_path = get_default_path_hook


@app.command()
def generate_token() -> None:
    """Generate a secure random token for API authentication."""
    token = secrets.token_urlsafe(32)
    typer.echo("Generated authentication token:")
    typer.echo(f"AUTH_TOKEN={token}")
    typer.echo("")
    typer.echo("Add this to your environment variables:")
    typer.echo(f"export AUTH_TOKEN={token}")
    typer.echo("")
    typer.echo("Or add to your .env file:")
    typer.echo(f"AUTH_TOKEN={token}")


@app.command()
def api(
    docker: bool = typer.Option(
        False,
        "--docker",
        "-d",
        help="Run API server using Docker instead of local execution",
    ),
    port: int = typer.Option(
        8000,
        "--port",
        "-p",
        help="Port to run the server on",
    ),
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        "-h",
        help="Host to bind the server to",
    ),
    reload: bool = typer.Option(
        False,
        "--reload",
        help="Enable auto-reload for development",
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
    user_mapping_enabled: bool | None = typer.Option(
        None,
        "--user-mapping/--no-user-mapping",
        help="Enable/disable UID/GID mapping (overrides config)",
    ),
    user_uid: int | None = typer.Option(
        None,
        "--user-uid",
        help="User ID to run container as (overrides config)",
        min=0,
    ),
    user_gid: int | None = typer.Option(
        None,
        "--user-gid",
        help="Group ID to run container as (overrides config)",
        min=0,
    ),
) -> None:
    """
    Start the Claude Code Proxy API server.

    This command starts the API server either locally or in Docker.
    The server provides both Anthropic and OpenAI-compatible endpoints.

    Examples:
        ccproxy run
        ccproxy run --port 8080 --reload
        ccproxy run --docker
        ccproxy run --docker --docker-image custom:latest --port 8080
    """
    try:
        if docker:
            # Load settings to get Docker configuration
            settings = get_settings(config_path=get_config_path_from_context())
            port = port if port is None else settings.port
            # Prepare server command using fastapi
            server_args = [
                "run",
                "--host",
                "0.0.0.0",  # Docker needs to bind to 0.0.0.0
                "--port",
                str(port),
            ]

            if reload:
                server_args.append("--reload")

            # Build and execute Docker command with settings and CLI overrides
            typer.echo("Starting Claude Code Proxy API server with Docker...")
            typer.echo(f"Server will be available at: http://{host}:{port}")

            # Show the command before executing
            docker_cmd = DockerCommandBuilder.from_settings_and_overrides(
                settings.docker_settings,
                docker_image=docker_image,
                docker_env=docker_env + [f"PORT={port}"],
                docker_volume=docker_volume,
                docker_arg=docker_arg + ["-p", f"{port}:{port}"],
                docker_home=docker_home,
                docker_workspace=docker_workspace,
                user_mapping_enabled=user_mapping_enabled,
                user_uid=user_uid,
                user_gid=user_gid,
                cmd_args=server_args,
            )
            typer.echo(f"Executing: {' '.join(docker_cmd)}")
            typer.echo("")

            # Execute using the new Docker builder method
            DockerCommandBuilder.execute_from_settings(
                settings.docker_settings,
                docker_image=docker_image,
                docker_env=docker_env + [f"PORT={port}"],
                docker_volume=docker_volume,
                docker_arg=docker_arg + ["-p", f"{port}:{port}"],
                docker_home=docker_home,
                docker_workspace=docker_workspace,
                user_mapping_enabled=user_mapping_enabled,
                user_uid=user_uid,
                user_gid=user_gid,
                cmd_args=server_args,
            )
        else:
            # Run server locally using fastapi-cli's _run function
            typer.echo("Starting Claude Code Proxy API server locally...")
            typer.echo(f"Server will be available at: http://{host}:{port}")
            typer.echo("")

            # Use fastapi-cli's internal _run function
            _run(
                command="production",
                path=get_default_path_hook(),
                host=host,
                port=port,
                reload=reload,
            )

    except Exception as e:
        typer.echo(f"Error starting server: {e}", err=True)
        raise typer.Exit(1) from e


@app.command()
def claude(
    args: list[str] | None = typer.Argument(
        default=None,
        help="Arguments to pass to claude CLI (e.g. --version, doctor, config)",
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
    user_mapping_enabled: bool | None = typer.Option(
        None,
        "--user-mapping/--no-user-mapping",
        help="Enable/disable UID/GID mapping (overrides config)",
    ),
    user_uid: int | None = typer.Option(
        None,
        "--user-uid",
        help="User ID to run container as (overrides config)",
        min=0,
    ),
    user_gid: int | None = typer.Option(
        None,
        "--user-gid",
        help="Group ID to run container as (overrides config)",
        min=0,
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
    # Handle None args case
    if args is None:
        args = []

    try:
        if docker:
            # Load settings to get Docker configuration
            settings = get_settings(config_path=get_config_path_from_context())

            # Show the command before executing
            docker_cmd = DockerCommandBuilder.from_settings_and_overrides(
                settings.docker_settings,
                docker_image=docker_image,
                docker_env=docker_env,
                docker_volume=docker_volume,
                docker_arg=docker_arg,
                docker_home=docker_home,
                docker_workspace=docker_workspace,
                user_mapping_enabled=user_mapping_enabled,
                user_uid=user_uid,
                user_gid=user_gid,
                command=["claude"],
                cmd_args=args,
            )

            typer.echo(f"Executing: {' '.join(docker_cmd)}")
            typer.echo("")

            # Execute using the new Docker builder method
            DockerCommandBuilder.execute_from_settings(
                settings.docker_settings,
                docker_image=docker_image,
                docker_env=docker_env,
                docker_volume=docker_volume,
                docker_arg=docker_arg,
                docker_home=docker_home,
                docker_workspace=docker_workspace,
                user_mapping_enabled=user_mapping_enabled,
                user_uid=user_uid,
                user_gid=user_gid,
                command=["claude"],
                cmd_args=args,
            )
        else:
            # Load settings to find claude path
            settings = get_settings(config_path=get_config_path_from_context())

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
            try:
                # Use os.execvp to replace current process with claude
                # This hands over full control to claude, including signal handling
                os.execvp(claude_path, [claude_path] + args)
            except OSError as e:
                typer.echo(f"Failed to execute command: {e}", err=True)
                raise typer.Exit(1) from e

    except Exception as e:
        typer.echo(f"Error executing claude command: {e}", err=True)
        raise typer.Exit(1) from e


if __name__ == "__main__":
    app()
