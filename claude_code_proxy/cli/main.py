"""Main entry point for Claude Proxy API Server."""

import json
import logging
import os
import secrets
from pathlib import Path
from typing import Any, Optional, cast

import typer
import uvicorn
from click import get_current_context

from claude_code_proxy._version import __version__
from claude_code_proxy.config.settings import (
    ConfigurationError,
    Settings,
    config_manager,
)
from claude_code_proxy.utils.docker_builder import DockerCommandBuilder
from claude_code_proxy.utils.helper import get_package_dir, get_root_package_name
from claude_code_proxy.utils.logging import get_logger

from .commands.config import app as config_app


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        typer.echo(f"claude-code-proxy-api {__version__}")
        raise typer.Exit()


app = typer.Typer(
    rich_markup_mode="rich",
    add_completion=False,
    no_args_is_help=False,
    pretty_exceptions_enable=False,
)

# Logger will be configured by configuration manager
logger = get_logger(__name__)


# Add global --version option
@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
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
    # Forward all API command options for default behavior
    port: int = typer.Option(None, "--port", "-p", help="Port to run the server on"),
    host: str = typer.Option(None, "--host", "-h", help="Host to bind the server to"),
    reload: bool = typer.Option(
        None, "--reload/--no-reload", help="Enable auto-reload for development"
    ),
    log_level: str = typer.Option(
        None,
        "--log-level",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    ),
    workers: int = typer.Option(
        None, "--workers", help="Number of worker processes", min=1, max=32
    ),
    docker: bool = typer.Option(
        False,
        "--docker",
        "-d",
        help="Run API server using Docker instead of local execution",
    ),
    cors_origins: str = typer.Option(
        None, "--cors-origins", help="CORS allowed origins (comma-separated)"
    ),
    auth_token: str = typer.Option(
        None, "--auth-token", help="Bearer token for API authentication"
    ),
    tools_handling: str = typer.Option(
        None,
        "--tools-handling",
        help="How to handle tools definitions: error, warning, or ignore",
    ),
    claude_cli_path: str = typer.Option(
        None, "--claude-cli-path", help="Path to Claude CLI executable"
    ),
    max_thinking_tokens: int = typer.Option(
        None, "--max-thinking-tokens", help="Maximum thinking tokens for Claude Code"
    ),
    allowed_tools: str = typer.Option(
        None, "--allowed-tools", help="List of allowed tools (comma-separated)"
    ),
    disallowed_tools: str = typer.Option(
        None, "--disallowed-tools", help="List of disallowed tools (comma-separated)"
    ),
    append_system_prompt: str = typer.Option(
        None, "--append-system-prompt", help="Additional system prompt to append"
    ),
    permission_mode: str = typer.Option(
        None,
        "--permission-mode",
        help="Permission mode: default, acceptEdits, or bypassPermissions",
    ),
    continue_conversation: bool = typer.Option(
        None,
        "--continue-conversation/--no-continue-conversation",
        help="Continue previous conversation",
    ),
    resume: str = typer.Option(None, "--resume", help="Resume conversation ID"),
    max_turns: int = typer.Option(
        None, "--max-turns", help="Maximum conversation turns"
    ),
    permission_prompt_tool_name: str = typer.Option(
        None, "--permission-prompt-tool-name", help="Permission prompt tool name"
    ),
    cwd: str = typer.Option(None, "--cwd", help="Working directory path"),
    docker_image: str | None = typer.Option(
        None, "--docker-image", help="Docker image to use (overrides config)"
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
        None, "--user-uid", help="User ID to run container as (overrides config)", min=0
    ),
    user_gid: int | None = typer.Option(
        None,
        "--user-gid",
        help="Group ID to run container as (overrides config)",
        min=0,
    ),
) -> None:
    """Claude Code Proxy API Server - Anthropic and OpenAI compatible interface for Claude."""
    # Store config path for commands to use
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config

    # If no subcommand was invoked, run the api command by default
    if ctx.invoked_subcommand is None:
        # Forward all parameters to the api command
        ctx.invoke(
            api,
            config=config,
            docker=docker,
            port=port,
            host=host,
            reload=reload,
            log_level=log_level,
            auth_token=auth_token,
            claude_cli_path=claude_cli_path,
            max_thinking_tokens=max_thinking_tokens,
            allowed_tools=allowed_tools,
            disallowed_tools=disallowed_tools,
            append_system_prompt=append_system_prompt,
            permission_mode=permission_mode,
            max_turns=max_turns,
            permission_prompt_tool_name=permission_prompt_tool_name,
            cwd=cwd,
            docker_image=docker_image,
            docker_env=docker_env,
            docker_volume=docker_volume,
            docker_arg=docker_arg,
            docker_home=docker_home,
            docker_workspace=docker_workspace,
            user_mapping_enabled=user_mapping_enabled,
            user_uid=user_uid,
            user_gid=user_gid,
        )


def _run(
    command: str,
    host: str,
    port: int,
    reload: bool,
    app: str = f"{get_root_package_name()}.{__name__}:app",
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
        logger.info(f"Server will run at http://{host}:{port}")

        # Run uvicorn with our already configured logging
        uvicorn.run(
            app=app,
            host=host,
            port=port,
            reload=reload,
            workers=workers if not reload else None,
            log_config=None,  # Use our already configured logging
        )
    except ConfigurationError as e:
        typer.echo(f"Configuration error: {e}", err=True)
        raise typer.Exit(1) from e
    except Exception as e:
        typer.echo(f"Error starting {command} server: {e}", err=True)
        raise typer.Exit(1) from e


# Create FastAPI sub-application
fastapi_app = typer.Typer(help="FastAPI development commands")


# Move run and dev commands to fastapi sub-application
@fastapi_app.command()
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


@fastapi_app.command()
def dev(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(
        True, "--reload/--no-reload", help="Enable auto-reload"
    ),
) -> None:
    """Run a FastAPI app in development mode."""
    _run("development", host, port, reload)


# Register config command
app.add_typer(config_app)

# Register fastapi command
app.add_typer(fastapi_app, name="fastapi")


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


@app.command(hidden=True)
def api(
    # Configuration
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file (TOML, JSON, or YAML)",
    ),
    # Core server settings
    docker: bool = typer.Option(
        False,
        "--docker",
        "-d",
        help="Run API server using Docker instead of local execution",
    ),
    port: int = typer.Option(
        None,
        "--port",
        "-p",
        help="Port to run the server on",
    ),
    host: str = typer.Option(
        None,
        "--host",
        "-h",
        help="Host to bind the server to",
    ),
    reload: bool = typer.Option(
        None,
        "--reload/--no-reload",
        help="Enable auto-reload for development",
    ),
    log_level: str = typer.Option(
        None,
        "--log-level",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    ),
    auth_token: str = typer.Option(
        None,
        "--auth-token",
        help="Bearer token for API authentication",
    ),
    claude_cli_path: str = typer.Option(
        None,
        "--claude-cli-path",
        help="Path to Claude CLI executable",
    ),
    # ClaudeCodeOptions parameters
    max_thinking_tokens: int = typer.Option(
        None,
        "--max-thinking-tokens",
        help="Maximum thinking tokens for Claude Code",
    ),
    allowed_tools: str = typer.Option(
        None,
        "--allowed-tools",
        help="List of allowed tools (comma-separated)",
    ),
    disallowed_tools: str = typer.Option(
        None,
        "--disallowed-tools",
        help="List of disallowed tools (comma-separated)",
    ),
    append_system_prompt: str = typer.Option(
        None,
        "--append-system-prompt",
        help="Additional system prompt to append",
    ),
    permission_mode: str = typer.Option(
        None,
        "--permission-mode",
        help="Permission mode: default, acceptEdits, or bypassPermissions",
    ),
    max_turns: int = typer.Option(
        None,
        "--max-turns",
        help="Maximum conversation turns",
    ),
    cwd: str = typer.Option(
        None,
        "--cwd",
        help="Working directory path",
    ),
    # Docker settings
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
    permission_prompt_tool_name: str = typer.Option(
        None,
        "--permission-prompt-tool-name",
        help="Permission prompt tool name",
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

    All configuration options can be provided via CLI parameters,
    which override values from configuration files and environment variables.

    Examples:
        ccproxy api
        ccproxy api --port 8080 --reload
        ccproxy api --docker
        ccproxy api --docker --docker-image custom:latest --port 8080
        ccproxy api --max-thinking-tokens 10000 --allowed-tools Read,Write,Bash
        ccproxy api --port 8080 --workers 4
    """
    try:
        # Get config path from context if not provided directly
        if config is None:
            config = get_config_path_from_context()

        # Extract CLI overrides from all provided arguments
        cli_overrides = config_manager.get_cli_overrides_from_args(
            host=host,
            port=port,
            reload=reload,
            log_level=log_level,
            auth_token=auth_token,
            claude_cli_path=claude_cli_path,
            max_thinking_tokens=max_thinking_tokens,
            allowed_tools=allowed_tools,
            disallowed_tools=disallowed_tools,
            append_system_prompt=append_system_prompt,
            permission_mode=permission_mode,
            max_turns=max_turns,
            permission_prompt_tool_name=permission_prompt_tool_name,
            cwd=cwd,
        )

        # Load settings with CLI overrides
        settings = config_manager.load_settings(
            config_path=config, cli_overrides=cli_overrides
        )

        # Set up logging once with the effective log level
        config_manager.setup_logging(log_level or settings.log_level)

        if docker:
            _run_docker_server(
                settings,
                docker_image=docker_image,
                docker_env=docker_env,
                docker_volume=docker_volume,
                docker_arg=docker_arg,
                docker_home=docker_home,
                docker_workspace=docker_workspace,
                user_mapping_enabled=user_mapping_enabled,
                user_uid=user_uid,
                user_gid=user_gid,
            )
        else:
            _run_local_server(settings, cli_overrides)

    except ConfigurationError as e:
        typer.echo(f"Configuration error: {e}", err=True)
        raise typer.Exit(1) from e
    except Exception as e:
        typer.echo(f"Error starting server: {e}", err=True)
        raise typer.Exit(1) from e


def _run_docker_server(
    settings: Settings,
    docker_image: str | None = None,
    docker_env: list[str] | None = None,
    docker_volume: list[str] | None = None,
    docker_arg: list[str] | None = None,
    docker_home: str | None = None,
    docker_workspace: str | None = None,
    user_mapping_enabled: bool | None = None,
    user_uid: int | None = None,
    user_gid: int | None = None,
) -> None:
    """Run the server using Docker."""
    docker_env = docker_env or []
    docker_volume = docker_volume or []
    docker_arg = docker_arg or []

    docker_env_dict = {}
    for env_var in docker_env:
        if "=" in env_var:
            key, value = env_var.split("=", 1)
            docker_env_dict[key] = value

    # Add server configuration to Docker environment
    if settings.reload:
        docker_env_dict["RELOAD"] = "true"
    docker_env_dict["PORT"] = str(settings.port)
    docker_env_dict["HOST"] = "0.0.0.0"

    # Display startup information
    typer.echo("Starting Claude Code Proxy API server with Docker...")
    typer.echo(f"Server will be available at: http://{settings.host}:{settings.port}")
    typer.echo("")

    # Show Docker configuration summary
    typer.echo("=== Docker Configuration Summary ===")

    # Determine effective directories for volume mapping
    home_dir = docker_home or settings.docker_settings.docker_home_directory
    workspace_dir = (
        docker_workspace or settings.docker_settings.docker_workspace_directory
    )

    # Show volume information
    typer.echo("Volumes:")
    if home_dir:
        typer.echo(f"  Home: {home_dir} → /data/home")
    if workspace_dir:
        typer.echo(f"  Workspace: {workspace_dir} → /data/workspace")
    if docker_volume:
        for vol in docker_volume:
            typer.echo(f"  Additional: {vol}")
    typer.echo("")

    # Show environment information
    typer.echo("Environment Variables:")
    key_env_vars = {
        "CLAUDE_HOME": "/data/home",
        "CLAUDE_WORKSPACE": "/data/workspace",
        "PORT": str(settings.port),
        "HOST": "0.0.0.0",
    }
    if settings.reload:
        key_env_vars["RELOAD"] = "true"

    for key, value in key_env_vars.items():
        typer.echo(f"  {key}={value}")

    # Show additional environment variables from CLI
    for env_var in docker_env:
        typer.echo(f"  {env_var}")

    # Show debug environment information if log level is DEBUG
    if settings.log_level == "DEBUG":
        typer.echo("")
        typer.echo("=== Debug: All Environment Variables ===")
        all_env = {**docker_env_dict}
        for key, value in sorted(all_env.items()):
            typer.echo(f"  {key}={value}")

    typer.echo("")

    # Execute using the Docker builder
    DockerCommandBuilder.execute_from_settings(
        settings.docker_settings,
        docker_image=docker_image,
        docker_env=[f"{k}={v}" for k, v in docker_env_dict.items()],
        docker_volume=docker_volume,
        docker_arg=docker_arg + ["-p", f"{settings.port}:{settings.port}"],
        docker_home=docker_home,
        docker_workspace=docker_workspace,
        user_mapping_enabled=user_mapping_enabled,
        user_uid=user_uid,
        user_gid=user_gid,
    )


def _run_local_server(settings: Settings, cli_overrides: dict[str, Any]) -> None:
    """Run the server locally."""
    typer.echo("Starting Claude Code Proxy API server locally...")
    typer.echo(f"Server will be available at: http://{settings.host}:{settings.port}")
    typer.echo("")

    # Set environment variables for server to access CLI overrides
    if cli_overrides:
        os.environ["CCPROXY_CONFIG_OVERRIDES"] = json.dumps(cli_overrides)

    logger.info(f"Starting production server at http://{settings.host}:{settings.port}")

    # Run uvicorn with our already configured logging
    uvicorn.run(
        app="claude_code_proxy.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        workers=settings.workers if not settings.reload else None,
        log_config=None,  # Use our already configured logging
    )


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
        # Load settings using configuration manager
        settings = config_manager.load_settings(
            config_path=get_config_path_from_context()
        )

        if docker:
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

            # Execute using the Docker builder method
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
            # Get claude path from settings
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

    except ConfigurationError as e:
        typer.echo(f"Configuration error: {e}", err=True)
        raise typer.Exit(1) from e
    except Exception as e:
        typer.echo(f"Error executing claude command: {e}", err=True)
        raise typer.Exit(1) from e


if __name__ == "__main__":
    import sys

    # Enhanced default command handling
    if len(sys.argv) == 1:
        # No arguments provided, run api command
        sys.argv.append("api")
    elif len(sys.argv) > 1:
        # Check if any argument is a known command
        known_commands = {"api", "claude", "config", "fastapi"}

        # Find the first non-option argument that could be a command
        has_command = False
        for arg in sys.argv[1:]:
            if not arg.startswith("-") and arg in known_commands:
                has_command = True
                break

        # If no known command found, but there are arguments,
        # assume they are for the api command
        if (
            not has_command
            and "--help" not in sys.argv
            and "-h" not in sys.argv
            and "--version" not in sys.argv
            and "-V" not in sys.argv
        ):
            sys.argv.insert(1, "api")

    app()
