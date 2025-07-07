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
from claude_code_proxy.docker import (
    DockerEnv,
    DockerPath,
    DockerUserContext,
    DockerVolume,
    create_docker_adapter,
)
from claude_code_proxy.models.responses import (
    PermissionToolAllowResponse,
    PermissionToolDenyResponse,
)
from claude_code_proxy.utils.cli import (
    get_rich_toolkit,
    get_uvicorn_log_config,
    is_running_in_docker,
    warning,
)
from claude_code_proxy.utils.helper import get_package_dir, get_root_package_name
from claude_code_proxy.utils.logging import get_logger

from .commands.config import app as config_app


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        toolkit = get_rich_toolkit()
        toolkit.print(f"claude-code-proxy-api {__version__}", tag="version")
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


def _create_docker_adapter_from_settings(
    settings: "Settings",
    docker_image: str | None = None,
    docker_env: list[str] | None = None,
    docker_volume: list[str] | None = None,
    docker_arg: list[str] | None = None,
    docker_home: str | None = None,
    docker_workspace: str | None = None,
    user_mapping_enabled: bool | None = None,
    user_uid: int | None = None,
    user_gid: int | None = None,
    command: list[str] | None = None,
    cmd_args: list[str] | None = None,
    **kwargs: Any,
) -> tuple[
    str,
    list[DockerVolume],
    DockerEnv,
    list[str] | None,
    DockerUserContext | None,
    list[str],
]:
    """Convert settings and overrides to Docker adapter parameters."""
    docker_settings = settings.docker_settings

    # Determine effective image
    image = docker_image or docker_settings.docker_image

    # Process volumes
    volumes: list[DockerVolume] = []

    # Add home/workspace volumes with effective directories
    home_dir = docker_home or docker_settings.docker_home_directory
    workspace_dir = docker_workspace or docker_settings.docker_workspace_directory

    if home_dir:
        volumes.append((str(Path(home_dir)), "/data/home"))
    if workspace_dir:
        volumes.append((str(Path(workspace_dir)), "/data/workspace"))

    # Add base volumes from settings
    for vol_str in docker_settings.docker_volumes:
        parts = vol_str.split(":", 2)
        if len(parts) >= 2:
            volumes.append((parts[0], parts[1]))

    # Add CLI override volumes
    if docker_volume:
        for vol_str in docker_volume:
            parts = vol_str.split(":", 2)
            if len(parts) >= 2:
                volumes.append((parts[0], parts[1]))

    # Process environment
    environment: DockerEnv = docker_settings.docker_environment.copy()

    # Add home/workspace environment variables
    if home_dir:
        environment["CLAUDE_HOME"] = "/data/home"
    if workspace_dir:
        environment["CLAUDE_WORKSPACE"] = "/data/workspace"

    # Add CLI override environment
    if docker_env:
        for env_var in docker_env:
            if "=" in env_var:
                key, value = env_var.split("=", 1)
                environment[key] = value

    # Create user context
    user_context = None
    effective_mapping_enabled = (
        user_mapping_enabled
        if user_mapping_enabled is not None
        else docker_settings.user_mapping_enabled
    )

    if effective_mapping_enabled:
        effective_uid = user_uid if user_uid is not None else docker_settings.user_uid
        effective_gid = user_gid if user_gid is not None else docker_settings.user_gid

        if effective_uid is not None and effective_gid is not None:
            # Create DockerPath instances for user context
            home_path = None
            workspace_path = None

            if home_dir:
                home_path = DockerPath(
                    host_path=Path(home_dir), container_path="/data/home"
                )
            if workspace_dir:
                workspace_path = DockerPath(
                    host_path=Path(workspace_dir), container_path="/data/workspace"
                )

            # Use a default username if not available
            import getpass

            username = getpass.getuser()

            user_context = DockerUserContext(
                uid=effective_uid,
                gid=effective_gid,
                username=username,
                home_path=home_path,
                workspace_path=workspace_path,
            )

    # Build command
    final_command = None
    if command:
        final_command = command.copy()
        if cmd_args:
            final_command.extend(cmd_args)

    # Additional Docker arguments
    additional_args = docker_settings.docker_additional_args.copy()
    if docker_arg:
        additional_args.extend(docker_arg)

    return image, volumes, environment, final_command, user_context, additional_args


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
        toolkit = get_rich_toolkit()
        toolkit.print(f"Configuration error: {e}", tag="error")
        raise typer.Exit(1) from e
    except Exception as e:
        toolkit = get_rich_toolkit()
        toolkit.print(f"Error starting server: {e}", tag="error")
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
    toolkit = get_rich_toolkit()

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
    toolkit.print_title(
        "Starting Claude Code Proxy API server with Docker", tag="docker"
    )
    toolkit.print(
        f"Server will be available at: http://{settings.host}:{settings.port}",
        tag="info",
    )
    toolkit.print_line()

    # Show Docker configuration summary
    toolkit.print_title("Docker Configuration Summary", tag="config")

    # Determine effective directories for volume mapping
    home_dir = docker_home or settings.docker_settings.docker_home_directory
    workspace_dir = (
        docker_workspace or settings.docker_settings.docker_workspace_directory
    )

    # Show volume information
    toolkit.print("Volumes:", tag="config")
    if home_dir:
        toolkit.print(f"  Home: {home_dir} → /data/home", tag="volume")
    if workspace_dir:
        toolkit.print(f"  Workspace: {workspace_dir} → /data/workspace", tag="volume")
    if docker_volume:
        for vol in docker_volume:
            toolkit.print(f"  Additional: {vol}", tag="volume")
    toolkit.print_line()

    # Show environment information
    toolkit.print("Environment Variables:", tag="config")
    key_env_vars = {
        "CLAUDE_HOME": "/data/home",
        "CLAUDE_WORKSPACE": "/data/workspace",
        "PORT": str(settings.port),
        "HOST": "0.0.0.0",
    }
    if settings.reload:
        key_env_vars["RELOAD"] = "true"

    for key, value in key_env_vars.items():
        toolkit.print(f"  {key}={value}", tag="env")

    # Show additional environment variables from CLI
    for env_var in docker_env:
        toolkit.print(f"  {env_var}", tag="env")

    # Show debug environment information if log level is DEBUG
    if settings.log_level == "DEBUG":
        toolkit.print_line()
        toolkit.print_title("Debug: All Environment Variables", tag="debug")
        all_env = {**docker_env_dict}
        for key, value in sorted(all_env.items()):
            toolkit.print(f"  {key}={value}", tag="debug")

    toolkit.print_line()

    toolkit.print_line()

    # Show API usage information if auth token is configured
    if settings.auth_token:
        _show_api_usage_info(toolkit, settings)

    # Execute using the new Docker adapter
    image, volumes, environment, command, user_context, additional_args = (
        _create_docker_adapter_from_settings(
            settings,
            docker_image=docker_image,
            docker_env=[f"{k}={v}" for k, v in docker_env_dict.items()],
            docker_volume=docker_volume,
            docker_arg=docker_arg,
            docker_home=docker_home,
            docker_workspace=docker_workspace,
            user_mapping_enabled=user_mapping_enabled,
            user_uid=user_uid,
            user_gid=user_gid,
        )
    )

    logger.info(f"image {settings.docker_settings.docker_image}")
    logger.info(f"image2 {image}")

    # Add port mapping
    ports = [f"{settings.port}:{settings.port}"]

    # Create Docker adapter and execute
    adapter = create_docker_adapter()
    adapter.exec_container(
        image=image,
        volumes=volumes,
        environment=environment,
        command=command,
        user_context=user_context,
        ports=ports,
    )


def _show_api_usage_info(toolkit: Any, settings: Settings) -> None:
    """Show API usage information when auth token is configured."""
    from rich.console import Console
    from rich.syntax import Syntax

    toolkit.print_title("API Client Configuration", tag="config")

    # Determine the base URLs
    anthropic_base_url = f"http://{settings.host}:{settings.port}"
    openai_base_url = f"http://{settings.host}:{settings.port}/openai"

    # Show environment variable exports using code blocks
    toolkit.print("Environment Variables for API Clients:", tag="info")
    toolkit.print_line()

    # Use rich console for code blocks
    console = Console()

    exports = f"""export ANTHROPIC_API_KEY={settings.auth_token}
export ANTHROPIC_BASE_URL={anthropic_base_url}
export OPENAI_API_KEY={settings.auth_token}
export OPENAI_BASE_URL={openai_base_url}"""

    console.print(Syntax(exports, "bash", theme="monokai", background_color="default"))
    toolkit.print_line()


def _run_local_server(settings: Settings, cli_overrides: dict[str, Any]) -> None:
    in_docker = is_running_in_docker()
    """Run the server locally."""
    toolkit = get_rich_toolkit()

    if in_docker:
        toolkit.print_title(
            f"Starting Claude Code Proxy API server in {warning('docker')}",
            tag="docker",
        )
        toolkit.print(
            f"uid={warning(str(os.getuid()))} gid={warning(str(os.getgid()))}"
        )
        toolkit.print(f"HOME={os.environ['HOME']}")
    else:
        toolkit.print_title("Starting Claude Code Proxy API server", tag="local")

    toolkit.print(
        f"Server will be available at: http://{settings.host}:{settings.port}",
        tag="info",
    )

    toolkit.print_line()

    # Show API usage information if auth token is configured
    if settings.auth_token:
        _show_api_usage_info(toolkit, settings)

    # Set environment variables for server to access CLI overrides
    if cli_overrides:
        os.environ["CCPROXY_CONFIG_OVERRIDES"] = json.dumps(cli_overrides)

    logger.info(f"Starting production server at http://{settings.host}:{settings.port}")

    # Run uvicorn with our already configured logging
    uvicorn.run(
        app=f"{get_root_package_name()}:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        workers=None,  # ,settings.workers,
        log_config=None,
        # log_config=get_uvicorn_log_config(),
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

    toolkit = get_rich_toolkit()

    try:
        # Load settings using configuration manager
        settings = config_manager.load_settings(
            config_path=get_config_path_from_context()
        )

        if docker:
            # Prepare Docker execution using new adapter

            toolkit.print_title(
                f"image {settings.docker_settings.docker_image}", tag="docker"
            )
            image, volumes, environment, command, user_context, additional_args = (
                _create_docker_adapter_from_settings(
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
                    command=["claude"],
                    cmd_args=args,
                )
            )

            cmd_str = " ".join(command or [])
            toolkit.print(f"Executing: docker run ... {image} {cmd_str}", tag="docker")
            toolkit.print_line()

            # Execute using the new Docker adapter
            adapter = create_docker_adapter()
            adapter.exec_container(
                image=image,
                volumes=volumes,
                environment=environment,
                command=command,
                user_context=user_context,
            )
        else:
            # Get claude path from settings
            claude_path = settings.claude_cli_path
            if not claude_path:
                toolkit.print("Error: Claude CLI not found.", tag="error")
                toolkit.print(
                    "Please install Claude CLI or configure claude_cli_path.",
                    tag="error",
                )
                raise typer.Exit(1)

            # Resolve to absolute path
            if not Path(claude_path).is_absolute():
                claude_path = str(Path(claude_path).resolve())

            toolkit.print(f"Executing: {claude_path} {' '.join(args)}", tag="claude")
            toolkit.print_line()

            # Execute command directly
            try:
                # Use os.execvp to replace current process with claude
                # This hands over full control to claude, including signal handling
                os.execvp(claude_path, [claude_path] + args)
            except OSError as e:
                toolkit.print(f"Failed to execute command: {e}", tag="error")
                raise typer.Exit(1) from e

    except ConfigurationError as e:
        toolkit.print(f"Configuration error: {e}", tag="error")
        raise typer.Exit(1) from e
    except Exception as e:
        toolkit.print(f"Error executing claude command: {e}", tag="error")
        raise typer.Exit(1) from e


@app.command()
def permission_tool(
    tool_name: str = typer.Argument(
        ..., help="Name of the tool to check permissions for"
    ),
    tool_input: str = typer.Argument(..., help="JSON string of the tool input"),
) -> None:
    """
    MCP permission prompt tool for Claude Code SDK.

    This tool is used by the Claude Code SDK to check permissions for tool calls.
    It returns a JSON response indicating whether the tool call should be allowed or denied.

    Response format:
    - Allow: {"behavior": "allow", "updatedInput": {...}}
    - Deny: {"behavior": "deny", "message": "reason"}

    Examples:
        ccproxy permission_tool "bash" '{"command": "ls -la"}'
        ccproxy permission_tool "edit_file" '{"path": "/etc/passwd", "content": "..."}'
    """
    toolkit = get_rich_toolkit()

    try:
        # Parse the tool input JSON
        try:
            input_data = json.loads(tool_input)
        except json.JSONDecodeError as e:
            response = PermissionToolDenyResponse(message=f"Invalid JSON input: {e}")
            toolkit.print(response.model_dump_json(), tag="result")
            raise typer.Exit(1) from e

        # Load settings to get permission configuration
        settings = config_manager.load_settings(
            config_path=get_config_path_from_context()
        )

        # Basic permission checking logic
        # This can be extended with more sophisticated rules

        # Check for potentially dangerous commands
        dangerous_patterns = [
            "rm -rf",
            "sudo",
            "passwd",
            "chmod 777",
            "/etc/passwd",
            "/etc/shadow",
            "format",
            "mkfs",
        ]

        # Convert input to string for pattern matching
        input_str = json.dumps(input_data).lower()

        # Check for dangerous patterns
        for pattern in dangerous_patterns:
            if pattern in input_str:
                response = PermissionToolDenyResponse(
                    message=f"Tool call contains potentially dangerous pattern: {pattern}"
                )
                toolkit.print(response.model_dump_json(), tag="result")
                return

        # Check for specific tool restrictions
        restricted_tools = {"exec", "system", "shell", "subprocess"}

        if tool_name.lower() in restricted_tools:
            response = PermissionToolDenyResponse(
                message=f"Tool {tool_name} is restricted for security reasons"
            )
            toolkit.print(response.model_dump_json(), tag="result")
            return

        # Allow the tool call with original input
        allow_response = PermissionToolAllowResponse(updatedInput=input_data)
        toolkit.print(allow_response.model_dump_json(), tag="result")

    except Exception as e:
        error_response = PermissionToolDenyResponse(
            message=f"Error processing permission request: {e}"
        )
        toolkit.print(error_response.model_dump_json(), tag="result")
        raise typer.Exit(1) from e


if __name__ == "__main__":
    import sys

    # Enhanced default command handling
    if len(sys.argv) == 1:
        # No arguments provided, run api command
        sys.argv.append("api")
    elif len(sys.argv) > 1:
        # Check if any argument is a known command
        known_commands = {"api", "claude", "config", "fastapi", "permission_tool"}

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
