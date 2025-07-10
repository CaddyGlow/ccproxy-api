"""Serve command for Claude Code Proxy API server - consolidates server-related commands."""

import json
import logging
import os
from pathlib import Path
from typing import Any

import typer
import uvicorn
from click import get_current_context

from ccproxy._version import __version__
from ccproxy.config.settings import (
    ConfigurationError,
    Settings,
    config_manager,
)
from ccproxy.docker import (
    DockerEnv,
    DockerPath,
    DockerUserContext,
    DockerVolume,
    create_docker_adapter,
)
from ccproxy.cli.helpers import (
    get_rich_toolkit,
    get_uvicorn_log_config,
    is_running_in_docker,
    warning,
)
from ccproxy.utils.helper import get_root_package_name
from ccproxy.utils.logging import get_logger

from ..docker import (
    _create_docker_adapter_from_settings,
)
from ..docker.params import (
    docker_arg_option,
    docker_env_option,
    docker_home_option,
    docker_image_option,
    docker_volume_option,
    docker_workspace_option,
    user_gid_option,
    user_mapping_option,
    user_uid_option,
)


# Logger will be configured by configuration manager
logger = get_logger(__name__)


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
        app=f"{get_root_package_name()}.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        workers=None,  # ,settings.workers,
        log_config=None,
        # log_config=get_uvicorn_log_config(),
    )


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
    permission_prompt_tool_name: str = typer.Option(
        None,
        "--permission-prompt-tool-name",
        help="Permission prompt tool name",
    ),
    # Docker settings using shared parameters
    docker_image: str | None = docker_image_option(),
    docker_env: list[str] = docker_env_option(),
    docker_volume: list[str] = docker_volume_option(),
    docker_arg: list[str] = docker_arg_option(),
    docker_home: str | None = docker_home_option(),
    docker_workspace: str | None = docker_workspace_option(),
    user_mapping_enabled: bool | None = user_mapping_option(),
    user_uid: int | None = user_uid_option(),
    user_gid: int | None = user_gid_option(),
) -> None:
    """
    Start the Claude Code Proxy API server.

    This command starts the API server either locally or in Docker.
    The server provides both Anthropic and OpenAI-compatible endpoints.

    All configuration options can be provided via CLI parameters,
    which override values from configuration files and environment variables.

    Examples:
        ccproxy serve
        ccproxy serve --port 8080 --reload
        ccproxy serve --docker
        ccproxy serve --docker --docker-image custom:latest --port 8080
        ccproxy serve --max-thinking-tokens 10000 --allowed-tools Read,Write,Bash
        ccproxy serve --port 8080 --workers 4
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
    # Docker settings using shared parameters
    docker_image: str | None = docker_image_option(),
    docker_env: list[str] = docker_env_option(),
    docker_volume: list[str] = docker_volume_option(),
    docker_arg: list[str] = docker_arg_option(),
    docker_home: str | None = docker_home_option(),
    docker_workspace: str | None = docker_workspace_option(),
    user_mapping_enabled: bool | None = user_mapping_option(),
    user_uid: int | None = user_uid_option(),
    user_gid: int | None = user_gid_option(),
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