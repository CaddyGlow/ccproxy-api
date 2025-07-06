"""Main entry point for Claude Proxy API Server."""

import json
import logging
import os
import secrets
from pathlib import Path
from typing import Any, Optional

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
    no_args_is_help=False,
    pretty_exceptions_enable=False,
)
logger = logging.getLogger(__name__)


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
    # Forward all API command options
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
    # Store config path in context for use by commands
    try:
        from click import get_current_context as click_get_current_context

        click_ctx = click_get_current_context()
        click_ctx.ensure_object(dict)
        click_ctx.obj["config_path"] = config
    except RuntimeError:
        # No active click context (e.g., in tests)
        pass

    # If no subcommand was invoked, run the api command by default
    if ctx.invoked_subcommand is None:
        # Call the api command with all the provided parameters
        ctx.invoke(
            api,
            docker=docker,
            port=port,
            host=host,
            reload=reload,
            log_level=log_level,
            workers=workers,
            cors_origins=cors_origins,
            auth_token=auth_token,
            tools_handling=tools_handling,
            claude_cli_path=claude_cli_path,
            max_thinking_tokens=max_thinking_tokens,
            allowed_tools=allowed_tools,
            disallowed_tools=disallowed_tools,
            append_system_prompt=append_system_prompt,
            permission_mode=permission_mode,
            continue_conversation=continue_conversation,
            resume=resume,
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


# Create fastapi subcommand group
fastapi_group = typer.Typer(
    name="fastapi",
    help="FastAPI development commands (run, dev)",
    no_args_is_help=True,
)

# Remove the fastapi callback to avoid the warning
fastapi_app.callback()(None)  # type: ignore[type-var]
# Register fastapi app under the fastapi group
fastapi_group.add_typer(fastapi_app, name="")

# Register fastapi group with main app
app.add_typer(fastapi_group)

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


@app.command(hidden=True)
def api(
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
    workers: int = typer.Option(
        None,
        "--workers",
        help="Number of worker processes",
        min=1,
        max=32,
    ),
    # Security settings
    cors_origins: str = typer.Option(
        None,
        "--cors-origins",
        help="CORS allowed origins (comma-separated)",
    ),
    auth_token: str = typer.Option(
        None,
        "--auth-token",
        help="Bearer token for API authentication",
    ),
    tools_handling: str = typer.Option(
        None,
        "--tools-handling",
        help="How to handle tools definitions: error, warning, or ignore",
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
    continue_conversation: bool = typer.Option(
        None,
        "--continue-conversation/--no-continue-conversation",
        help="Continue previous conversation",
    ),
    resume: str = typer.Option(
        None,
        "--resume",
        help="Resume conversation ID",
    ),
    max_turns: int = typer.Option(
        None,
        "--max-turns",
        help="Maximum conversation turns",
    ),
    permission_prompt_tool_name: str = typer.Option(
        None,
        "--permission-prompt-tool-name",
        help="Permission prompt tool name",
    ),
    cwd: str = typer.Option(
        None,
        "--cwd",
        help="Working directory path",
    ),
    # Pool settings removed - connection pooling functionality has been removed
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
        # Prepare CLI overrides dictionary
        cli_overrides: dict[str, Any] = {}

        # Server settings
        if host is not None:
            cli_overrides["host"] = host
        if port is not None:
            cli_overrides["port"] = port
        if reload is not None:
            cli_overrides["reload"] = reload
        if log_level is not None:
            cli_overrides["log_level"] = log_level
        if workers is not None:
            cli_overrides["workers"] = workers

        # Security settings
        if cors_origins is not None:
            cli_overrides["cors_origins"] = [
                origin.strip() for origin in cors_origins.split(",")
            ]
        if auth_token is not None:
            cli_overrides["auth_token"] = auth_token
        if tools_handling is not None:
            cli_overrides["tools_handling"] = tools_handling
        if claude_cli_path is not None:
            cli_overrides["claude_cli_path"] = claude_cli_path

        # ClaudeCodeOptions parameters
        claude_code_opts: dict[str, Any] = {}
        if max_thinking_tokens is not None:
            claude_code_opts["max_thinking_tokens"] = max_thinking_tokens
        if allowed_tools is not None:
            claude_code_opts["allowed_tools"] = [
                tool.strip() for tool in allowed_tools.split(",")
            ]
        if disallowed_tools is not None:
            claude_code_opts["disallowed_tools"] = [
                tool.strip() for tool in disallowed_tools.split(",")
            ]
        if append_system_prompt is not None:
            claude_code_opts["append_system_prompt"] = append_system_prompt
        if permission_mode is not None:
            claude_code_opts["permission_mode"] = permission_mode
        if continue_conversation is not None:
            claude_code_opts["continue_conversation"] = continue_conversation
        if resume is not None:
            claude_code_opts["resume"] = resume
        if max_turns is not None:
            claude_code_opts["max_turns"] = max_turns
        if permission_prompt_tool_name is not None:
            claude_code_opts["permission_prompt_tool_name"] = (
                permission_prompt_tool_name
            )
        if cwd is not None:
            claude_code_opts["cwd"] = cwd

        if claude_code_opts:
            cli_overrides["claude_code_options"] = claude_code_opts

        # Pool settings removed - connection pooling functionality has been removed

        # Load settings with CLI overrides
        settings = get_settings(config_path=get_config_path_from_context())
        if cli_overrides:
            # Create new settings instance with overrides
            from claude_code_proxy.config.settings import Settings

            settings = Settings.from_config(
                config_path=get_config_path_from_context(), **cli_overrides
            )

        if docker:
            # Prepare server command using fastapi
            server_args = [
                "run",
                "--host",
                "0.0.0.0",  # Docker needs to bind to 0.0.0.0
                "--port",
                str(settings.port),
            ]

            if settings.reload:
                server_args.append("--reload")

            # Build and execute Docker command with settings and CLI overrides
            typer.echo("Starting Claude Code Proxy API server with Docker...")
            typer.echo(
                f"Server will be available at: http://{settings.host}:{settings.port}"
            )

            # Show the command before executing
            docker_cmd = DockerCommandBuilder.from_settings_and_overrides(
                settings.docker_settings,
                docker_image=docker_image,
                docker_env=docker_env + [f"PORT={settings.port}"],
                docker_volume=docker_volume,
                docker_arg=docker_arg + ["-p", f"{settings.port}:{settings.port}"],
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
                docker_env=docker_env + [f"PORT={settings.port}"],
                docker_volume=docker_volume,
                docker_arg=docker_arg + ["-p", f"{settings.port}:{settings.port}"],
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
            typer.echo(
                f"Server will be available at: http://{settings.host}:{settings.port}"
            )
            typer.echo("")

            # Set environment variables for the server to access the settings
            if cli_overrides:
                # Set environment variables for any CLI overrides
                os.environ["CCPROXY_CONFIG_OVERRIDES"] = json.dumps(cli_overrides)

            # Use fastapi-cli's internal _run function
            _run(
                command="production",
                path=get_default_path_hook(),
                host=settings.host,
                port=settings.port,
                reload=settings.reload,
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
