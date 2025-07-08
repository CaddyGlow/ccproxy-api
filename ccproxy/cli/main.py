"""Main entry point for Claude Proxy API Server."""

import json
import logging
import os
import secrets
from pathlib import Path
from typing import Any, Optional, cast

import typer
from click import get_current_context

from ccproxy._version import __version__
from ccproxy.config.settings import (
    ConfigurationError,
    Settings,
    config_manager,
)
from ccproxy.models.responses import (
    PermissionToolAllowResponse,
    PermissionToolDenyResponse,
)
from ccproxy.utils.cli import (
    get_rich_toolkit,
    is_running_in_docker,
    warning,
)
from ccproxy.utils.helper import get_package_dir, get_root_package_name
from ccproxy.utils.logging import get_logger

from .commands.api import api, get_config_path_from_context
from .commands.auth import app as auth_app
from .commands.claude import claude
from .commands.config import app as config_app
from .commands.fastapi import app as fastapi_app


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        toolkit = get_rich_toolkit()
        toolkit.print(f"ccproxy {__version__}", tag="version")
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


# Register config command
app.add_typer(config_app)

# Register auth command
app.add_typer(auth_app)

# Register fastapi command
app.add_typer(fastapi_app, name="fastapi")


# Register imported commands
app.command(hidden=True)(api)
app.command()(claude)


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
            toolkit.print(response.model_dump_json(by_alias=True), tag="result")
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
                toolkit.print(response.model_dump_json(by_alias=True), tag="result")
                return

        # Check for specific tool restrictions
        restricted_tools = {"exec", "system", "shell", "subprocess"}

        if tool_name.lower() in restricted_tools:
            response = PermissionToolDenyResponse(
                message=f"Tool {tool_name} is restricted for security reasons"
            )
            toolkit.print(response.model_dump_json(by_alias=True), tag="result")
            return

        # Allow the tool call with original input
        allow_response = PermissionToolAllowResponse(updatedInput=input_data)
        toolkit.print(allow_response.model_dump_json(by_alias=True), tag="result")

    except Exception as e:
        error_response = PermissionToolDenyResponse(
            message=f"Error processing permission request: {e}"
        )
        toolkit.print(error_response.model_dump_json(by_alias=True), tag="result")
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
__all__ = ["app", "main", "version_callback", "claude"]
