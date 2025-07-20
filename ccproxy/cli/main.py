"""Main entry point for Claude Proxy API Server."""

import json
import os
import secrets
from pathlib import Path
from typing import Any, Optional, cast

import typer
from click import get_current_context
from structlog import get_logger
from typer import Typer

from ccproxy._version import __version__
from ccproxy.api.middleware.cors import setup_cors_middleware
from ccproxy.cli.helpers import (
    get_rich_toolkit,
    is_running_in_docker,
    warning,
)
from ccproxy.config.settings import (
    ConfigurationError,
    Settings,
    config_manager,
    get_settings,
)
from ccproxy.core.async_utils import get_package_dir, get_root_package_name
from ccproxy.core.logging import setup_logging
from ccproxy.models.responses import (
    PermissionToolAllowResponse,
    PermissionToolDenyResponse,
)

from .commands.auth import app as auth_app
from .commands.config import app as config_app
from .commands.serve import api, get_config_path_from_context


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        toolkit = get_rich_toolkit()
        toolkit.print(f"ccproxy {__version__}", tag="version")
        raise typer.Exit()


app = typer.Typer(
    rich_markup_mode="rich",
    add_completion=True,
    no_args_is_help=False,
    pretty_exceptions_enable=False,
)

# Logger will be configured by configuration manager
logger = get_logger(__name__)


# Add global options
@app.callback()
def app_main(
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
) -> None:
    """Claude Code Proxy API Server - Anthropic and OpenAI compatible interface for Claude."""
    # Store config path for commands to use
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config


# Register config command
app.add_typer(config_app)

# Register auth command
app.add_typer(auth_app)


# Register imported commands
app.command(name="serve")(api)
# Claude command removed - functionality moved to serve command


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
        allow_response = PermissionToolAllowResponse(updated_input=input_data)
        toolkit.print(allow_response.model_dump_json(by_alias=True), tag="result")

    except Exception as e:
        error_response = PermissionToolDenyResponse(
            message=f"Error processing permission request: {e}"
        )
        toolkit.print(error_response.model_dump_json(by_alias=True), tag="result")
        raise typer.Exit(1) from e


def main() -> None:
    """Entry point for the CLI application."""
    app()


if __name__ == "__main__":
    import sys

    sys.exit(app())
