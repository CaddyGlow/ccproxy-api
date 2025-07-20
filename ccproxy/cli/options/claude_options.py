"""Claude-specific CLI options."""

from pathlib import Path
from typing import Any

import typer


def validate_max_thinking_tokens(
    ctx: typer.Context, param: typer.CallbackParam, value: int | None
) -> int | None:
    """Validate max thinking tokens."""
    if value is None:
        return None

    if value < 0:
        raise typer.BadParameter("Max thinking tokens must be non-negative")

    return value


def validate_max_turns(
    ctx: typer.Context, param: typer.CallbackParam, value: int | None
) -> int | None:
    """Validate max turns."""
    if value is None:
        return None

    if value < 1:
        raise typer.BadParameter("Max turns must be at least 1")

    return value


def validate_permission_mode(
    ctx: typer.Context, param: typer.CallbackParam, value: str | None
) -> str | None:
    """Validate permission mode."""
    if value is None:
        return None

    valid_modes = {"default", "acceptEdits", "bypassPermissions"}
    if value not in valid_modes:
        raise typer.BadParameter(
            f"Permission mode must be one of: {', '.join(valid_modes)}"
        )

    return value


def validate_claude_cli_path(
    ctx: typer.Context, param: typer.CallbackParam, value: str | None
) -> str | None:
    """Validate Claude CLI path."""
    if value is None:
        return None

    path = Path(value)
    if not path.exists():
        raise typer.BadParameter(f"Claude CLI path does not exist: {value}")

    return value


def validate_cwd(
    ctx: typer.Context, param: typer.CallbackParam, value: str | None
) -> str | None:
    """Validate working directory."""
    if value is None:
        return None

    path = Path(value)
    if not path.exists():
        raise typer.BadParameter(f"Working directory does not exist: {value}")
    if not path.is_dir():
        raise typer.BadParameter(f"Working directory is not a directory: {value}")

    return value


def max_thinking_tokens_option() -> Any:
    """Max thinking tokens parameter."""
    return typer.Option(
        None,
        "--max-thinking-tokens",
        help="Maximum thinking tokens for Claude Code",
        callback=validate_max_thinking_tokens,
        rich_help_panel="Claude Settings",
    )


def allowed_tools_option() -> Any:
    """Allowed tools parameter."""
    return typer.Option(
        None,
        "--allowed-tools",
        help="List of allowed tools (comma-separated)",
        rich_help_panel="Claude Settings",
    )


def disallowed_tools_option() -> Any:
    """Disallowed tools parameter."""
    return typer.Option(
        None,
        "--disallowed-tools",
        help="List of disallowed tools (comma-separated)",
        rich_help_panel="Claude Settings",
    )


def claude_cli_path_option() -> Any:
    """Claude CLI path parameter."""
    return typer.Option(
        None,
        "--claude-cli-path",
        help="Path to Claude CLI executable",
        callback=validate_claude_cli_path,
        rich_help_panel="Claude Settings",
    )


def append_system_prompt_option() -> Any:
    """Append system prompt parameter."""
    return typer.Option(
        None,
        "--append-system-prompt",
        help="Additional system prompt to append",
        rich_help_panel="Claude Settings",
    )


def permission_mode_option() -> Any:
    """Permission mode parameter."""
    return typer.Option(
        None,
        "--permission-mode",
        help="Permission mode: default, acceptEdits, or bypassPermissions",
        callback=validate_permission_mode,
        rich_help_panel="Claude Settings",
    )


def max_turns_option() -> Any:
    """Max turns parameter."""
    return typer.Option(
        None,
        "--max-turns",
        help="Maximum conversation turns",
        callback=validate_max_turns,
        rich_help_panel="Claude Settings",
    )


def cwd_option() -> Any:
    """Working directory parameter."""
    return typer.Option(
        None,
        "--cwd",
        help="Working directory path",
        callback=validate_cwd,
        rich_help_panel="Claude Settings",
    )


def permission_prompt_tool_name_option() -> Any:
    """Permission prompt tool name parameter."""
    return typer.Option(
        None,
        "--permission-prompt-tool-name",
        help="Permission prompt tool name",
        rich_help_panel="Claude Settings",
    )


class ClaudeOptions:
    """Container for all Claude-related CLI options.

    This class provides a convenient way to include all Claude-related
    options in a command using typed attributes.
    """

    def __init__(
        self,
        max_thinking_tokens: int | None = None,
        allowed_tools: str | None = None,
        disallowed_tools: str | None = None,
        claude_cli_path: str | None = None,
        append_system_prompt: str | None = None,
        permission_mode: str | None = None,
        max_turns: int | None = None,
        cwd: str | None = None,
        permission_prompt_tool_name: str | None = None,
    ):
        """Initialize Claude options.

        Args:
            max_thinking_tokens: Maximum thinking tokens for Claude Code
            allowed_tools: List of allowed tools (comma-separated)
            disallowed_tools: List of disallowed tools (comma-separated)
            claude_cli_path: Path to Claude CLI executable
            append_system_prompt: Additional system prompt to append
            permission_mode: Permission mode
            max_turns: Maximum conversation turns
            cwd: Working directory path
            permission_prompt_tool_name: Permission prompt tool name
        """
        self.max_thinking_tokens = max_thinking_tokens
        self.allowed_tools = allowed_tools
        self.disallowed_tools = disallowed_tools
        self.claude_cli_path = claude_cli_path
        self.append_system_prompt = append_system_prompt
        self.permission_mode = permission_mode
        self.max_turns = max_turns
        self.cwd = cwd
        self.permission_prompt_tool_name = permission_prompt_tool_name
