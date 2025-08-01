"""Claude-specific CLI options."""

from pathlib import Path

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


def validate_sdk_message_mode(
    ctx: typer.Context, param: typer.CallbackParam, value: str | None
) -> str | None:
    """Validate SDK message mode."""
    if value is None:
        return None

    valid_modes = {"forward", "ignore", "formatted"}
    if value not in valid_modes:
        raise typer.BadParameter(
            f"SDK message mode must be one of: {', '.join(valid_modes)}"
        )

    return value


def validate_pool_size(
    ctx: typer.Context, param: typer.CallbackParam, value: int | None
) -> int | None:
    """Validate pool size."""
    if value is None:
        return None

    if value < 1:
        raise typer.BadParameter("Pool size must be at least 1")

    if value > 20:
        raise typer.BadParameter("Pool size must not exceed 20")

    return value


def validate_system_prompt_injection_mode(
    ctx: typer.Context, param: typer.CallbackParam, value: str | None
) -> str | None:
    """Validate system prompt injection mode."""
    if value is None:
        return None

    valid_modes = {"minimal", "full"}
    if value not in valid_modes:
        raise typer.BadParameter(
            f"System prompt injection mode must be one of: {', '.join(valid_modes)}"
        )

    return value


# Factory functions removed - use Annotated syntax directly in commands


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
        sdk_message_mode: str | None = None,
        sdk_pool: bool = False,
        sdk_pool_size: int | None = None,
        sdk_session_pool: bool = False,
        system_prompt_injection_mode: str | None = None,
        builtin_permissions: bool = True,
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
            sdk_message_mode: SDK message handling mode
            sdk_pool: Enable general Claude SDK client connection pooling
            sdk_pool_size: Number of clients to maintain in the general pool
            sdk_session_pool: Enable session-aware Claude SDK client pooling
            system_prompt_injection_mode: System prompt injection mode
            builtin_permissions: Enable built-in permission handling infrastructure
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
        self.sdk_message_mode = sdk_message_mode
        self.sdk_pool = sdk_pool
        self.sdk_pool_size = sdk_pool_size
        self.sdk_session_pool = sdk_session_pool
        self.system_prompt_injection_mode = system_prompt_injection_mode
        self.builtin_permissions = builtin_permissions
