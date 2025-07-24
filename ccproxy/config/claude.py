"""Claude-specific configuration settings."""

import os
import shutil
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, Field, field_validator

from ccproxy.core.async_utils import get_package_dir, patched_typing


# For further information visit https://errors.pydantic.dev/2.11/u/typed-dict-version
with patched_typing():
    from claude_code_sdk import ClaudeCodeOptions  # noqa: E402

logger = structlog.get_logger(__name__)


def _create_default_claude_code_options() -> ClaudeCodeOptions:
    """Create ClaudeCodeOptions with default values."""
    return ClaudeCodeOptions(
        mcp_servers={
            "confirmation": {"type": "sse", "url": "http://127.0.0.1:8000/mcp"}
        },
        permission_prompt_tool_name="mcp__confirmation__check_permission",
    )


class ClaudeSettings(BaseModel):
    """Claude-specific configuration settings."""

    cli_path: str | None = Field(
        default=None,
        description="Path to Claude CLI executable",
    )

    code_options: ClaudeCodeOptions = Field(
        default_factory=_create_default_claude_code_options,
        description="Claude Code SDK options configuration",
    )

    @field_validator("cli_path")
    @classmethod
    def validate_claude_cli_path(cls, v: str | None) -> str | None:
        """Validate Claude CLI path if provided."""
        if v is not None:
            path = Path(v)
            if not path.exists():
                raise ValueError(f"Claude CLI path does not exist: {v}")
            if not path.is_file():
                raise ValueError(f"Claude CLI path is not a file: {v}")
            if not os.access(path, os.X_OK):
                raise ValueError(f"Claude CLI path is not executable: {v}")
        return v

    @field_validator("code_options", mode="before")
    @classmethod
    def validate_claude_code_options(cls, v: Any) -> Any:
        """Validate and convert Claude Code options."""
        if v is None:
            # Create instance with default values (same as default_factory)
            return _create_default_claude_code_options()

        # If it's already a ClaudeCodeOptions instance, return as-is
        if isinstance(v, ClaudeCodeOptions):
            return v

        # If it's an empty dict, treat it like None and use defaults
        if isinstance(v, dict) and not v:
            return _create_default_claude_code_options()

        # For non-empty dicts, merge with defaults instead of replacing them
        if isinstance(v, dict):
            # Start with default values
            defaults = _create_default_claude_code_options()

            # Extract default values as a dict for merging
            default_values = {
                "mcp_servers": defaults.mcp_servers.copy(),
                "permission_prompt_tool_name": defaults.permission_prompt_tool_name,
            }

            # Add other default attributes if they exist
            for attr in [
                "max_thinking_tokens",
                "allowed_tools",
                "disallowed_tools",
                "cwd",
                "append_system_prompt",
                "max_turns",
                "continue_conversation",
                "permission_mode",
                "model",
                "system_prompt",
            ]:
                if hasattr(defaults, attr):
                    default_value = getattr(defaults, attr, None)
                    if default_value is not None:
                        default_values[attr] = default_value

            # Merge CLI overrides with defaults (CLI overrides take precedence)
            merged_values = {**default_values, **v}

            return ClaudeCodeOptions(**merged_values)

        # Try to convert to dict if possible
        if hasattr(v, "model_dump"):
            return v.model_dump()
        elif hasattr(v, "__dict__"):
            return v.__dict__

        return v

    def find_claude_cli(self) -> tuple[str | None, bool]:
        """Find Claude CLI executable in PATH or specified location.

        Returns:
            tuple: (path_to_claude, found_in_path)
        """
        if self.cli_path:
            return self.cli_path, False

        # Try to find claude in PATH
        claude_path = shutil.which("claude")
        if claude_path:
            return claude_path, True

        # Common installation paths (in order of preference)
        common_paths = [
            # User-specific Claude installation
            Path.home() / ".claude" / "local" / "claude",
            # User's global node_modules (npm install -g)
            Path.home() / "node_modules" / ".bin" / "claude",
            # Package installation directory node_modules
            get_package_dir() / "node_modules" / ".bin" / "claude",
            # Current working directory node_modules
            Path.cwd() / "node_modules" / ".bin" / "claude",
            # System-wide installations
            Path("/usr/local/bin/claude"),
            Path("/opt/homebrew/bin/claude"),
        ]

        for path in common_paths:
            if path.exists() and path.is_file() and os.access(path, os.X_OK):
                return str(path), False

        return None, False

    def get_searched_paths(self) -> list[str]:
        """Get list of paths that would be searched for Claude CLI auto-detection."""
        paths = []

        # PATH search
        paths.append("PATH environment variable")

        # Common installation paths (in order of preference)
        common_paths = [
            # User-specific Claude installation
            Path.home() / ".claude" / "local" / "claude",
            # User's global node_modules (npm install -g)
            Path.home() / "node_modules" / ".bin" / "claude",
            # Package installation directory node_modules
            get_package_dir() / "node_modules" / ".bin" / "claude",
            # Current working directory node_modules
            Path.cwd() / "node_modules" / ".bin" / "claude",
            # System-wide installations
            Path("/usr/local/bin/claude"),
            Path("/opt/homebrew/bin/claude"),
        ]

        for path in common_paths:
            paths.append(str(path))

        return paths
