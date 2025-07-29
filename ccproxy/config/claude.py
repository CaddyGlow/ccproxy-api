"""Claude-specific configuration settings."""

import os
import shutil
from enum import Enum
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, Field, field_validator, model_validator

from ccproxy.core.async_utils import get_package_dir, patched_typing


# For further information visit https://errors.pydantic.dev/2.11/u/typed-dict-version
with patched_typing():
    from claude_code_sdk import ClaudeCodeOptions  # noqa: E402

logger = structlog.get_logger(__name__)


def _create_default_claude_code_options(
    builtin_permissions: bool = True,
) -> ClaudeCodeOptions:
    """Create ClaudeCodeOptions with default values.

    Args:
        builtin_permissions: Whether to include built-in permission handling defaults
    """
    if builtin_permissions:
        return ClaudeCodeOptions(
            mcp_servers={
                "confirmation": {"type": "sse", "url": "http://127.0.0.1:8000/mcp"}
            },
            permission_prompt_tool_name="mcp__confirmation__check_permission",
        )
    else:
        return ClaudeCodeOptions(
            mcp_servers={},
            permission_prompt_tool_name=None,
        )


class SDKMessageMode(str, Enum):
    """Modes for handling SDK messages from Claude SDK.

    - forward: Forward SDK content blocks directly with original types and metadata
    - ignore: Skip SDK messages and blocks completely
    - formatted: Format as XML tags with JSON data in text deltas
    """

    FORWARD = "forward"
    IGNORE = "ignore"
    FORMATTED = "formatted"


class SystemPromptInjectionMode(str, Enum):
    """Modes for system prompt injection.

    - minimal: Only inject Claude Code identification prompt
    - full: Inject all detected system messages from Claude CLI
    """

    MINIMAL = "minimal"
    FULL = "full"


class ClaudePoolSettings(BaseModel):
    """Configuration settings for Claude SDK client connection pooling."""

    pool_size: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of standby clients to maintain in the pool",
    )

    max_pool_size: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of clients that can be created (pool + on-demand)",
    )

    connection_timeout: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Timeout in seconds when acquiring a client from the pool",
    )

    idle_timeout: float = Field(
        default=300.0,
        ge=60.0,
        le=3600.0,
        description="Time in seconds before idle clients are removed from the pool",
    )

    health_check_interval: float = Field(
        default=60.0,
        ge=10.0,
        le=600.0,
        description="Interval in seconds between health checks of pooled clients",
    )

    enable_health_checks: bool = Field(
        default=True,
        description="Whether to enable background health checks for pooled clients",
    )

    @field_validator("max_pool_size")
    @classmethod
    def validate_max_pool_size(cls, v: int, info: Any) -> int:
        """Ensure max_pool_size is at least as large as pool_size."""
        if info.data and "pool_size" in info.data:
            pool_size = info.data["pool_size"]
            if v < pool_size:
                raise ValueError(
                    f"max_pool_size ({v}) must be >= pool_size ({pool_size})"
                )
        return v


class ClaudeSettings(BaseModel):
    """Claude-specific configuration settings."""

    cli_path: str | None = Field(
        default=None,
        description="Path to Claude CLI executable",
    )

    builtin_permissions: bool = Field(
        default=True,
        description="Whether to enable built-in permission handling infrastructure (MCP server and SSE endpoints). When disabled, users can still configure custom MCP servers and permission tools.",
    )

    code_options: ClaudeCodeOptions | None = Field(
        default=None,
        description="Claude Code SDK options configuration",
    )

    sdk_message_mode: SDKMessageMode = Field(
        default=SDKMessageMode.FORWARD,
        description="Mode for handling SDK messages from Claude SDK. Options: forward (direct SDK blocks), ignore (skip blocks), formatted (XML tags with JSON data)",
    )

    system_prompt_injection_mode: SystemPromptInjectionMode = Field(
        default=SystemPromptInjectionMode.MINIMAL,
        description="Mode for system prompt injection. Options: minimal (Claude Code ID only), full (all detected system messages)",
    )

    pretty_format: bool = Field(
        default=True,
        description="Whether to use pretty formatting (indented JSON, newlines after XML tags, unescaped content). When false: compact JSON, no newlines, escaped content between XML tags",
    )

    use_client_pool: bool = Field(
        default=False,
        description="Whether to enable Claude SDK client connection pooling for improved performance. When enabled, maintains a pool of pre-connected clients to reduce subprocess startup overhead.",
    )

    pool_settings: ClaudePoolSettings = Field(
        default_factory=ClaudePoolSettings,
        description="Configuration settings for Claude SDK client connection pooling. Only used when use_client_pool=True.",
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
    def validate_claude_code_options(cls, v: Any, info: Any) -> Any:
        """Validate and convert Claude Code options."""
        # Get builtin_permissions setting from the model data
        builtin_permissions = True  # default
        if info.data and "builtin_permissions" in info.data:
            builtin_permissions = info.data["builtin_permissions"]

        if v is None:
            # Create instance with default values based on builtin_permissions
            return _create_default_claude_code_options(builtin_permissions)

        # If it's already a ClaudeCodeOptions instance, return as-is
        if isinstance(v, ClaudeCodeOptions):
            return v

        # If it's an empty dict, treat it like None and use defaults
        if isinstance(v, dict) and not v:
            return _create_default_claude_code_options(builtin_permissions)

        # For non-empty dicts, merge with defaults instead of replacing them
        if isinstance(v, dict):
            # Start with default values based on builtin_permissions
            defaults = _create_default_claude_code_options(builtin_permissions)

            # Extract default values as a dict for merging
            default_values = {
                "mcp_servers": dict(defaults.mcp_servers)
                if defaults.mcp_servers
                else {},
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

            # Handle MCP server merging when builtin_permissions is enabled
            if builtin_permissions and "mcp_servers" in v:
                user_mcp_servers = v["mcp_servers"]
                if isinstance(user_mcp_servers, dict):
                    # Merge user MCP servers with built-in ones (user takes precedence)
                    default_mcp = default_values["mcp_servers"]
                    if isinstance(default_mcp, dict):
                        merged_mcp_servers = {
                            **default_mcp,
                            **user_mcp_servers,
                        }
                        v = {**v, "mcp_servers": merged_mcp_servers}

            # Merge CLI overrides with defaults (CLI overrides take precedence)
            merged_values = {**default_values, **v}

            return ClaudeCodeOptions(**merged_values)

        # Try to convert to ClaudeCodeOptions if possible
        if hasattr(v, "model_dump"):
            return ClaudeCodeOptions(**v.model_dump())
        elif hasattr(v, "__dict__"):
            return ClaudeCodeOptions(**v.__dict__)

        # Fallback: use default values
        return _create_default_claude_code_options(builtin_permissions)

    @model_validator(mode="after")
    def validate_code_options_after(self) -> "ClaudeSettings":
        """Ensure code_options is properly initialized after field validation."""
        if self.code_options is None:
            self.code_options = _create_default_claude_code_options(
                self.builtin_permissions
            )
        return self

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
