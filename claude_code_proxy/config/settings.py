"""Settings configuration for Claude Proxy API Server."""

import os
import shutil
from pathlib import Path
from typing import Any, Literal

from claude_code_sdk import ClaudeCodeOptions
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


try:
    import importlib.util

    # Get the path to the claude_code_proxy package and resolve it
    spec = importlib.util.find_spec("claude_code_proxy")
    if spec and spec.origin:
        PACKAGE_DIR = Path(spec.origin).parent.parent.resolve()
    else:
        PACKAGE_DIR = Path(__file__).parent.parent.parent.resolve()
except Exception:
    PACKAGE_DIR = Path(__file__).parent.parent.parent.resolve()


class Settings(BaseSettings):
    """
    Configuration settings for the Claude Proxy API Server.

    Settings are loaded from environment variables and .env files.
    Environment variables take precedence over .env file values.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Claude Code SDK handles authentication directly

    # Claude Code CLI settings
    claude_cli_path: str | None = Field(
        default=None,
        description="Path to Claude Code CLI executable (e.g., /usr/local/bin/claude)",
    )

    # Server settings
    host: str = Field(
        default="0.0.0.0",
        description="Server host address",
    )

    port: int = Field(
        default=8000,
        description="Server port number",
        ge=1,
        le=65535,
    )

    # Logging settings
    log_level: str = Field(
        default="INFO",
        description="Logging level",
    )

    # Optional server settings
    workers: int = Field(
        default=1,
        description="Number of worker processes",
        ge=1,
        le=32,
    )

    reload: bool = Field(
        default=False,
        description="Enable auto-reload for development",
    )

    # Security settings
    cors_origins: list[str] = Field(
        default_factory=lambda: ["*"],
        description="CORS allowed origins",
    )

    # Configuration file path
    config_file: Path | None = Field(
        default=None,
        description="Path to JSON configuration file",
    )

    # Tools handling behavior
    tools_handling: Literal["error", "warning", "ignore"] = Field(
        default="error",
        description="How to handle tools definitions in requests: error, warning, or ignore",
    )

    # Security settings for subprocess execution
    claude_user: str | None = Field(
        default="claude",
        description="Username to drop privileges to when executing Claude subprocess",
    )

    claude_group: str | None = Field(
        default="claude",
        description="Group name to drop privileges to when executing Claude subprocess",
    )

    claude_working_directory: str = Field(
        default="/tmp/claude-workspace",
        description="Working directory for Claude subprocess execution",
    )

    # Claude Code SDK Options
    claude_code_options: ClaudeCodeOptions | dict[str, Any] = Field(
        default_factory=dict,
        description="Claude Code SDK options configuration",
    )

    @field_validator("claude_code_options", mode="before")
    @classmethod
    def validate_claude_code_options(cls, v: Any) -> Any:
        """Validate and convert Claude Code options."""
        if v is None:
            return {}

        # If it's already a ClaudeCodeOptions instance or dict, return as-is
        if isinstance(v, dict | type(None)):
            return v

        # If ClaudeCodeOptions is available and it's an instance, return as-is
        if ClaudeCodeOptions and isinstance(v, ClaudeCodeOptions):
            return v

        # Try to convert to dict if possible
        if hasattr(v, "model_dump"):
            return v.model_dump()
        elif hasattr(v, "__dict__"):
            return v.__dict__

        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate and normalize log level."""
        upper_v = v.upper()
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if upper_v not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return upper_v

    @field_validator("cors_origins", mode="before")
    @classmethod
    def validate_cors_origins(cls, v: str | list[str]) -> list[str]:
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            # Split comma-separated string
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @field_validator("config_file", mode="before")
    @classmethod
    def validate_config_file(cls, v: str | Path | None) -> Path | None:
        """Convert string path to Path object."""
        if v is None:
            return None
        if isinstance(v, str):
            return Path(v)
        return v

    @property
    def server_url(self) -> str:
        """Get the complete server URL."""
        return f"http://{self.host}:{self.port}"

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.reload or self.log_level == "DEBUG"

    @field_validator("claude_cli_path")
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

    @model_validator(mode="after")
    def setup_claude_cli_path(self) -> "Settings":
        """Set up Claude CLI path in environment if provided or found."""
        # If not explicitly set, try to find it
        if not self.claude_cli_path:
            found_path, found_in_path = self.find_claude_cli()
            if found_path:
                self.claude_cli_path = found_path
                # Only add to PATH if it wasn't found via which()
                if not found_in_path:
                    cli_dir = str(Path(self.claude_cli_path).parent)
                    current_path = os.environ.get("PATH", "")
                    if cli_dir not in current_path:
                        os.environ["PATH"] = f"{cli_dir}:{current_path}"
        elif self.claude_cli_path:
            # If explicitly set, always add to PATH
            cli_dir = str(Path(self.claude_cli_path).parent)
            current_path = os.environ.get("PATH", "")
            if cli_dir not in current_path:
                os.environ["PATH"] = f"{cli_dir}:{current_path}"
        return self

    def find_claude_cli(self) -> tuple[str | None, bool]:
        """Find Claude CLI executable in PATH or specified location.

        Returns:
            tuple: (path_to_claude, found_in_path)
        """
        if self.claude_cli_path:
            return self.claude_cli_path, False

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
            PACKAGE_DIR / "node_modules" / ".bin" / "claude",
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
            PACKAGE_DIR / "node_modules" / ".bin" / "claude",
            # Current working directory node_modules
            Path.cwd() / "node_modules" / ".bin" / "claude",
            # System-wide installations
            Path("/usr/local/bin/claude"),
            Path("/opt/homebrew/bin/claude"),
        ]

        for path in common_paths:
            paths.append(str(path))

        return paths

    def model_dump_safe(self) -> dict[str, Any]:
        """
        Dump model data with sensitive information masked.

        Returns:
            dict: Configuration with sensitive data masked
        """
        return self.model_dump()

    def get_claude_code_options(self) -> ClaudeCodeOptions | None:
        """
        Get ClaudeCodeOptions instance from settings.

        Returns:
            ClaudeCodeOptions instance if SDK is available, None otherwise
        """
        if not ClaudeCodeOptions:
            return None

        # If it's already a ClaudeCodeOptions instance, return it
        if isinstance(self.claude_code_options, ClaudeCodeOptions):
            return self.claude_code_options

        # If it's a dict, create ClaudeCodeOptions from it
        if isinstance(self.claude_code_options, dict):
            return ClaudeCodeOptions(**self.claude_code_options)

        return None

    def get_claude_code_options_dict(self) -> dict[str, Any]:
        """
        Get Claude Code options as a dictionary.

        Returns:
            dict: Configuration for ClaudeCodeOptions
        """
        if isinstance(self.claude_code_options, dict):
            return self.claude_code_options
        elif ClaudeCodeOptions and isinstance(
            self.claude_code_options, ClaudeCodeOptions
        ):
            # Convert ClaudeCodeOptions to dict if it has model_dump method
            if hasattr(self.claude_code_options, "model_dump"):
                return self.claude_code_options.model_dump()
            elif hasattr(self.claude_code_options, "__dict__"):
                return self.claude_code_options.__dict__

        return {}


def get_settings() -> Settings:
    """Get the global settings instance."""
    try:
        return Settings()
    except Exception as e:
        # If settings can't be loaded (e.g., missing API key),
        # this will be handled by the caller
        raise ValueError(f"Configuration error: {e}") from e
