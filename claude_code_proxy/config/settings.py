"""Settings configuration for Claude Proxy API Server."""

import os
import shutil
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from claude_code_proxy.utils import find_toml_config_file, get_claude_cli_config_dir
from claude_code_proxy.utils.helper import get_package_dir, patched_typing


# For further information visit https://errors.pydantic.dev/2.11/u/typed-dict-version
with patched_typing():
    from claude_code_sdk import ClaudeCodeOptions  # noqa: E402


class DockerSettings(BaseModel):
    """Docker configuration settings for running Claude commands in containers."""

    docker_image: str = Field(
        default="claude-code-proxy",
        description="Docker image to use for Claude commands",
    )

    docker_volumes: list[str] = Field(
        default_factory=list,
        description="List of volume mounts in 'host:container[:options]' format",
    )

    docker_environment: dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables to pass to Docker container",
    )

    docker_additional_args: list[str] = Field(
        default_factory=list,
        description="Additional arguments to pass to docker run command",
    )

    docker_home_directory: str | None = Field(
        default=None,
        description="Local host directory to mount as the home directory in container",
    )

    docker_workspace_directory: str | None = Field(
        default=None,
        description="Local host directory to mount as the workspace directory in container",
    )

    @field_validator("docker_volumes")
    @classmethod
    def validate_docker_volumes(cls, v: list[str]) -> list[str]:
        """Validate Docker volume mount format."""
        for volume in v:
            if ":" not in volume:
                raise ValueError(
                    f"Invalid volume format: '{volume}'. Expected 'host:container[:options]'"
                )
            parts = volume.split(":")
            if len(parts) < 2:
                raise ValueError(
                    f"Invalid volume format: '{volume}'. Expected 'host:container[:options]'"
                )
            # Convert relative paths to absolute and validate
            host_path = os.path.expandvars(parts[0])
            path_obj = Path(host_path)

            # If it's a relative path, convert to absolute
            if not path_obj.is_absolute():
                host_path = str(path_obj.resolve())

            # Check if the absolute path exists
            if not Path(host_path).exists():
                raise ValueError(f"Host path does not exist: '{host_path}'")
        return v

    @field_validator("docker_home_directory")
    @classmethod
    def validate_docker_home_directory(cls, v: str | None) -> str | None:
        """Validate and normalize Docker home directory (host path)."""
        if v is None:
            return None

        # Expand environment variables
        expanded_path = os.path.expandvars(v)
        path_obj = Path(expanded_path)

        # If it's a relative path, convert to absolute
        if not path_obj.is_absolute():
            v = str(path_obj.resolve())
        else:
            v = expanded_path

        return v

    @field_validator("docker_workspace_directory")
    @classmethod
    def validate_docker_workspace_directory(cls, v: str | None) -> str | None:
        """Validate and normalize Docker workspace directory (host path)."""
        if v is None:
            return None

        # Expand environment variables
        expanded_path = os.path.expandvars(v)
        path_obj = Path(expanded_path)

        # If it's a relative path, convert to absolute
        if not path_obj.is_absolute():
            v = str(path_obj.resolve())
        else:
            v = expanded_path

        return v

    @model_validator(mode="after")
    def setup_docker_volumes(self) -> "DockerSettings":
        """Set up Docker volumes based on home and workspace directories."""
        # Create default volumes if none are explicitly set and no custom directories
        if (
            not self.docker_volumes
            and not self.docker_home_directory
            and not self.docker_workspace_directory
        ):
            # Use XDG config directory for Claude CLI data
            claude_config_dir = get_claude_cli_config_dir()
            home_host_path = str(claude_config_dir)
            workspace_host_path = os.path.expandvars("$PWD")

            self.docker_volumes = [
                f"{home_host_path}:/data/home",
                f"{workspace_host_path}:/data/workspace",
            ]

        # Update environment variables to point to container paths
        if "CLAUDE_HOME" not in self.docker_environment:
            self.docker_environment["CLAUDE_HOME"] = "/data/home"
        if "CLAUDE_WORKSPACE" not in self.docker_environment:
            self.docker_environment["CLAUDE_WORKSPACE"] = "/data/workspace"

        return self

    def get_docker_command_args(self) -> list[str]:
        """Generate Docker command arguments from settings."""
        args = ["docker", "run", "--rm", "-it"]

        # Add volumes
        for volume in self.docker_volumes:
            args.extend(["--volume", volume])

        # Add environment variables
        for key, value in self.docker_environment.items():
            args.extend(["--env", f"{key}={value}"])

        # Add additional arguments
        args.extend(self.docker_additional_args)

        # Add image
        args.append(self.docker_image)

        return args


class Settings(BaseSettings):
    """
    Configuration settings for the Claude Proxy API Server.

    Settings are loaded from environment variables, .env files, and TOML configuration files.
    Environment variables take precedence over .env file values.
    TOML configuration files are loaded in the following order:
    1. .ccproxy.toml in current directory
    2. ccproxy.toml in git repository root
    3. config.toml in XDG_CONFIG_HOME/ccproxy/
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
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

    auth_token: str | None = Field(
        default=None,
        description="Bearer token for API authentication (optional)",
    )

    # Configuration file path
    config_file: Path | None = Field(
        default=None,
        description="Path to JSON configuration file",
    )

    # Tools handling behavior
    tools_handling: Literal["error", "warning", "ignore"] = Field(
        default="warning",
        description="How to handle tools definitions in requests: error, warning, or ignore",
    )

    # Claude CLI path
    claude_cli_path: str | None = Field(
        default=None,
        description="Path to Claude CLI executable",
    )

    # Claude Code SDK Options
    claude_code_options: ClaudeCodeOptions = Field(
        default_factory=lambda: ClaudeCodeOptions(),
        description="Claude Code SDK options configuration",
    )

    # Docker settings
    docker_settings: DockerSettings = Field(
        default_factory=DockerSettings,
        description="Docker configuration for running Claude commands in containers",
    )

    @field_validator("claude_code_options", mode="before")
    @classmethod
    def validate_claude_code_options(cls, v: Any) -> Any:
        """Validate and convert Claude Code options."""
        if v is None:
            return ClaudeCodeOptions()

        # If it's already a ClaudeCodeOptions instance, return as-is
        if isinstance(v, ClaudeCodeOptions):
            return v

        # Try to convert to dict if possible
        if hasattr(v, "model_dump"):
            return v.model_dump()
        elif hasattr(v, "__dict__"):
            return v.__dict__

        return v

    @field_validator("docker_settings", mode="before")
    @classmethod
    def validate_docker_settings(cls, v: Any) -> Any:
        """Validate and convert Docker settings."""
        if v is None:
            return DockerSettings()

        # If it's already a DockerSettings instance, return as-is
        if isinstance(v, DockerSettings):
            return v

        # If it's a dict, create DockerSettings from it
        if isinstance(v, dict):
            return DockerSettings(**v)

        # Try to convert to dict if possible
        if hasattr(v, "model_dump"):
            return DockerSettings(**v.model_dump())
        elif hasattr(v, "__dict__"):
            return DockerSettings(**v.__dict__)

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

    @field_validator("claude_code_options", mode="before")
    @classmethod
    def validate_claude_cwd(cls, v: ClaudeCodeOptions) -> ClaudeCodeOptions:
        if v.cwd is None:
            v.cwd = Path.cwd()
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

    def model_dump_safe(self) -> dict[str, Any]:
        """
        Dump model data with sensitive information masked.

        Returns:
            dict: Configuration with sensitive data masked
        """
        return self.model_dump()

    @classmethod
    def load_toml_config(cls, toml_path: Path) -> dict[str, Any]:
        """Load configuration from a TOML file.

        Args:
            toml_path: Path to the TOML configuration file

        Returns:
            dict: Configuration data from the TOML file

        Raises:
            ValueError: If the TOML file is invalid or cannot be read
        """
        try:
            with toml_path.open("rb") as f:
                return tomllib.load(f)
        except OSError as e:
            raise ValueError(f"Cannot read TOML config file {toml_path}: {e}") from e
        except tomllib.TOMLDecodeError as e:
            raise ValueError(f"Invalid TOML syntax in {toml_path}: {e}") from e

    @classmethod
    def from_toml(cls, toml_path: Path | None = None, **kwargs: Any) -> "Settings":
        """Create Settings instance from TOML configuration.

        Args:
            toml_path: Path to TOML configuration file. If None, auto-discovers file.
            **kwargs: Additional keyword arguments to override config values

        Returns:
            Settings: Configured Settings instance
        """
        # Auto-discover TOML config file if not provided
        if toml_path is None:
            toml_path = find_toml_config_file()

        # Load TOML config if found
        toml_config = {}
        if toml_path and toml_path.exists():
            toml_config = cls.load_toml_config(toml_path)

        # Merge TOML config with kwargs (kwargs take precedence)
        merged_config = {**toml_config, **kwargs}

        # Create Settings instance with merged config
        return cls(**merged_config)


def get_settings() -> Settings:
    """Get the global settings instance with TOML configuration support."""
    try:
        return Settings.from_toml()
    except Exception as e:
        # If settings can't be loaded (e.g., missing API key),
        # this will be handled by the caller
        raise ValueError(f"Configuration error: {e}") from e
