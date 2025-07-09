"""Settings configuration for Claude Proxy API Server."""

import contextlib
import json
import os
import shutil
import tomllib
from pathlib import Path
from typing import Any, Literal

from ccproxy import __version__
from ccproxy.utils.version import format_version


try:
    import yaml  # type: ignore[import-untyped]

    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ccproxy.services.credentials import CredentialsConfig
from ccproxy.utils import find_toml_config_file, get_claude_cli_config_dir
from ccproxy.utils.helper import get_package_dir, patched_typing

from .docker_settings import DockerSettings


__all__ = [
    "Settings",
    "ConfigurationError",
    "ConfigurationManager",
    "config_manager",
    "get_settings",
]


class ConfigurationError(Exception):
    """Raised when configuration loading or validation fails."""

    pass


# For further information visit https://errors.pydantic.dev/2.11/u/typed-dict-version
with patched_typing():
    from claude_code_sdk import ClaudeCodeOptions  # noqa: E402


# PoolSettings class removed - connection pooling functionality has been removed


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
        env_nested_delimiter="__",
    )

    # Server settings
    host: str = Field(
        default="127.0.0.1",
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

    cors_credentials: bool = Field(
        default=True,
        description="CORS allow credentials",
    )

    cors_methods: list[str] = Field(
        default_factory=lambda: ["*"],
        description="CORS allowed methods",
    )

    cors_headers: list[str] = Field(
        default_factory=lambda: ["*"],
        description="CORS allowed headers",
    )

    cors_origin_regex: str | None = Field(
        default=None,
        description="CORS origin regex pattern",
    )

    cors_expose_headers: list[str] = Field(
        default_factory=list,
        description="CORS exposed headers",
    )

    cors_max_age: int = Field(
        default=600,
        description="CORS preflight max age in seconds",
        ge=0,
    )

    auth_token: str | None = Field(
        default=None,
        description="Bearer token for API authentication (optional)",
    )

    # Tools handling behavior
    api_tools_handling: Literal["error", "warning", "ignore"] = Field(
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

    # Reverse Proxy settings
    reverse_proxy_target_url: str = Field(
        default="https://api.anthropic.com",
        description="Target URL for reverse proxy requests",
    )

    reverse_proxy_timeout: float = Field(
        default=120.0,
        description="Timeout for reverse proxy requests in seconds",
        ge=1.0,
        le=600.0,
    )

    # Reverse proxy mode configuration
    default_proxy_mode: Literal["claude_code", "full", "minimal"] = Field(
        default="claude_code",
        description="Default transformation mode for root path reverse proxy, over claude code or auth injection with full",
    )

    # Claude Code SDK endpoint configuration
    claude_code_prefix: str = Field(
        default="/cc",
        description="URL prefix for Claude Code SDK endpoints",
    )

    # Credentials configuration
    credentials: CredentialsConfig = Field(
        default_factory=CredentialsConfig,
        description="Credentials management configuration",
    )

    # Pool settings removed - connection pooling functionality has been removed

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

    @field_validator("credentials", mode="before")
    @classmethod
    def validate_credentials(cls, v: Any) -> Any:
        """Validate and convert credentials configuration."""
        if v is None:
            return CredentialsConfig()

        # If it's already a CredentialsConfig instance, return as-is
        if isinstance(v, CredentialsConfig):
            return v

        # If it's a dict, create CredentialsConfig from it
        if isinstance(v, dict):
            return CredentialsConfig(**v)

        # Try to convert to dict if possible
        if hasattr(v, "model_dump"):
            return CredentialsConfig(**v.model_dump())
        elif hasattr(v, "__dict__"):
            return CredentialsConfig(**v.__dict__)

        return v

    # validate_pool_settings method removed - connection pooling functionality has been removed

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

    @field_validator("cors_methods", mode="before")
    @classmethod
    def validate_cors_methods(cls, v: str | list[str]) -> list[str]:
        """Parse CORS methods from string or list."""
        if isinstance(v, str):
            # Split comma-separated string
            return [method.strip().upper() for method in v.split(",") if method.strip()]
        return [method.upper() for method in v]

    @field_validator("cors_headers", mode="before")
    @classmethod
    def validate_cors_headers(cls, v: str | list[str]) -> list[str]:
        """Parse CORS headers from string or list."""
        if isinstance(v, str):
            # Split comma-separated string
            return [header.strip() for header in v.split(",") if header.strip()]
        return v

    @field_validator("cors_expose_headers", mode="before")
    @classmethod
    def validate_cors_expose_headers(cls, v: str | list[str]) -> list[str]:
        """Parse CORS expose headers from string or list."""
        if isinstance(v, str):
            # Split comma-separated string
            return [header.strip() for header in v.split(",") if header.strip()]
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
    def load_json_config(cls, json_path: Path) -> dict[str, Any]:
        """Load configuration from a JSON file.

        Args:
            json_path: Path to the JSON configuration file

        Returns:
            dict: Configuration data from the JSON file

        Raises:
            ValueError: If the JSON file is invalid or cannot be read
        """
        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except OSError as e:
            raise ValueError(f"Cannot read JSON config file {json_path}: {e}") from e
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON syntax in {json_path}: {e}") from e

    @classmethod
    def load_yaml_config(cls, yaml_path: Path) -> dict[str, Any]:
        """Load configuration from a YAML file.

        Args:
            yaml_path: Path to the YAML configuration file

        Returns:
            dict: Configuration data from the YAML file

        Raises:
            ValueError: If the YAML file is invalid or cannot be read
        """
        if not HAS_YAML:
            raise ValueError(
                "YAML support is not available. Install with: pip install pyyaml"
            )

        try:
            with yaml_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                return data if isinstance(data, dict) else {}
        except OSError as e:
            raise ValueError(f"Cannot read YAML config file {yaml_path}: {e}") from e
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML syntax in {yaml_path}: {e}") from e

    @classmethod
    def load_config_file(cls, config_path: Path) -> dict[str, Any]:
        """Load configuration from a file based on its extension.

        Args:
            config_path: Path to the configuration file

        Returns:
            dict: Configuration data from the file

        Raises:
            ValueError: If the file format is unsupported or invalid
        """
        suffix = config_path.suffix.lower()

        if suffix in [".toml"]:
            return cls.load_toml_config(config_path)
        elif suffix in [".json"]:
            return cls.load_json_config(config_path)
        elif suffix in [".yaml", ".yml"]:
            return cls.load_yaml_config(config_path)
        else:
            raise ValueError(
                f"Unsupported config file format: {suffix}. "
                "Supported formats: .toml, .json, .yaml, .yml"
            )

    @classmethod
    def from_toml(cls, toml_path: Path | None = None, **kwargs: Any) -> "Settings":
        """Create Settings instance from TOML configuration.

        Args:
            toml_path: Path to TOML configuration file. If None, auto-discovers file.
            **kwargs: Additional keyword arguments to override config values

        Returns:
            Settings: Configured Settings instance
        """
        # Use the more generic from_config method
        return cls.from_config(config_path=toml_path, **kwargs)

    @classmethod
    def from_config(
        cls, config_path: Path | str | None = None, **kwargs: Any
    ) -> "Settings":
        """Create Settings instance from configuration file.

        Args:
            config_path: Path to configuration file. Can be:
                - None: Auto-discover config file or use CONFIG_FILE env var
                - Path or str: Use this specific config file
            **kwargs: Additional keyword arguments to override config values

        Returns:
            Settings: Configured Settings instance
        """
        # Check for CONFIG_FILE environment variable first
        if config_path is None:
            config_path_env = os.environ.get("CONFIG_FILE")
            if config_path_env:
                config_path = Path(config_path_env)

        # Convert string to Path if needed
        if isinstance(config_path, str):
            config_path = Path(config_path)

        # Auto-discover config file if not provided
        if config_path is None:
            config_path = find_toml_config_file()

        # Load config if found
        config_data = {}
        if config_path and config_path.exists():
            config_data = cls.load_config_file(config_path)

        # Merge config with kwargs (kwargs take precedence)
        merged_config = {**config_data, **kwargs}

        # Create Settings instance with merged config
        return cls(**merged_config)


class ConfigurationManager:
    """Centralized configuration management for CLI and server."""

    def __init__(self) -> None:
        self._settings: Settings | None = None
        self._config_path: Path | None = None
        self._logging_configured = False

    def load_settings(
        self,
        config_path: Path | None = None,
        cli_overrides: dict[str, Any] | None = None,
    ) -> Settings:
        """Load settings with CLI overrides and caching."""
        if self._settings is None or config_path != self._config_path:
            try:
                self._settings = Settings.from_config(
                    config_path=config_path, **(cli_overrides or {})
                )
                self._config_path = config_path
            except Exception as e:
                raise ConfigurationError(f"Failed to load configuration: {e}") from e

        return self._settings

    def setup_logging(self, log_level: str | None = None) -> None:
        """Configure logging once based on settings."""
        if self._logging_configured:
            return

        # Import here to avoid circular import
        from ccproxy.utils.logging import setup_rich_logging

        effective_level = log_level or (
            self._settings.log_level if self._settings else "INFO"
        )

        setup_rich_logging(level=effective_level)
        self._logging_configured = True

    def get_cli_overrides_from_args(self, **cli_args: Any) -> dict[str, Any]:
        """Extract non-None CLI arguments as configuration overrides."""
        overrides = {}

        # Server settings
        for key in [
            "host",
            "port",
            "reload",
            "log_level",
            "auth_token",
            "claude_cli_path",
        ]:
            if cli_args.get(key) is not None:
                overrides[key] = cli_args[key]

        # Claude Code options
        claude_opts = {}
        for key in [
            "max_thinking_tokens",
            "permission_mode",
            "cwd",
            "max_turns",
            "append_system_prompt",
            "permission_prompt_tool_name",
            "continue_conversation",
        ]:
            if cli_args.get(key) is not None:
                claude_opts[key] = cli_args[key]

        # Handle comma-separated lists
        for key, target_key in [
            ("allowed_tools", "allowed_tools"),
            ("disallowed_tools", "disallowed_tools"),
            ("cors_origins", "cors_origins"),
        ]:
            if cli_args.get(key):
                if key == "cors_origins":
                    overrides["cors_origins"] = [
                        origin.strip() for origin in cli_args[key].split(",")
                    ]
                else:
                    claude_opts[target_key] = [
                        tool.strip() for tool in cli_args[key].split(",")
                    ]

        if claude_opts:
            overrides["claude_code_options"] = claude_opts

        return overrides

    def reset(self) -> None:
        """Reset configuration state (useful for testing)."""
        self._settings = None
        self._config_path = None
        self._logging_configured = False


# Global configuration manager instance
config_manager = ConfigurationManager()


def get_settings(config_path: Path | str | None = None) -> Settings:
    """Get the global settings instance with configuration file support.

    Args:
        config_path: Optional path to configuration file. If None, uses CONFIG_FILE env var
                    or auto-discovers config file.

    Returns:
        Settings: Configured Settings instance
    """
    try:
        # Check for CLI overrides from environment variable
        cli_overrides = {}
        cli_overrides_json = os.environ.get("CCPROXY_CONFIG_OVERRIDES")
        if cli_overrides_json:
            with contextlib.suppress(json.JSONDecodeError):
                cli_overrides = json.loads(cli_overrides_json)

        return Settings.from_config(config_path=config_path, **cli_overrides)
    except Exception as e:
        # If settings can't be loaded (e.g., missing API key),
        # this will be handled by the caller
        raise ValueError(f"Configuration error: {e}") from e
