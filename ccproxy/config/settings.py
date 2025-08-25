"""Settings configuration for Claude Proxy API Server."""

import contextlib
import json
import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ccproxy.config.discovery import find_toml_config_file
from ccproxy.core.logging import get_logger

from .auth import AuthSettings
from .binary import BinarySettings
from .cors import CORSSettings
from .docker_settings import DockerSettings
from .http import HTTPSettings
from .observability import ObservabilitySettings
from .reverse_proxy import ReverseProxySettings
from .scheduler import SchedulerSettings
from .security import SecuritySettings
from .server import ServerSettings


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


class LoggingHooksSettings(BaseModel):
    """Settings for logging hooks migration."""

    use_hook_logging: bool = True
    enable_access_logging: bool = True
    enable_content_logging: bool = True
    enable_streaming_logging: bool = True
    parallel_run_mode: bool = False
    disable_middleware_during_parallel: bool = False


class HooksSettings(BaseModel):
    """Hook system configuration."""

    enabled: bool = True
    metrics_enabled: bool = True
    logging_enabled: bool = True
    analytics_enabled: bool = False
    analytics_batch_size: int = 100
    enable_chunk_events: bool = False  # For streaming chunk events
    logging: LoggingHooksSettings = Field(default_factory=LoggingHooksSettings)


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

    # Core application settings
    server: ServerSettings = Field(
        default_factory=ServerSettings,
        description="Server configuration settings",
    )

    security: SecuritySettings = Field(
        default_factory=SecuritySettings,
        description="Security configuration settings",
    )

    cors: CORSSettings = Field(
        default_factory=CORSSettings,
        description="CORS configuration settings",
    )

    http: HTTPSettings = Field(
        default_factory=HTTPSettings,
        description="HTTP client configuration settings",
    )

    # Proxy and authentication
    reverse_proxy: ReverseProxySettings = Field(
        default_factory=ReverseProxySettings,
        description="Reverse proxy configuration settings",
    )

    auth: AuthSettings = Field(
        default_factory=AuthSettings,
        description="Authentication and credentials configuration",
    )

    # Binary resolution settings
    binary: BinarySettings = Field(
        default_factory=BinarySettings,
        description="Binary resolution and package manager fallback configuration",
    )

    # Container settings
    docker: DockerSettings = Field(
        default_factory=DockerSettings,
        description="Docker configuration for running Claude commands in containers",
    )

    # Observability settings
    observability: ObservabilitySettings = Field(
        default_factory=ObservabilitySettings,
        description="Observability configuration settings",
    )

    # Scheduler settings
    scheduler: SchedulerSettings = Field(
        default_factory=SchedulerSettings,
        description="Task scheduler configuration settings",
    )

    # Plugin settings
    plugin_dir: str = Field(
        default="plugins",
        description="Directory to load plugins from",
    )

    enable_plugins: bool = Field(
        default=True,
        description="Enable plugin system",
    )

    # Plugin configurations
    plugins: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Plugin-specific configurations keyed by plugin name",
    )

    # Hook system settings
    hooks: HooksSettings = Field(
        default_factory=HooksSettings,
        description="Hook system configuration settings",
    )

    # Redundant validators removed - Pydantic handles these automatically with default_factory

    @property
    def server_url(self) -> str:
        """Get the complete server URL."""
        return f"http://{self.server.host}:{self.server.port}"

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.server.reload or self.server.log_level == "DEBUG"

    def model_dump_safe(self) -> dict[str, Any]:
        """
        Dump model data with sensitive information masked.

        Returns:
            dict: Configuration with sensitive data masked
        """
        # Use serialization mode that properly handles SecretStr
        return self.model_dump(mode="json")

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
        else:
            raise ValueError(
                f"Unsupported config file format: {suffix}. "
                "Only TOML (.toml) files are supported."
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
            # Log loaded config only if not already logged by config manager
            if not config_manager._config_logged:
                logger = get_logger(__name__)
                logger.info(
                    "config_file_loaded",
                    path=str(config_path),
                    category="config",
                )
                config_manager.mark_config_logged()

        # Merge config with kwargs (kwargs take precedence)
        merged_config = {**config_data, **kwargs}

        # Create Settings instance with merged config
        settings = cls(**merged_config)

        return settings


class ConfigurationManager:
    """Centralized configuration management for CLI and server."""

    def __init__(self) -> None:
        self._settings: Settings | None = None
        self._config_path: Path | None = None
        self._logging_configured = False
        self._config_logged = False

    def _apply_plugin_settings_overrides(
        self, settings: dict[str, Any], overrides: list[str]
    ) -> None:
        """Apply plugin settings overrides from the CLI."""
        if not overrides:
            return

        if "plugins" not in settings:
            settings["plugins"] = {}

        for override in overrides:
            try:
                key, value = override.split("=", 1)
                plugin_name, setting_key = key.split(".", 1)

                # Convert value to appropriate type
                if value.lower() == "true":
                    typed_value: Any = True
                elif value.lower() == "false":
                    typed_value = False
                elif value.isdigit():
                    typed_value = int(value)
                else:
                    try:
                        typed_value = float(value)
                    except ValueError:
                        typed_value = value

                # Update nested dictionaries
                plugin_settings = settings["plugins"].setdefault(plugin_name, {})
                keys = setting_key.split(".")
                current_level = plugin_settings
                for k in keys[:-1]:
                    current_level = current_level.setdefault(k, {})
                current_level[keys[-1]] = typed_value

            except ValueError:
                logger.warning(
                    f"Invalid plugin setting format: {override}", category="config"
                )

    def mark_config_logged(self) -> None:
        """Mark that config loading has been logged to prevent duplicates."""
        self._config_logged = True

    def load_settings(
        self,
        config_path: Path | None = None,
        cli_overrides: dict[str, Any] | None = None,
    ) -> Settings:
        """Load settings with CLI overrides and caching."""
        if self._settings is None or config_path != self._config_path:
            try:
                # Load base settings from file
                config_data = {}
                if config_path and config_path.exists():
                    config_data = Settings.load_config_file(config_path)
                    # Log config loading once
                    if not self._config_logged:
                        logger.info(
                            "config_file_loaded",
                            path=str(config_path),
                            category="config",
                        )
                        self._config_logged = True

                # Apply CLI overrides to the loaded config data
                if cli_overrides:
                    # Apply plugin settings overrides
                    plugin_settings_overrides = cli_overrides.pop("plugin_settings", [])
                    self._apply_plugin_settings_overrides(
                        config_data, plugin_settings_overrides
                    )

                    # Merge other CLI overrides
                    for key, value in cli_overrides.items():
                        if isinstance(value, dict) and isinstance(
                            config_data.get(key), dict
                        ):
                            config_data[key].update(value)
                        else:
                            config_data[key] = value

                self._settings = Settings(**config_data)
                self._config_path = config_path

            except Exception as e:
                raise ConfigurationError(f"Failed to load configuration: {e}") from e

        return self._settings

    def setup_logging(self, log_level: str | None = None) -> None:
        """Configure logging once based on settings."""
        if self._logging_configured:
            return

        # Import here to avoid circular import

        effective_level = log_level or (
            self._settings.server.log_level if self._settings else "INFO"
        )

        # Determine format based on log level - Rich for DEBUG, JSON for production
        format_type = "rich" if effective_level.upper() == "DEBUG" else "json"

        # setup_dual_logging(
        #     level=effective_level,
        #     format_type=format_type,
        #     configure_uvicorn=True,
        #     verbose_tracebacks=effective_level.upper() == "DEBUG",
        # )
        self._logging_configured = True

    def get_cli_overrides_from_args(self, **cli_args: Any) -> dict[str, Any]:
        """Extract non-None CLI arguments as configuration overrides."""
        overrides = {}

        # Server settings
        server_settings = {}
        for key in ["host", "port", "reload", "log_level", "log_file"]:
            if cli_args.get(key) is not None:
                server_settings[key] = cli_args[key]
        if server_settings:
            overrides["server"] = server_settings

        # Security settings
        if cli_args.get("auth_token") is not None:
            overrides["security"] = {"auth_token": cli_args["auth_token"]}

        # CORS settings
        if cli_args.get("cors_origins"):
            overrides["cors"] = {
                "origins": [
                    origin.strip() for origin in cli_args["cors_origins"].split(",")
                ]
            }

        # Plugin settings
        if cli_args.get("plugin_setting"):
            overrides["plugin_settings"] = cli_args["plugin_setting"]

        return overrides

    def reset(self) -> None:
        """Reset configuration state (useful for testing)."""
        self._settings = None
        self._config_path = None
        self._logging_configured = False
        self._config_logged = False


# Global configuration manager instance
config_manager = ConfigurationManager()

logger = get_logger(__name__)


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

        settings = Settings.from_config(config_path=config_path, **cli_overrides)
        return settings
    except Exception as e:
        # If settings can't be loaded (e.g., missing API key),
        # this will be handled by the caller
        raise ValueError(f"Configuration error: {e}") from e
