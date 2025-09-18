import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ccproxy.core.logging import get_logger

from .core import CORSSettings, HTTPSettings, LoggingSettings, ServerSettings
from .runtime import BinarySettings
from .security import AuthSettings, SecuritySettings
from .utils import SchedulerSettings, find_toml_config_file


def _auth_default() -> AuthSettings:
    return AuthSettings()  # type: ignore[call-arg]


__all__ = ["Settings", "ConfigurationError"]


class ConfigurationError(Exception):
    """Raised when configuration loading or validation fails."""

    pass


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

    server: ServerSettings = Field(
        default_factory=ServerSettings,
        description="Server configuration settings",
    )

    logging: LoggingSettings = Field(
        default_factory=LoggingSettings,
        description="Centralized logging configuration",
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

    auth: AuthSettings = Field(
        default_factory=_auth_default,
        description="Authentication manager settings (e.g., credentials caching)",
    )

    binary: BinarySettings = Field(
        default_factory=BinarySettings,
        description="Binary resolution and package manager fallback configuration",
    )

    scheduler: SchedulerSettings = Field(
        default_factory=SchedulerSettings,
        description="Task scheduler configuration settings",
    )

    enable_plugins: bool = Field(
        default=True,
        description="Enable plugin system",
    )

    plugins_disable_local_discovery: bool = Field(
        default=True,
        description=(
            "If true, skip filesystem plugin discovery from the local 'plugins/' directory "
            "and load plugins only from installed entry points."
        ),
    )

    enabled_plugins: list[str] | None = Field(
        default=None,
        description="List of explicitly enabled plugins (None = all enabled). Takes precedence over disabled_plugins.",
    )

    disabled_plugins: list[str] | None = Field(
        default=None,
        description="List of explicitly disabled plugins.",
    )

    # CLI context for plugin access (set dynamically)
    cli_context: dict[str, Any] = Field(default_factory=dict, exclude=True)

    plugins: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Plugin-specific configurations keyed by plugin name",
    )

    @property
    def server_url(self) -> str:
        """Get the complete server URL."""
        return f"http://{self.server.host}:{self.server.port}"

    def model_dump_safe(self) -> dict[str, Any]:
        """
        Dump model data with sensitive information masked.

        Returns:
            dict: Configuration with sensitive data masked
        """
        return self.model_dump(mode="json")

    @classmethod
    def _validate_deprecated_keys(cls, config_data: dict[str, Any]) -> None:
        """Fail fast if deprecated legacy config keys are present."""
        deprecated_hits: list[tuple[str, str]] = []

        scheduler_cfg = config_data.get("scheduler") or {}
        if isinstance(scheduler_cfg, dict):
            key_map = {
                "pushgateway_enabled": "plugins.metrics.pushgateway_enabled",
                "pushgateway_url": "plugins.metrics.pushgateway_url",
                "pushgateway_job": "plugins.metrics.pushgateway_job",
                "pushgateway_interval_seconds": "plugins.metrics.pushgateway_push_interval",
            }
            for old_key, new_key in key_map.items():
                if old_key in scheduler_cfg:
                    deprecated_hits.append((f"scheduler.{old_key}", new_key))

        if "observability" in config_data:
            deprecated_hits.append(
                ("observability.*", "plugins.* (metrics/analytics/dashboard)")
            )

        for env_key in os.environ:
            upper = env_key.upper()
            if upper.startswith("SCHEDULER__PUSHGATEWAY_"):
                env_map = {
                    "SCHEDULER__PUSHGATEWAY_ENABLED": "plugins.metrics.pushgateway_enabled",
                    "SCHEDULER__PUSHGATEWAY_URL": "plugins.metrics.pushgateway_url",
                    "SCHEDULER__PUSHGATEWAY_JOB": "plugins.metrics.pushgateway_job",
                    "SCHEDULER__PUSHGATEWAY_INTERVAL_SECONDS": "plugins.metrics.pushgateway_push_interval",
                }
                target = env_map.get(upper, "plugins.metrics.*")
                deprecated_hits.append((env_key, target))
            if upper.startswith("OBSERVABILITY__"):
                deprecated_hits.append(
                    (env_key, "plugins.* (metrics/analytics/dashboard)")
                )

        if deprecated_hits:
            lines = [
                "Removed configuration keys detected. The following are no longer supported:",
            ]
            for old, new in deprecated_hits:
                lines.append(f"- {old} → {new}")
            lines.append(
                "Configure corresponding plugin settings under [plugins.*]. "
                "See: ccproxy/plugins/metrics/README.md and the Plugin Config Quickstart."
            )
            raise ValueError("\n".join(lines))

    @classmethod
    def load_toml_config(cls, toml_path: Path) -> dict[str, Any]:
        """Load configuration from a TOML file."""
        try:
            with toml_path.open("rb") as f:
                return tomllib.load(f)
        except OSError as e:
            raise ValueError(f"Cannot read TOML config file {toml_path}: {e}") from e
        except tomllib.TOMLDecodeError as e:
            raise ValueError(f"Invalid TOML syntax in {toml_path}: {e}") from e

    @classmethod
    def load_config_file(cls, config_path: Path) -> dict[str, Any]:
        """Load configuration from a file based on its extension."""
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
        """Create Settings instance from TOML configuration."""
        return cls.from_config(config_path=toml_path, **kwargs)

    @classmethod
    def from_config(
        cls,
        config_path: Path | str | None = None,
        cli_context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> "Settings":
        """Create Settings instance from configuration file."""
        if config_path is None:
            config_path_env = os.environ.get("CONFIG_FILE")
            if config_path_env:
                config_path = Path(config_path_env)

        if isinstance(config_path, str):
            config_path = Path(config_path)

        if config_path is None:
            config_path = find_toml_config_file()

        config_data: dict[str, Any] = {}
        if config_path and config_path.exists():
            config_data = cls.load_config_file(config_path)
            logger = get_logger(__name__)

            logger.info(
                "config_file_loaded",
                path=str(config_path),
                category="config",
            )

        cls._validate_deprecated_keys(config_data)

        settings = cls()

        for key, value in config_data.items():
            if hasattr(settings, key):
                if key in ["logging", "server", "security"] and isinstance(value, dict):
                    nested_obj = getattr(settings, key)
                    for nested_key, nested_value in value.items():
                        env_key = f"{key.upper()}__{nested_key.upper()}"
                        if os.getenv(env_key) is None:
                            setattr(nested_obj, nested_key, nested_value)
                elif key == "plugins" and isinstance(value, dict):
                    current_plugins = getattr(settings, key, {})

                    for plugin_name, plugin_config in value.items():
                        if isinstance(plugin_config, dict):
                            env_prefix = f"PLUGINS__{plugin_name.upper()}__"
                            has_env_override = any(
                                k.startswith(env_prefix) for k in os.environ
                            )

                            if has_env_override:
                                if plugin_name in current_plugins:
                                    merged_plugin_config = dict(plugin_config)
                                    merged_plugin_config.update(
                                        current_plugins[plugin_name]
                                    )
                                    current_plugins[plugin_name] = merged_plugin_config
                                else:
                                    pass
                            else:
                                current_plugins[plugin_name] = plugin_config
                        else:
                            current_plugins[plugin_name] = plugin_config

                    setattr(settings, key, current_plugins)
                else:
                    env_key = key.upper()
                    if os.getenv(env_key) is None:
                        setattr(settings, key, value)

        def _apply_overrides(target: Any, overrides: dict[str, Any]) -> None:
            for k, v in overrides.items():
                if (
                    isinstance(v, dict)
                    and hasattr(target, k)
                    and isinstance(getattr(target, k), BaseModel | dict)
                ):
                    sub = getattr(target, k)
                    if isinstance(sub, BaseModel):
                        _apply_overrides(sub, v)
                    elif isinstance(sub, dict):
                        sub.update(v)
                else:
                    setattr(target, k, v)

        if kwargs:
            _apply_overrides(settings, kwargs)

        if cli_context:
            # Store raw CLI context for plugins
            settings.cli_context = cli_context

            # Apply common serve CLI overrides directly to settings
            # Only override when a value is explicitly provided (not None / empty)
            server_overrides: dict[str, Any] = {}
            if cli_context.get("host") is not None:
                server_overrides["host"] = cli_context["host"]
            if cli_context.get("port") is not None:
                server_overrides["port"] = cli_context["port"]
            if cli_context.get("reload") is not None:
                server_overrides["reload"] = cli_context["reload"]

            logging_overrides: dict[str, Any] = {}
            if cli_context.get("log_level") is not None:
                logging_overrides["level"] = cli_context["log_level"]
            if cli_context.get("log_file") is not None:
                logging_overrides["file"] = cli_context["log_file"]

            security_overrides: dict[str, Any] = {}
            if cli_context.get("auth_token") is not None:
                security_overrides["auth_token"] = cli_context["auth_token"]

            if server_overrides:
                _apply_overrides(settings, {"server": server_overrides})
            if logging_overrides:
                _apply_overrides(settings, {"logging": logging_overrides})
            if security_overrides:
                _apply_overrides(settings, {"security": security_overrides})

            # Apply plugin enable/disable lists if provided
            enabled_plugins = cli_context.get("enabled_plugins")
            disabled_plugins = cli_context.get("disabled_plugins")
            if enabled_plugins is not None:
                settings.enabled_plugins = list(enabled_plugins)
            if disabled_plugins is not None:
                settings.disabled_plugins = list(disabled_plugins)

        return settings

    def get_cli_context(self) -> dict[str, Any]:
        """Get CLI context for plugin access."""
        return self.cli_context

    class LLMSettings(BaseModel):
        """LLM-specific feature toggles and defaults."""

        openai_thinking_xml: bool = Field(
            default=True, description="Serialize thinking as XML in OpenAI streams"
        )

    llm: LLMSettings = Field(
        default_factory=LLMSettings,
        description="Large Language Model (LLM) settings",
    )
