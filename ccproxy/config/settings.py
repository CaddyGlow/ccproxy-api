import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ccproxy.core.logging import get_logger

from .core import (
    CORSSettings,
    HTTPSettings,
    LoggingSettings,
    PluginDiscoverySettings,
    ServerSettings,
)
from .runtime import BinarySettings
from .security import AuthSettings, SecuritySettings
from .utils import SchedulerSettings, find_toml_config_file, get_ccproxy_config_dir


_CONFIG_MISSING_LOGGED = False

# Default plugins enabled when no config file exists
DEFAULT_ENABLED_PLUGINS = [
    "codex",
    "copilot",
    "claude_api",
    "claude_sdk",
    "oauth_codex",
    "oauth_claude",
]


def _auth_default() -> AuthSettings:
    return AuthSettings(credentials_ttl_seconds=3600.0)


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
        json_schema_extra={"config_example_hidden": True},
    )

    auth: AuthSettings = Field(
        default_factory=_auth_default,
        description="Authentication manager settings (e.g., credentials caching)",
    )

    binary: BinarySettings = Field(
        default_factory=BinarySettings,
        description="Binary resolution and package manager fallback configuration",
        json_schema_extra={"config_example_hidden": True},
    )

    scheduler: SchedulerSettings = Field(
        default_factory=SchedulerSettings,
        description="Task scheduler configuration settings",
        json_schema_extra={"config_example_hidden": True},
    )

    plugin_discovery: PluginDiscoverySettings = Field(
        default_factory=PluginDiscoverySettings,
        description="Filesystem plugin discovery search paths",
        json_schema_extra={"config_example_hidden": True},
    )

    enable_plugins: bool = Field(
        default=True,
        description="Enable plugin system",
        json_schema_extra={"config_example_hidden": True},
    )

    plugins_disable_local_discovery: bool = Field(
        default=False,
        description=(
            "If true, skip filesystem plugin discovery from the local 'plugins/' directory "
            "and load plugins only from installed entry points."
        ),
        json_schema_extra={"config_example_hidden": True},
    )

    enabled_plugins: list[str] | None = Field(
        default=None,
        description="List of explicitly enabled plugins (None = all enabled). Takes precedence over disabled_plugins.",
        json_schema_extra={"config_example_hidden": False},
    )

    disabled_plugins: list[str] | None = Field(
        default=None,
        description="List of explicitly disabled plugins.",
        json_schema_extra={"config_example_hidden": True},
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

    # ------------------------------
    # Internal helpers (merging/overrides)
    # ------------------------------
    @staticmethod
    def _env_has_prefix(prefix: str) -> bool:
        p = prefix.upper()
        return any(k.upper().startswith(p) for k in os.environ)

    @staticmethod
    def _merge_model(
        model: BaseModel, overrides: dict[str, Any], env_prefix: str
    ) -> BaseModel:
        """
        Deep-merge a dict of overrides into a BaseModel while preserving env-var precedence.
        env_prefix should end with '__' when called for nested fields.
        """
        update_payload: dict[str, Any] = {}

        for field_name, override_value in overrides.items():
            field_env_key = f"{env_prefix}{field_name.upper()}"
            # If an env var exists for this field, do NOT override from file.
            if os.getenv(field_env_key) is not None:
                continue

            current_value = getattr(model, field_name, None)

            if isinstance(current_value, BaseModel) and isinstance(
                override_value, dict
            ):
                nested_prefix = f"{field_env_key}__"
                merged_nested = Settings._merge_model(
                    current_value, override_value, nested_prefix
                )
                update_payload[field_name] = merged_nested
            elif isinstance(current_value, dict) and isinstance(override_value, dict):
                # Deep-merge dict but skip keys that have env overrides
                merged_dict = current_value.copy()
                for nk, nv in override_value.items():
                    nested_env_key = f"{field_env_key}__{nk.upper()}"
                    if os.getenv(nested_env_key) is None:
                        if isinstance(merged_dict.get(nk), dict) and isinstance(
                            nv, dict
                        ):
                            # deep merge nested dicts with respect to env
                            merged_dict[nk] = Settings._merge_dict(
                                merged_dict.get(nk, {}), nv, f"{nested_env_key}__"
                            )
                        else:
                            merged_dict[nk] = nv
                update_payload[field_name] = merged_dict
            else:
                update_payload[field_name] = override_value

        if not update_payload:
            return model
        return model.model_copy(update=update_payload)

    @staticmethod
    def _merge_dict(
        base: dict[str, Any], overrides: dict[str, Any], env_prefix: str
    ) -> dict[str, Any]:
        """
        Deep-merge dicts while respecting env-var precedence using the given env_prefix (no trailing __ required).
        """
        out = dict(base)
        for k, v in overrides.items():
            key_env = f"{env_prefix}{k.upper()}"
            if os.getenv(key_env) is not None:
                continue
            if isinstance(out.get(k), dict) and isinstance(v, dict):
                out[k] = Settings._merge_dict(out[k], v, f"{key_env}__")
            else:
                out[k] = v
        return out

    @staticmethod
    def _merge_plugins(
        current_plugins: dict[str, Any], overrides: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Merge plugin configuration trees with env precedence at both plugin and nested key levels.
        """
        merged = dict(current_plugins)
        for plugin_name, plugin_cfg in overrides.items():
            env_prefix = f"PLUGINS__{plugin_name.upper()}__"

            # If any env for this plugin exists, we keep current_plugins[plugin_name] as-is,
            # but we still allow env-free nested keys to merge if we already have a dict.
            if isinstance(plugin_cfg, dict):
                if Settings._env_has_prefix(env_prefix):
                    # Partial merge respecting env at nested levels if the plugin already exists as a dict.
                    if isinstance(merged.get(plugin_name), dict):
                        merged[plugin_name] = Settings._merge_dict(
                            merged[plugin_name],
                            plugin_cfg,
                            env_prefix,
                        )
                    else:
                        # Keep existing unless it's missing entirely.
                        merged.setdefault(plugin_name, merged.get(plugin_name, {}))
                else:
                    existing = merged.get(plugin_name, {})
                    if isinstance(existing, dict):
                        merged[plugin_name] = Settings._merge_dict(
                            existing, plugin_cfg, env_prefix
                        )
                    else:
                        merged[plugin_name] = plugin_cfg
            else:
                # Non-dict plugin setting: only apply if no top-level env overrides present.
                if not Settings._env_has_prefix(env_prefix):
                    merged[plugin_name] = plugin_cfg
        return merged

    @staticmethod
    def _apply_overrides(target: Any, overrides: dict[str, Any]) -> None:
        """
        Apply CLI/kwargs overrides after file and env processing.
        Dicts are shallow-merged; nested BaseModels recurse.
        """
        for k, v in overrides.items():
            if (
                isinstance(v, dict)
                and hasattr(target, k)
                and isinstance(getattr(target, k), (BaseModel | dict))
            ):
                sub = getattr(target, k)
                if isinstance(sub, BaseModel):
                    # Apply directly field-by-field
                    Settings._apply_overrides(sub, v)
                elif isinstance(sub, dict):
                    sub.update(v)
            else:
                setattr(target, k, v)

    # ------------------------------
    # Factory
    # ------------------------------
    @classmethod
    def from_config(
        cls,
        config_path: Path | str | None = None,
        cli_context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> "Settings":
        """Create Settings instance from configuration file with env precedence and safe merging."""
        logger = get_logger(__name__)

        global _CONFIG_MISSING_LOGGED

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
            logger.debug(
                "config_file_loaded",
                path=str(config_path),
                category="config",
            )
        elif not _CONFIG_MISSING_LOGGED:
            suggestion = f"ccproxy config init --output-dir {get_ccproxy_config_dir()}"
            log_kwargs: dict[str, Any] = {
                "category": "config",
                "suggested_command": suggestion,
            }
            if config_path is not None:
                log_kwargs["path"] = str(config_path)
            logger.warning("config_file_missing", **log_kwargs)
            _CONFIG_MISSING_LOGGED = True

        cls._validate_deprecated_keys(config_data)

        # Start from env + .env via BaseSettings
        settings = cls()

        # Merge file-based configuration with env-var precedence
        for key, value in config_data.items():
            if not hasattr(settings, key):
                continue

            if key == "plugins" and isinstance(value, dict):
                current_plugins = getattr(settings, key, {})
                merged_plugins = cls._merge_plugins(current_plugins, value)
                setattr(settings, key, merged_plugins)
                continue

            current_attr = getattr(settings, key)

            if isinstance(value, dict) and isinstance(current_attr, BaseModel):
                merged_model = cls._merge_model(current_attr, value, f"{key.upper()}__")
                setattr(settings, key, merged_model)
            else:
                # Only set top-level simple types if there is no top-level env override
                env_key = key.upper()
                if os.getenv(env_key) is None:
                    setattr(settings, key, value)

        # Smart default: if no config file exists and enabled_plugins is still None,
        # set a curated default list of core plugins
        if not config_path or not config_path.exists():
            if settings.enabled_plugins is None:
                settings.enabled_plugins = DEFAULT_ENABLED_PLUGINS

        # Apply direct kwargs overrides (highest precedence within process)
        if kwargs:
            cls._apply_overrides(settings, kwargs)

        # Apply CLI context (explicit flags)
        if cli_context:
            # Store raw CLI context for plugin access
            settings.cli_context = cli_context

            # Apply common serve CLI overrides directly to settings
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
                cls._apply_overrides(settings, {"server": server_overrides})
            if logging_overrides:
                cls._apply_overrides(settings, {"logging": logging_overrides})
            if security_overrides:
                cls._apply_overrides(settings, {"security": security_overrides})

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
