"""Core configuration settings - server, HTTP, CORS, logging, and plugins."""

from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, Field, field_validator

from ccproxy.core.system import get_xdg_config_home


class ServerSettings(BaseModel):
    """Server-specific configuration settings."""

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

    bypass_mode: bool = Field(
        default=False,
        description="Enable bypass mode for testing (uses mock responses instead of real API calls)",
    )


class HTTPSettings(BaseModel):
    """HTTP client configuration settings.

    Controls how the core HTTP client handles compression and other HTTP-level settings.
    """

    compression_enabled: bool = Field(
        default=True,
        description="Enable compression for provider requests (Accept-Encoding header)",
    )

    accept_encoding: str = Field(
        default="gzip, deflate",
        description="Accept-Encoding header value when compression is enabled",
    )


class CORSSettings(BaseModel):
    """CORS-specific configuration settings."""

    origins: list[str] = Field(
        default_factory=lambda: [
            "vscode-file://vscode-app",
            "http://localhost/*",
            "http://localhost:*/*",
            "http://127.0.0.1:*/*",
            "http://127.0.0.1:/*",
        ],
        description="CORS allowed origins (avoid using '*' for security)",
    )

    credentials: bool = Field(
        default=True,
        description="CORS allow credentials",
    )

    methods: list[str] = Field(
        default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        description="CORS allowed methods",
    )

    headers: list[str] = Field(
        default_factory=lambda: [
            "Content-Type",
            "Authorization",
            "Accept",
            "Origin",
            "X-Requested-With",
        ],
        description="CORS allowed headers",
    )

    origin_regex: str | None = Field(
        default=None,
        description="CORS origin regex pattern",
    )

    expose_headers: list[str] = Field(
        default_factory=list,
        description="CORS exposed headers",
    )

    max_age: int = Field(
        default=600,
        description="CORS preflight max age in seconds",
        ge=0,
    )


# === Logging Configuration ===


LogLevelName = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "TRACE"]
LOG_LEVEL_OPTIONS: tuple[str, ...] = (
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
    "TRACE",
)

LogFormatName = Literal["auto", "rich", "json", "plain"]
LOG_FORMAT_OPTIONS: tuple[str, ...] = ("auto", "rich", "json", "plain")

LOG_FORMAT_DESCRIPTION = "Logging format: 'rich', 'json', 'plain', or 'auto' (auto-selects based on environment)"


class LoggingSettings(BaseModel):
    """Centralized logging configuration - core app only."""

    level: LogLevelName = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL, TRACE)",
    )

    format: LogFormatName = Field(
        default="auto",
        description=LOG_FORMAT_DESCRIPTION,
    )

    file: str | None = Field(
        default=None,
        description="Path to JSON log file. If specified, logs will be written to this file in JSON format",
    )

    verbose_api: bool = Field(
        default=False,
        description="Enable verbose API request/response logging",
    )

    request_log_dir: str | None = Field(
        default=None,
        description="Directory to save individual request/response logs when verbose_api is enabled",
    )

    plugin_log_base_dir: str = Field(
        default="/tmp/ccproxy",
        description="Shared base directory for all plugin log outputs",
    )

    @field_validator("level", mode="before")
    @classmethod
    def validate_log_level(cls, value: LogLevelName | str) -> LogLevelName:
        """Validate and normalize log level."""
        if isinstance(value, str):
            candidate = value.upper()
        else:
            candidate = value

        if candidate not in LOG_LEVEL_OPTIONS:
            raise ValueError(
                f"Invalid log level: {value}. Must be one of {list(LOG_LEVEL_OPTIONS)}"
            )

        return cast(LogLevelName, candidate)

    @field_validator("format", mode="before")
    @classmethod
    def validate_log_format(cls, value: LogFormatName | str) -> LogFormatName:
        """Validate and normalize log format."""
        if isinstance(value, str):
            candidate = value.lower()
        else:
            candidate = value

        if candidate not in LOG_FORMAT_OPTIONS:
            raise ValueError(
                f"Invalid log format: {value}. Must be one of {list(LOG_FORMAT_OPTIONS)}"
            )

        return cast(LogFormatName, candidate)


def _default_plugin_directories() -> list[Path]:
    """Default directories scanned for filesystem plugins."""

    # package_plugins = Path(__file__).resolve().parent.parent.parent / "plugins"
    # we don't include packages from ccproxy.plugins because they
    # are using importlib.metadata entry points
    user_plugins = get_xdg_config_home() / "ccproxy" / "plugins"

    seen: set[Path] = set()
    ordered: list[Path] = []
    for candidate in [user_plugins]:
        normalized = candidate.resolve()
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(candidate)
    return ordered


class PluginDiscoverySettings(BaseModel):
    """Configuration for filesystem plugin discovery."""

    directories: list[Path] = Field(
        default_factory=_default_plugin_directories,
        description=(
            "Ordered directories scanned for local plugins."
            " Defaults to the bundled ccproxy/plugins directory and"
            " ${XDG_CONFIG_HOME}/ccproxy/plugins."
        ),
    )
