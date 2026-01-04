"""Configuration utilities - constants, validators, discovery, and scheduler."""

import re
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ccproxy.core.system import get_xdg_cache_home, get_xdg_config_home


# === Configuration Constants ===

# Plugin System Constants
PLUGIN_HEALTH_CHECK_TIMEOUT = 10.0  # seconds
PLUGIN_SUMMARY_CACHE_TTL = 300.0  # 5 minutes
PLUGIN_SUMMARY_CACHE_SIZE = 32  # entries

# Task Scheduler Constants
DEFAULT_TASK_INTERVAL = 3600  # 1 hour in seconds

# URL Constants
CLAUDE_API_BASE_URL = "https://api.anthropic.com"
CODEX_API_BASE_URL = "https://chatgpt.com/backend-api"

# API Endpoints
CLAUDE_MESSAGES_ENDPOINT = "/v1/messages"
CODEX_RESPONSES_ENDPOINT = "/codex/responses"

# Format Conversion Patterns
OPENAI_CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
OPENAI_COMPLETIONS_PATH = "/chat/completions"
ANTHROPIC_MESSAGES_PATH = "/v1/messages"

# HTTP Client Configuration
HTTP_CLIENT_TIMEOUT = 120.0  # 2 minutes default timeout
HTTP_STREAMING_TIMEOUT = 300.0  # 5 minutes for streaming requests
HTTP_CLIENT_POOL_SIZE = 20  # Max connections per pool


# === Configuration Validators ===


class ConfigValidationError(Exception):
    """Configuration validation error."""

    pass


def validate_host(host: str) -> str:
    """Validate host address.

    Args:
        host: Host address to validate

    Returns:
        The validated host address

    Raises:
        ConfigValidationError: If host is invalid
    """
    if not host:
        raise ConfigValidationError("Host cannot be empty")

    # Allow localhost, IP addresses, and domain names
    if host in ["localhost", "0.0.0.0", "127.0.0.1"]:
        return host

    # Basic IP address validation
    if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", host):
        parts = host.split(".")
        if all(0 <= int(part) <= 255 for part in parts):
            return host
        raise ConfigValidationError(f"Invalid IP address: {host}")

    # Basic domain name validation
    if re.match(r"^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", host):
        return host

    return host  # Allow other formats for flexibility


def validate_port(port: int | str) -> int:
    """Validate port number.

    Args:
        port: Port number to validate

    Returns:
        The validated port number

    Raises:
        ConfigValidationError: If port is invalid
    """
    if isinstance(port, str):
        try:
            port = int(port)
        except ValueError as e:
            raise ConfigValidationError(f"Port must be a valid integer: {port}") from e

    if not isinstance(port, int):
        raise ConfigValidationError(f"Port must be an integer: {port}")

    if port < 1 or port > 65535:
        raise ConfigValidationError(f"Port must be between 1 and 65535: {port}")

    return port


def validate_url(url: str) -> str:
    """Validate URL format.

    Args:
        url: URL to validate

    Returns:
        The validated URL

    Raises:
        ConfigValidationError: If URL is invalid
    """
    if not url:
        raise ConfigValidationError("URL cannot be empty")

    try:
        result = urlparse(url)
        if not result.scheme or not result.netloc:
            raise ConfigValidationError(f"Invalid URL format: {url}")
    except Exception as e:
        raise ConfigValidationError(f"Invalid URL: {url}") from e

    return url


def validate_path(path: str | Path) -> Path:
    """Validate file path.

    Args:
        path: Path to validate

    Returns:
        The validated Path object

    Raises:
        ConfigValidationError: If path is invalid
    """
    if isinstance(path, str):
        path = Path(path)

    if not isinstance(path, Path):
        raise ConfigValidationError(f"Path must be a string or Path object: {path}")

    return path


def validate_log_level(level: str) -> str:
    """Validate log level.

    Args:
        level: Log level to validate

    Returns:
        The validated log level

    Raises:
        ConfigValidationError: If log level is invalid
    """
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    level = level.upper()

    if level not in valid_levels:
        raise ConfigValidationError(
            f"Invalid log level: {level}. Must be one of: {valid_levels}"
        )

    return level


def validate_cors_origins(origins: list[str]) -> list[str]:
    """Validate CORS origins.

    Args:
        origins: List of origin URLs to validate

    Returns:
        The validated list of origins

    Raises:
        ConfigValidationError: If any origin is invalid
    """
    if not isinstance(origins, list):
        raise ConfigValidationError("CORS origins must be a list")

    validated_origins = []
    for origin in origins:
        if origin == "*":
            validated_origins.append(origin)
        else:
            validated_origins.append(validate_url(origin))

    return validated_origins


def validate_timeout(timeout: int | float) -> int | float:
    """Validate timeout value.

    Args:
        timeout: Timeout value to validate

    Returns:
        The validated timeout value

    Raises:
        ConfigValidationError: If timeout is invalid
    """
    if not isinstance(timeout, int | float):
        raise ConfigValidationError(f"Timeout must be a number: {timeout}")

    if timeout <= 0:
        raise ConfigValidationError(f"Timeout must be positive: {timeout}")

    return timeout


def validate_config_dict(config: dict[str, Any]) -> dict[str, Any]:
    """Validate configuration dictionary.

    Args:
        config: Configuration dictionary to validate

    Returns:
        The validated configuration dictionary

    Raises:
        ConfigValidationError: If configuration is invalid
    """
    if not isinstance(config, dict):
        raise ConfigValidationError("Configuration must be a dictionary")

    validated_config: dict[str, Any] = {}

    # Validate specific fields if present
    if "host" in config:
        validated_config["host"] = validate_host(config["host"])

    if "port" in config:
        validated_config["port"] = validate_port(config["port"])

    if "target_url" in config:
        validated_config["target_url"] = validate_url(config["target_url"])

    if "log_level" in config:
        validated_config["log_level"] = validate_log_level(config["log_level"])

    if "cors_origins" in config:
        validated_config["cors_origins"] = validate_cors_origins(config["cors_origins"])

    if "timeout" in config:
        validated_config["timeout"] = validate_timeout(config["timeout"])

    # Copy other fields without validation
    for key, value in config.items():
        if key not in validated_config:
            validated_config[key] = value

    return validated_config


# === Configuration Discovery ===


def find_toml_config_file() -> Path | None:
    """Find the TOML configuration file for ccproxy.

    Searches in the following order:
    1. .ccproxy.toml in current directory
    2. ccproxy.toml in git repository root (if in a git repo)
    3. config.toml in XDG_CONFIG_HOME/ccproxy/
    """
    # Check current directory first
    candidates = [
        Path(".ccproxy.toml").resolve(),
        Path("ccproxy.toml").resolve(),
    ]

    # Check git repo root
    git_root = find_git_root()
    if git_root:
        candidates.extend(
            [
                git_root / ".ccproxy.toml",
                git_root / "ccproxy.toml",
            ]
        )

    # Check XDG config directory
    config_dir = get_ccproxy_config_dir()
    candidates.append(config_dir / "config.toml")

    # Return first existing file
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    return None


def find_git_root(path: Path | None = None) -> Path | None:
    """Find the root directory of a git repository."""
    if path is None:
        path = Path.cwd()

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=path,
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_ccproxy_config_dir() -> Path:
    """Get the ccproxy configuration directory.

    Returns:
        Path to the ccproxy configuration directory within XDG_CONFIG_HOME.
    """
    return get_xdg_config_home() / "ccproxy"


def get_claude_cli_config_dir() -> Path:
    """Get the Claude CLI configuration directory.

    Returns:
        Path to the Claude CLI configuration directory within XDG_CONFIG_HOME.
    """
    return get_xdg_config_home() / "claude"


def get_claude_docker_home_dir() -> Path:
    """Get the Claude Docker home directory.

    Returns:
        Path to the Claude Docker home directory within XDG_DATA_HOME.
    """
    return get_ccproxy_config_dir() / "home"


def get_ccproxy_cache_dir() -> Path:
    """Get the ccproxy cache directory.

    Returns:
        Path to the ccproxy cache directory within XDG_CACHE_HOME.
    """
    return get_xdg_cache_home() / "ccproxy"


# === Scheduler Configuration ===


class SchedulerSettings(BaseSettings):
    """
    Configuration settings for the unified scheduler system.

    Controls global scheduler behavior and individual task configurations.
    Settings can be configured via environment variables with SCHEDULER__ prefix.
    """

    # Global scheduler settings
    enabled: bool = Field(
        default=True,
        description="Whether the scheduler system is enabled",
    )

    max_concurrent_tasks: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of tasks that can run concurrently",
    )

    graceful_shutdown_timeout: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Timeout in seconds for graceful task shutdown",
    )

    # Pricing updater task settings
    pricing_update_enabled: bool = Field(
        default=True,
        description="Whether pricing cache update task is enabled. Enabled by default for privacy - downloads from GitHub when enabled",
    )

    pricing_update_interval_hours: int = Field(
        default=24,
        ge=1,
        le=168,  # Max 1 week
        description="Interval in hours between pricing cache updates",
    )

    pricing_force_refresh_on_startup: bool = Field(
        default=False,
        description="Whether to force pricing refresh immediately on startup",
    )

    # Pushgateway settings are handled by the metrics plugin
    # The metrics plugin now manages its own pushgateway configuration

    stats_printing_enabled: bool = Field(
        default=False,
        description="Whether stats printing task is enabled",
    )

    stats_printing_interval_seconds: float = Field(
        default=300.0,
        ge=1.0,
        le=3600.0,  # Max 1 hour
        description="Interval in seconds between stats printing",
    )

    # Version checking task settings
    version_check_enabled: bool = Field(
        default=True,
        description="Whether version update checking is enabled. Enabled by default for privacy - checks GitHub API when enabled",
    )

    version_check_interval_hours: int = Field(
        default=6,
        ge=1,
        le=168,  # Max 1 week
        description="Interval in hours between version checks",
    )

    version_check_cache_ttl_hours: float = Field(
        default=6,
        ge=0.1,
        le=24.0,
        description="Maximum age in hours since last check version check",
    )

    model_config = SettingsConfigDict(
        env_prefix="SCHEDULER__",
        case_sensitive=False,
    )
