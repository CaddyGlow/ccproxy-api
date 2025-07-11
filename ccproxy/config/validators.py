"""Configuration validation utilities."""

import re
from pathlib import Path
from typing import Any, Optional, Union
from urllib.parse import urlparse


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
