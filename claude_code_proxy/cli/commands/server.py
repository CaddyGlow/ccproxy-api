"""Shared server utilities for CLI commands."""

import logging
from pathlib import Path
from typing import Optional

from claude_code_proxy.config.settings import (
    ConfigurationError,
    Settings,
    config_manager,
)
from claude_code_proxy.utils.logging import get_logger


logger = get_logger(__name__)


def validate_server_settings(settings: Settings) -> None:
    """Validate server settings before starting.

    Args:
        settings: The settings to validate

    Raises:
        ConfigurationError: If settings are invalid
    """
    # Validate port range
    if not 1 <= settings.port <= 65535:
        raise ConfigurationError(
            f"Port must be between 1 and 65535, got {settings.port}"
        )

    # Validate host
    if not settings.host:
        raise ConfigurationError("Host cannot be empty")

    # Validate log level
    valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if settings.log_level.upper() not in valid_log_levels:
        raise ConfigurationError(
            f"Invalid log level: {settings.log_level}. "
            f"Must be one of: {', '.join(valid_log_levels)}"
        )

    # Validate workers
    if settings.workers and settings.workers < 1:
        raise ConfigurationError("Workers must be at least 1")


def get_server_startup_message(
    host: str,
    port: int,
    workers: int | None = None,
    reload: bool = False,
) -> str:
    """Generate a server startup message.

    Args:
        host: Server host
        port: Server port
        workers: Number of workers
        reload: Whether reload is enabled

    Returns:
        Formatted startup message
    """
    message = f"Server starting at http://{host}:{port}"

    details = []
    if workers and workers > 1:
        details.append(f"workers: {workers}")
    if reload:
        details.append("reload: enabled")

    if details:
        message += f" ({', '.join(details)})"

    return message


def check_port_availability(host: str, port: int) -> bool:
    """Check if a port is available for binding.

    Args:
        host: Host to check
        port: Port to check

    Returns:
        True if port is available, False otherwise
    """
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((host, port))
            return True
    except OSError:
        return False
