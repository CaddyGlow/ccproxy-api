"""Shared server utilities for CLI commands."""

from pathlib import Path
from typing import Optional

from ccproxy.config.settings import (
    ConfigurationError,
    Settings,
    config_manager,
)
from ccproxy.core.logging import get_structlog_logger


logger = get_structlog_logger(__name__)


def validate_server_settings(settings: Settings) -> None:
    """Validate server settings before starting.

    Args:
        settings: The settings to validate

    Raises:
        ConfigurationError: If settings are invalid
    """
    logger.debug(
        "validating_server_settings",
        host=settings.server.host,
        port=settings.server.port,
        log_level=settings.server.log_level,
        workers=settings.server.workers,
    )

    # Validate port range
    if not 1 <= settings.server.port <= 65535:
        raise ConfigurationError(
            f"Port must be between 1 and 65535, got {settings.server.port}"
        )

    # Validate host
    if not settings.server.host:
        raise ConfigurationError("Host cannot be empty")

    # Validate log level
    valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if settings.server.log_level.upper() not in valid_log_levels:
        raise ConfigurationError(
            f"Invalid log level: {settings.server.log_level}. "
            f"Must be one of: {', '.join(valid_log_levels)}"
        )

    # Validate workers
    if settings.server.workers and settings.server.workers < 1:
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

    logger.debug("checking_port_availability", host=host, port=port)

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((host, port))
            logger.debug("port_available", host=host, port=port)
            return True
    except OSError:
        logger.debug("port_unavailable", host=host, port=port)
        return False
