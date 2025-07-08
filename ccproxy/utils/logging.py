"""Rich logging configuration for Claude Code Proxy."""

import logging
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme
from uvicorn.logging import DefaultFormatter

from ccproxy.utils.cli import get_rich_toolkit


# Custom theme for the logger
CUSTOM_THEME = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "critical": "bold white on red",
        "debug": "dim white",
        "timestamp": "dim cyan",
        "path": "dim blue",
    }
)

# Create console with custom theme
console = Console(theme=CUSTOM_THEME)


def setup_rich_logging(
    level: str = "INFO",
    show_path: bool = False,
    show_time: bool = True,
    console_width: int | None = None,
    configure_uvicorn: bool = True,
) -> None:
    """Configure rich logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        show_path: Whether to show the module path in logs
        show_time: Whether to show timestamps in logs
        console_width: Optional console width override
        configure_uvicorn: Whether to configure uvicorn loggers
    """
    # Create rich handler with custom settings
    rich_handler = RichHandler(
        console=Console(theme=CUSTOM_THEME, width=console_width),
        show_time=show_time,
        show_path=show_path,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
        tracebacks_suppress=[],
        markup=True,
        enable_link_path=True,
    )

    # Configure the handler format
    rich_handler.setFormatter(
        logging.Formatter(
            "%(message)s",
            # datefmt="[%Y-%m-%d %H:%M:%S]",
            datefmt="[%H:%M:%S]",
        )
    )

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        handlers=[rich_handler],
        force=True,
    )

    # Configure specific loggers
    if configure_uvicorn:
        # Configure uvicorn loggers to use our rich handler
        uvicorn_loggers = {
            "uvicorn": logging.INFO,
            "uvicorn.error": logging.INFO,
            "uvicorn.access": logging.INFO,  # Always show access logs
        }

        for logger_name, log_level in uvicorn_loggers.items():
            uvicorn_logger = logging.getLogger(logger_name)
            uvicorn_logger.handlers = []
            uvicorn_logger.addHandler(rich_handler)
            uvicorn_logger.setLevel(log_level)
            uvicorn_logger.propagate = False

        # Disable propagation for specific noisy loggers
        for logger_name in ["httpx", "httpcore", "keyring"]:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


class CustomFormatter(DefaultFormatter):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.toolkit = get_rich_toolkit()

    def formatMessage(self, record: logging.LogRecord) -> str:  # noqa: N802
        return self.toolkit.print_as_string(record.getMessage(), tag=record.levelname)


def get_uvicorn_log_config() -> dict[str, Any]:
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": CustomFormatter,
                "fmt": "%(levelprefix)s %(message)s",
                "use_colors": None,
            },
            "access": {
                "()": CustomFormatter,
                "fmt": "%(levelprefix)s %(client_addr)s - '%(request_line)s' %(status_code)s",
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": "INFO"},
            "uvicorn.error": {"level": "INFO"},
            "uvicorn.access": {
                "handlers": ["access"],
                "level": "INFO",
                "propagate": False,
            },
        },
    }
