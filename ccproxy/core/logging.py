"""Core logging configuration for Claude Code Proxy."""

import logging
import sys
from collections.abc import Callable, Iterable, MutableMapping
from typing import Any

import httpx
import prometheus_client
import structlog
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme
from uvicorn.logging import DefaultFormatter


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

        # Configure specific loggers based on main log level
        main_level = getattr(logging, level.upper())

        # For httpx, enable info logging only when main level is DEBUG
        httpx_logger = logging.getLogger("httpx")
        if main_level <= logging.DEBUG:
            httpx_logger.setLevel(logging.INFO)
            httpx_logger.addHandler(rich_handler)
            httpx_logger.propagate = False
        else:
            httpx_logger.setLevel(logging.WARNING)

        # Disable httpx's internal trace events (those bare lines without timestamps)
        # These are separate from the logger and print directly to stdout
        import os

        os.environ.pop("HTTPX_LOG_LEVEL", None)  # Remove any HTTPX_LOG_LEVEL env var

        # Always suppress these noisy loggers
        for logger_name in [
            "httpcore",
            "urllib3",
            "urllib3.connectionpool",
            "keyring",
            "asyncio",
        ]:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.WARNING)
            # Completely disable httpcore and urllib3 to prevent debug traces that bypass our logging
            if logger_name in ["httpcore", "urllib3"]:
                logger.disabled = True


def configure_structlog(
    format_type: str = "rich",
    level: str = "INFO",
    enable_observability: bool = False,
    show_path: bool = False,
    show_time: bool = True,
) -> None:
    """Configure structlog with support for Rich output and observability.

    Args:
        format_type: Output format - "rich" for development, "json" for production
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_observability: Whether to enable observability pipeline integration
        show_path: Whether to show the module path in logs
        show_time: Whether to show timestamps in logs
    """
    import structlog
    from structlog.stdlib import LoggerFactory

    # Base processors that are always included
    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.TimeStamper(fmt="iso" if show_time else None),
        structlog.processors.add_log_level,
    ]

    # Add callsite info if path is requested
    if show_path:
        processors.append(
            structlog.processors.CallsiteParameterAdder(
                parameters=[
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                    structlog.processors.CallsiteParameter.LINENO,
                    structlog.processors.CallsiteParameter.FILENAME,
                ]
            )
        )

    # Add observability processor if enabled
    if enable_observability:
        try:
            from ccproxy.observability.pipeline import create_structlog_processor

            processors.append(create_structlog_processor())
        except ImportError:
            # Observability not available, continue without it
            pass

    # Add format-specific processors
    if format_type == "rich":
        # For Rich output, use Rich-aware renderer that processes markup
        processors.extend(
            [
                _rich_structlog_processor,
                _rich_console_renderer,
            ]
        )
    elif format_type == "json":
        # For JSON output, use standard JSON renderer
        processors.append(structlog.processors.JSONRenderer())
    else:
        # Default to plain text
        processors.append(structlog.dev.ConsoleRenderer(colors=False))

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=LoggerFactory(),
        context_class=dict,
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging to work with structlog
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, level.upper()),
        force=True,
    )


def _rich_structlog_processor(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Custom structlog processor that preserves Rich markup in messages."""
    # Extract the main message
    event = event_dict.get("event", "")

    # Add Rich styling based on log level
    level = event_dict.get("level", "info").lower()
    level_colors = {
        "debug": "dim white",
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "critical": "bold white on red",
    }

    color = level_colors.get(level, "white")

    # Apply Rich styling while preserving existing markup
    if event and not event.startswith("["):
        event_dict["event"] = f"[{color}]{event}[/{color}]"

    return event_dict


def _rich_console_renderer(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> str:
    """Custom renderer that uses Rich console to properly process markup."""
    # Import Rich console for markup processing
    from rich.console import Console

    # Create a Rich console that can process markup
    rich_console = Console(markup=True, highlight=False, force_terminal=True)

    # Extract timestamp
    timestamp = event_dict.get("timestamp", "")
    if timestamp:
        timestamp_str = f"[dim]{timestamp}[/dim] "
    else:
        timestamp_str = ""

    # Extract log level
    level = event_dict.get("level", "info").upper()
    level_colors = {
        "DEBUG": "dim white",
        "INFO": "cyan",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold white on red",
    }
    level_color = level_colors.get(level, "white")
    level_str = f"[{level_color}][{level:<8}][/{level_color}] "

    # Extract main event message
    event = event_dict.get("event", "")

    # Build context string from remaining fields
    context_parts = []
    for key, value in event_dict.items():
        if key not in ("timestamp", "level", "event"):
            context_parts.append(f"[dim]{key}[/dim]=[bright_blue]{value}[/bright_blue]")

    context_str = " ".join(context_parts)
    if context_str:
        context_str = " " + context_str

    # Combine all parts
    message = f"{timestamp_str}{level_str}{event}{context_str}"

    # Use Rich console to render the markup to a string
    with rich_console.capture() as capture:
        rich_console.print(message, markup=True, highlight=False)

    return capture.get().rstrip("\n")


def get_structlog_logger(name: str) -> Any:
    """Get a structlog logger instance with the given name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured structlog logger instance
    """
    import structlog

    return structlog.get_logger(name)


def setup_dual_logging(
    level: str = "INFO",
    format_type: str = "rich",
    enable_observability: bool = False,
    show_path: bool = False,
    show_time: bool = True,
    console_width: int | None = None,
    configure_uvicorn: bool = True,
) -> None:
    """Setup both Rich and structlog logging for dual-mode support.

    This function configures both Rich logging (for beautiful console output)
    and structlog (for structured logging), allowing code to use either approach.
    The format_type parameter controls the overall output style.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Output format - "rich" for development, "json" for production
        enable_observability: Whether to enable observability pipeline integration
        show_path: Whether to show the module path in logs
        show_time: Whether to show timestamps in logs
        console_width: Optional console width override
        configure_uvicorn: Whether to configure uvicorn loggers
    """
    if format_type == "rich":
        # Setup Rich logging for beautiful console output
        setup_rich_logging(
            level=level,
            show_path=show_path,
            show_time=show_time,
            console_width=console_width,
            configure_uvicorn=configure_uvicorn,
        )

        # Setup structlog with Rich output compatibility
        configure_structlog(
            format_type="rich",
            level=level,
            enable_observability=enable_observability,
            show_path=show_path,
            show_time=show_time,
        )
    else:
        # For JSON or other formats, use structlog-only configuration
        configure_structlog(
            format_type=format_type,
            level=level,
            enable_observability=enable_observability,
            show_path=show_path,
            show_time=show_time,
        )

        # Configure minimal standard logging for JSON output
        if configure_uvicorn:
            # Minimal uvicorn configuration for production
            uvicorn_loggers = {
                "uvicorn": logging.INFO,
                "uvicorn.error": logging.INFO,
                "uvicorn.access": logging.INFO,
            }

            json_formatter = logging.Formatter("%(message)s")
            for logger_name, log_level in uvicorn_loggers.items():
                uvicorn_logger = logging.getLogger(logger_name)
                uvicorn_logger.handlers = []

                # Add simple handler for JSON output
                handler = logging.StreamHandler()
                handler.setFormatter(json_formatter)
                uvicorn_logger.addHandler(handler)
                uvicorn_logger.setLevel(log_level)
                uvicorn_logger.propagate = False

        # Configure verbose loggers based on main log level
        main_level = getattr(logging, level.upper())

        # For httpx, enable info logging only when main level is DEBUG
        httpx_logger = logging.getLogger("httpx")
        if main_level <= logging.DEBUG:
            httpx_logger.setLevel(logging.INFO)
        else:
            httpx_logger.setLevel(logging.WARNING)

        # Always suppress these noisy loggers
        for logger_name in [
            "httpcore",
            "urllib3",
            "urllib3.connectionpool",
            "keyring",
            "asyncio",
            "httpx",
        ]:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.WARNING)


def configure_logging_from_settings(
    log_level: str = "INFO",
    log_format: str = "rich",
    enable_observability: bool = False,
    show_path: bool = False,
    show_time: bool = True,
    console_width: int | None = None,
) -> None:
    """Configure logging system from application settings.

    This is the main entry point for logging configuration that integrates
    with the application's configuration system.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Output format - "rich" for development, "json" for production
        enable_observability: Whether to enable observability pipeline integration
        show_path: Whether to show the module path in logs
        show_time: Whether to show timestamps in logs
        console_width: Optional console width override
    """
    # Determine format based on log level and environment
    if log_format == "auto":
        # Auto-detect format based on environment
        format_type = "rich" if log_level == "DEBUG" else "json"
    else:
        format_type = log_format

    # Setup dual logging with the determined format
    setup_dual_logging(
        level=log_level,
        format_type=format_type,
        enable_observability=enable_observability,
        show_path=show_path,
        show_time=show_time,
        console_width=console_width,
        configure_uvicorn=True,
    )


def get_logging_config_from_settings() -> dict[str, Any]:
    """Get logging configuration from application settings.

    This function extracts logging-related configuration from the application
    settings and returns a dictionary suitable for configuring the logging system.

    Returns:
        Dictionary with logging configuration parameters
    """
    try:
        from ccproxy.config import get_settings

        settings = get_settings()

        # Determine log format based on server settings with observability fallback
        log_format = settings.server.log_format
        if log_format == "auto":
            # Check observability settings first, then fallback to development detection
            if (
                hasattr(settings.observability, "logging_format")
                and settings.observability.logging_format != "auto"
            ):
                log_format = settings.observability.logging_format
            else:
                log_format = "rich" if settings.is_development else "json"

        # Extract logging configuration from settings
        return {
            "log_level": settings.server.log_level,
            "log_format": log_format,
            "enable_observability": settings.observability.enabled,
            "show_path": settings.server.log_show_path
            or settings.server.log_level == "DEBUG",
            "show_time": settings.server.log_show_time,
            "console_width": settings.server.log_console_width,
        }
    except ImportError:
        # Fallback if settings are not available
        return {
            "log_level": "INFO",
            "log_format": "rich",
            "enable_observability": False,
            "show_path": False,
            "show_time": True,
            "console_width": None,
        }


def setup_logging_from_config() -> None:
    """Setup logging system using application configuration.

    This is a convenience function that loads configuration from settings
    and applies it to the logging system.
    """
    config = get_logging_config_from_settings()
    configure_logging_from_settings(**config)


def migrate_to_structlog(logger_name: str) -> tuple[Any, Any]:
    """Migration helper to get both standard and structlog loggers.

    This function helps with gradual migration from standard logging to structlog
    by returning both logger types for the same name.

    Args:
        logger_name: Logger name (typically __name__)

    Returns:
        Tuple of (standard_logger, structlog_logger)
    """
    standard_logger = get_logger(logger_name)
    structlog_logger = get_structlog_logger(logger_name)
    return standard_logger, structlog_logger


def log_with_context(
    logger: Any,
    level: str,
    message: str,
    **context: Any,
) -> None:
    """Helper function to log with structured context.

    This function works with both standard loggers and structlog loggers,
    automatically detecting the type and using appropriate methods.

    Args:
        logger: Logger instance (standard or structlog)
        level: Log level (debug, info, warning, error, critical)
        message: Log message
        **context: Additional context to include in structured logs
    """
    log_method = getattr(logger, level.lower(), None)
    if not log_method:
        return

    # Check if it's a structlog logger
    if hasattr(logger, "bind"):
        # Structlog logger - use context binding
        if context:
            logger = logger.bind(**context)
        log_method(message)
    else:
        # Standard logger - log with extra context
        if context:
            # Format context for standard logger
            context_str = " ".join(f"{k}={v}" for k, v in context.items())
            message = f"{message} [{context_str}]"
        log_method(message)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return structlog.getLogger(name)


# HTTP request/response logging helpers
async def log_http_request(request: httpx.Request) -> None:
    """Log httpx request details."""


async def log_http_response(response: httpx.Response) -> None:
    """Log httpx response details."""


def get_http_event_hooks() -> dict[str, list[Any]]:
    """Get httpx event hooks for request/response logging.

    Returns:
        Event hooks dictionary for httpx.AsyncClient
    """
    return {}


class CustomFormatter(DefaultFormatter):
    """Custom formatter for uvicorn logs with rich toolkit integration."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        from ccproxy.cli.helpers import get_rich_toolkit

        self.toolkit = get_rich_toolkit()

    def formatMessage(self, record: logging.LogRecord) -> str:  # noqa: N802
        return self.toolkit.print_as_string(record.getMessage(), tag=record.levelname)


def get_uvicorn_log_config() -> dict[str, Any]:
    """Get uvicorn logging configuration.

    Returns:
        Dictionary with uvicorn logging configuration
    """
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
