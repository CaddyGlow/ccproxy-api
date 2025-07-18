import logging
import sys
from collections.abc import MutableMapping
from typing import Any

import structlog
from structlog.stdlib import BoundLogger
from structlog.typing import Processor


def configure_structlog(json_logs: bool = False, log_level: str = "INFO") -> None:
    """Configure structlog with your preferred processors."""
    # Use different timestamp format based on log level
    # Dev mode (DEBUG): only hours without microseconds
    # Info mode: full date without microseconds
    if log_level.upper() == "DEBUG":
        timestamper = structlog.processors.TimeStamper(fmt="%H:%M:%S")
    else:
        timestamper = structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S")

    # Processors that will be used for structlog loggers
    processors: list[Processor] = [
        structlog.stdlib.filter_by_level,
    ]

    # Only add logger name if NOT in INFO mode
    if log_level.upper() != "INFO":
        processors.append(structlog.stdlib.add_logger_name)

    processors.extend(
        [
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            timestamper,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
        ]
    )

    # Only add CallsiteParameterAdder if NOT in INFO mode
    if log_level.upper() != "INFO":
        processors.append(
            structlog.processors.CallsiteParameterAdder(
                parameters=[
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.LINENO,
                ]
            )
        )

    # This wrapper passes the event dictionary to the ProcessorFormatter
    # so we don't double-render
    processors.append(structlog.stdlib.ProcessorFormatter.wrap_for_formatter)

    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,  # Don't cache to allow reconfiguration
    )


def setup_logging(
    json_logs: bool = False, log_level: str = "INFO", log_file: str | None = None
) -> BoundLogger:
    """
    Setup logging for the entire application including uvicorn and fastapi.
    Returns a structlog logger instance.
    """
    # Set the log level for the root logger first so structlog can see it
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Configure structlog after setting the log level
    configure_structlog(json_logs=json_logs, log_level=log_level)

    # Create a handler that will format stdlib logs through structlog
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Use the appropriate renderer based on json_logs setting
    renderer = (
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer()
    )

    # Use ProcessorFormatter to handle both structlog and stdlib logs
    # Use the same timestamp format for foreign logs
    if log_level.upper() == "DEBUG":
        foreign_timestamper = structlog.processors.TimeStamper(fmt="%H:%M:%S")
    else:
        foreign_timestamper = structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S")

    # Build foreign_pre_chain conditionally
    foreign_pre_chain: list[Processor] = [structlog.stdlib.add_log_level]

    # Only add logger name if NOT in INFO mode
    if log_level.upper() != "INFO":
        foreign_pre_chain.append(structlog.stdlib.add_logger_name)

    foreign_pre_chain.append(foreign_timestamper)

    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=renderer,
            foreign_pre_chain=foreign_pre_chain,
        )
    )

    # Configure root logger (level already set above)
    handlers: list[logging.Handler] = [handler]

    # Add file handler if log_file is specified
    if log_file:
        from pathlib import Path

        # Ensure parent directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Create a file handler that always outputs JSON
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        file_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processor=structlog.processors.JSONRenderer(),
                foreign_pre_chain=foreign_pre_chain,
            )
        )
        handlers.append(file_handler)

    root_logger.handlers = handlers

    # Make sure uvicorn and fastapi loggers use our configuration
    for logger_name in [
        "uvicorn",
        "uvicorn.access",
        "uvicorn.error",
        "fastapi",
        "ccproxy",
    ]:
        logger = logging.getLogger(logger_name)
        logger.handlers = []  # Remove default handlers
        logger.propagate = True  # Use root logger's handlers

        # Set uvicorn loggers to WARNING when app log level is INFO to reduce noise
        if logger_name.startswith("uvicorn") and log_level.upper() == "INFO":
            logger.setLevel(logging.WARNING)
        else:
            logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Configure httpx logger separately - INFO when app is DEBUG, WARNING otherwise
    httpx_logger = logging.getLogger("httpx")
    httpx_logger.handlers = []  # Remove default handlers
    httpx_logger.propagate = True  # Use root logger's handlers
    if log_level.upper() == "DEBUG":
        httpx_logger.setLevel(logging.INFO)
    else:
        httpx_logger.setLevel(logging.WARNING)

    # Set noisy HTTP-related loggers to WARNING when app log level >= WARNING, else use app log level
    app_log_level = getattr(logging, log_level.upper(), logging.INFO)
    noisy_log_level = (
        logging.WARNING if app_log_level <= logging.WARNING else app_log_level
    )

    for noisy_logger_name in [
        "urllib3",
        "urllib3.connectionpool",
        "requests",
        "aiohttp",
        "httpcore",
        "httpcore.http11",
    ]:
        noisy_logger = logging.getLogger(noisy_logger_name)
        noisy_logger.handlers = []  # Remove default handlers
        noisy_logger.propagate = True  # Use root logger's handlers
        noisy_logger.setLevel(noisy_log_level)

    return structlog.get_logger()  # type: ignore[no-any-return]


# Create a convenience function for getting loggers
def get_logger(name: str | None = None) -> BoundLogger:
    """Get a structlog logger instance."""
    return structlog.get_logger(name)  # type: ignore[no-any-return]
