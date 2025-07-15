import logging
import sys

import structlog
from structlog.stdlib import BoundLogger


def configure_structlog(json_logs: bool = False) -> None:
    """Configure structlog with your preferred processors."""
    timestamper = structlog.processors.TimeStamper(fmt="iso")

    # Processors that will be used for structlog loggers
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.CallsiteParameterAdder(
            parameters=[
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.LINENO,
            ]
        ),
        # This wrapper passes the event dictionary to the ProcessorFormatter
        # so we don't double-render
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ]

    structlog.configure(
        processors=processors,  # type: ignore[arg-type]
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,  # Don't cache to allow reconfiguration
    )


def setup_logging(json_logs: bool = False, log_level: str = "INFO") -> BoundLogger:
    """
    Setup logging for the entire application including uvicorn and fastapi.
    Returns a structlog logger instance.
    """
    # Set the log level for the root logger first so structlog can see it
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Configure structlog after setting the log level
    configure_structlog(json_logs=json_logs)

    # Create a handler that will format stdlib logs through structlog
    handler = logging.StreamHandler(sys.stdout)

    # Use the appropriate renderer based on json_logs setting
    renderer = (
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer()
    )

    # Use ProcessorFormatter to handle both structlog and stdlib logs
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=renderer,
            foreign_pre_chain=[
                structlog.stdlib.add_log_level,
                structlog.stdlib.add_logger_name,
                structlog.processors.TimeStamper(fmt="iso"),
            ],
        )
    )

    # Configure root logger (level already set above)
    root_logger.handlers = [handler]

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
        logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Configure httpx logger separately - INFO when app is DEBUG, WARNING otherwise
    httpx_logger = logging.getLogger("httpx")
    # httpx_logger.handlers = []  # Remove default handlers
    # httpx_logger.propagate = True  # Use root logger's handlers
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
