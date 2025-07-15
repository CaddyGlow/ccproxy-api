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
        cache_logger_on_first_use=True,
    )


def setup_logging(json_logs: bool = False, log_level: str = "INFO") -> BoundLogger:
    """
    Setup logging for the entire application including uvicorn and fastapi.
    Returns a structlog logger instance.
    """
    # Configure structlog first
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

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(log_level)

    # Make sure uvicorn and fastapi loggers use our configuration
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "fastapi"]:
        logger = logging.getLogger(logger_name)
        logger.handlers = []  # Remove default handlers
        logger.propagate = True  # Use root logger's handlers

    return structlog.get_logger()  # type: ignore[no-any-return]


# Create a convenience function for getting loggers
def get_logger(name: str | None = None) -> BoundLogger:
    """Get a structlog logger instance."""
    return structlog.get_logger(name)  # type: ignore[no-any-return]
