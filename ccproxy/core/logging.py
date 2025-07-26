import logging
import sys
from pathlib import Path

import structlog
from structlog.stdlib import BoundLogger
from structlog.typing import Processor


def configure_structlog(log_level: int = logging.INFO) -> None:
    """Configure structlog with shared processors following canonical pattern."""
    # Shared processors for all structlog loggers
    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,  # For request context in web apps
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
    ]

    # Add debug-specific processors
    if log_level < logging.INFO:
        # Dev mode (DEBUG): add callsite information
        processors.append(
            structlog.processors.CallsiteParameterAdder(
                parameters=[
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.LINENO,
                ]
            )
        )

    # Common processors for all log levels
    processors.extend(
        [
            # Use human-readable timestamp for structlog logs in debug mode, normal otherwise
            structlog.processors.TimeStamper(
                fmt="%H:%M:%S" if log_level < logging.INFO else "%Y-%m-%d %H:%M:%S"
            ),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,  # Handle exceptions properly
            # This MUST be the last processor - allows different renderers per handler
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ]
    )

    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,  # Cache for performance
    )


def setup_logging(
    json_logs: bool = False, log_level_name: str = "DEBUG", log_file: str | None = None
) -> BoundLogger:
    """
    Setup logging for the entire application using canonical structlog pattern.
    Returns a structlog logger instance.
    """
    log_level = getattr(logging, log_level_name.upper(), logging.INFO)

    # Get root logger and set level BEFORE configuring structlog
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # 1. Configure structlog with shared processors
    configure_structlog(log_level=log_level)

    # 2. Setup root logger handlers
    root_logger.handlers = []  # Clear any existing handlers

    # 3. Create shared processors for foreign (stdlib) logs
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.dev.set_exc_info,
    ]

    # Add debug processors if needed
    if log_level < logging.INFO:
        shared_processors.append(
            structlog.processors.CallsiteParameterAdder(  # type: ignore[arg-type]
                parameters=[
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.LINENO,
                ]
            )
        )

    # Add appropriate timestamper for console vs file
    console_timestamper = (
        structlog.processors.TimeStamper(fmt="%H:%M:%S")
        if log_level < logging.INFO
        else structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S")
    )

    file_timestamper = structlog.processors.TimeStamper(fmt="iso")

    # 4. Setup console handler with ConsoleRenderer
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_renderer = (
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer()
    )

    # Console gets human-readable timestamps for both structlog and stdlib logs
    console_processors = shared_processors + [console_timestamper]
    console_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=console_processors,
            processor=console_renderer,
        )
    )
    root_logger.addHandler(console_handler)

    # 5. Setup file handler with JSONRenderer (if log_file provided)
    if log_file:
        # Ensure parent directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8", delay=True)
        file_handler.setLevel(log_level)

        # File gets ISO timestamps for both structlog and stdlib logs
        file_processors = shared_processors + [file_timestamper]
        file_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                foreign_pre_chain=file_processors,
                processor=structlog.processors.JSONRenderer(),
            )
        )
        root_logger.addHandler(file_handler)

    # 6. Configure stdlib loggers to propagate to our handlers
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

        # In DEBUG mode, let all logs through at DEBUG level
        # Otherwise, reduce uvicorn noise by setting to WARNING
        if log_level == logging.DEBUG:
            logger.setLevel(logging.DEBUG)
        elif logger_name.startswith("uvicorn"):
            logger.setLevel(logging.WARNING)
        else:
            logger.setLevel(log_level)

    # Configure httpx logger separately - INFO when app is DEBUG, WARNING otherwise
    httpx_logger = logging.getLogger("httpx")
    httpx_logger.handlers = []
    httpx_logger.propagate = True
    httpx_logger.setLevel(logging.INFO if log_level < logging.INFO else logging.WARNING)

    # Set noisy HTTP-related loggers to WARNING
    noisy_log_level = logging.WARNING if log_level <= logging.WARNING else log_level
    for noisy_logger_name in [
        "urllib3",
        "urllib3.connectionpool",
        "requests",
        "aiohttp",
        "httpcore",
        "httpcore.http11",
        "fastapi_mcp",
        "sse_starlette",
        "mcp",
    ]:
        noisy_logger = logging.getLogger(noisy_logger_name)
        noisy_logger.handlers = []
        noisy_logger.propagate = True
        noisy_logger.setLevel(noisy_log_level)

    return structlog.get_logger()  # type: ignore[no-any-return]


# Create a convenience function for getting loggers
def get_logger(name: str | None = None) -> BoundLogger:
    """Get a structlog logger instance."""
    return structlog.get_logger(name)  # type: ignore[no-any-return]
