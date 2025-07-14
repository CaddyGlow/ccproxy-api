"""
Observability configuration for structlog and metrics pipeline.

This module configures structlog to integrate with the metrics storage pipeline,
enabling structured logging that feeds into the observability system.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Mapping, MutableMapping
from typing import Any

import structlog
from structlog.stdlib import LoggerFactory

from .pipeline import create_structlog_processor


def configure_observability(
    format_type: str = "json",
    level: str = "INFO",
    enable_pipeline: bool = True,
    show_path: bool = False,
    show_time: bool = True,
) -> None:
    """
    Configure structlog and observability system with flexible formatting.

    This sets up structlog processors to integrate with the metrics pipeline,
    ensuring that relevant log events are captured for analytics. Now supports
    both Rich and JSON output formats.

    Args:
        format_type: Output format - "rich" for development, "json" for production
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_pipeline: Whether to enable pipeline processor for metrics storage
        show_path: Whether to show the module path in logs
        show_time: Whether to show timestamps in logs
    """
    # Base processors that are always included
    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
    ]

    # Add timestamp if requested
    if show_time:
        processors.append(structlog.processors.TimeStamper(fmt="iso"))

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

    # Add pipeline processor if enabled
    if enable_pipeline:
        import contextlib

        with contextlib.suppress(ImportError):
            processors.append(create_structlog_processor())

    # Add format-specific processors
    if format_type == "rich":
        # For Rich output, use Rich-compatible processors
        try:
            # Import the Rich processor from core logging
            from ccproxy.core.logging import _rich_structlog_processor

            processors.extend(
                [
                    _rich_structlog_processor,
                    structlog.dev.ConsoleRenderer(colors=True),
                ]
            )
        except ImportError:
            # Fallback to plain console renderer
            processors.append(structlog.dev.ConsoleRenderer(colors=True))
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
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
        force=True,
    )


def get_observability_status() -> dict[str, Any]:
    """
    Get current observability system status.

    Returns:
        Dictionary with observability system status information
    """
    return {
        "structlog_configured": True,
        "pipeline_processor_enabled": True,
        "format_support": ["rich", "json", "plain"],
        "processors": [
            "merge_contextvars",
            "add_log_level",
            "TimeStamper (conditional)",
            "CallsiteParameterAdder (conditional)",
            "pipeline_processor (conditional)",
            "format_specific_renderer",
        ],
        "integrations": {
            "rich_console": True,
            "json_structured": True,
            "observability_pipeline": True,
        },
    }
