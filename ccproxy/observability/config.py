"""
Observability configuration for structlog and metrics pipeline.

This module configures structlog to integrate with the metrics storage pipeline,
enabling structured logging that feeds into the observability system.
"""

import json
import logging
import sys
from collections.abc import Mapping, MutableMapping
from typing import Any

import structlog
from structlog.stdlib import LoggerFactory

from .pipeline import create_structlog_processor


def configure_observability() -> None:
    """
    Configure structlog and observability system.

    This sets up structlog processors to integrate with the metrics pipeline,
    ensuring that relevant log events are captured for analytics.
    """
    # Configure structlog
    structlog.configure(
        processors=[
            # Add request ID and timestamp
            structlog.contextvars.merge_contextvars,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            # Pipeline processor for metrics storage
            create_structlog_processor(),
            # JSON formatting for structured logs
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=LoggerFactory(),
        context_class=dict,
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging to work with structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO,
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
        "processors": [
            "merge_contextvars",
            "TimeStamper",
            "add_log_level",
            "pipeline_processor",
            "JSONRenderer",
        ],
    }
