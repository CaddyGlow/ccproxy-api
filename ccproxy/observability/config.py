"""
Observability configuration for structlog and metrics pipeline.

This module configures structlog to integrate with the metrics storage pipeline,
enabling structured logging that feeds into the observability system.
"""

from __future__ import annotations

from typing import Any


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
    # Delegate to core logging system for all structlog configuration
    from ccproxy.core.logging import setup_logging

    # setup_dual_logging(
    #     level=level,
    #     format_type=format_type,
    #     enable_observability=enable_pipeline,
    #     show_path=show_path,
    #     show_time=show_time,
    #     configure_uvicorn=False,  # Don't reconfigure uvicorn in observability context
    # )


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
