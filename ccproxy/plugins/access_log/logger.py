"""Utility functions for comprehensive access logging.

This module provides logging utilities adapted from the observability
module for use within the access_log plugin.
"""

import time
from typing import Any

from ccproxy.core.logging import get_plugin_logger


logger = get_plugin_logger(__name__)


async def log_request_access(
    request_id: str,
    method: str | None = None,
    path: str | None = None,
    status_code: int | None = None,
    duration_ms: float | None = None,
    client_ip: str | None = None,
    user_agent: str | None = None,
    query: str | None = None,
    error_message: str | None = None,
    **additional_metadata: Any,
) -> None:
    """Log comprehensive access information for a request.

    This function generates a unified access log entry with complete request
    metadata including timing, tokens, costs, and any additional context.

    Args:
        request_id: Request identifier
        method: HTTP method
        path: Request path
        status_code: HTTP status code
        duration_ms: Request duration in milliseconds
        client_ip: Client IP address
        user_agent: User agent string
        query: Query parameters
        error_message: Error message if applicable
        **additional_metadata: Any additional fields to include
    """
    # Prepare basic log data (always included)
    log_data: dict[str, Any] = {
        "request_id": request_id,
        "method": method,
        "path": path,
        "query": query,
        "client_ip": client_ip,
        "user_agent": user_agent,
    }

    # Add response-specific fields
    log_data.update(
        {
            "status_code": status_code,
            "duration_ms": duration_ms,
            "duration_seconds": duration_ms / 1000 if duration_ms else None,
            "error_message": error_message,
        }
    )

    # Add token and cost metrics if available in metadata
    token_fields = [
        "tokens_input",
        "tokens_output",
        "cache_read_tokens",
        "cache_write_tokens",
        "cost_usd",
        "num_turns",
    ]

    for field in token_fields:
        value = additional_metadata.get(field)
        if value is not None:
            log_data[field] = value

    # Add service and endpoint info
    service_fields = ["endpoint", "model", "streaming", "service_type", "provider"]

    for field in service_fields:
        value = additional_metadata.get(field)
        if value is not None:
            log_data[field] = value

    # Add session context metadata if available
    session_fields = [
        "session_id",
        "session_type",
        "session_status",
        "session_age_seconds",
        "session_message_count",
        "session_pool_enabled",
        "session_idle_seconds",
        "session_error_count",
        "session_is_new",
    ]

    for field in session_fields:
        value = additional_metadata.get(field)
        if value is not None:
            log_data[field] = value

    # Add rate limit headers if available
    rate_limit_fields = [
        "x-ratelimit-limit",
        "x-ratelimit-remaining",
        "x-ratelimit-reset",
        "anthropic-ratelimit-requests-limit",
        "anthropic-ratelimit-requests-remaining",
        "anthropic-ratelimit-requests-reset",
        "anthropic-ratelimit-tokens-limit",
        "anthropic-ratelimit-tokens-remaining",
        "anthropic-ratelimit-tokens-reset",
        "anthropic_request_id",
    ]

    for field in rate_limit_fields:
        value = additional_metadata.get(field)
        if value is not None:
            log_data[field] = value

    # Add any additional metadata provided
    log_data.update(additional_metadata)

    # Remove None values to keep log clean
    log_data = {k: v for k, v in log_data.items() if v is not None}

    # Log with appropriate level
    bound_logger = logger.bind(**log_data)

    if error_message:
        bound_logger.warning("access_log", exc_info=additional_metadata.get("error"))
    else:
        is_streaming = additional_metadata.get("streaming", False)
        is_streaming_complete = (
            additional_metadata.get("event_type", "") == "streaming_complete"
        )

        if not is_streaming or is_streaming_complete:
            bound_logger.info("access_log")
        else:
            # If streaming is true, and not streaming_complete log as debug
            bound_logger.info("access_log_streaming_start")


def log_request_start(
    request_id: str,
    method: str,
    path: str,
    client_ip: str | None = None,
    user_agent: str | None = None,
    query: str | None = None,
    **additional_metadata: Any,
) -> None:
    """Log request start event with basic information.

    This is used for early/hook logging when full context isn't available yet.

    Args:
        request_id: Request identifier
        method: HTTP method
        path: Request path
        client_ip: Client IP address
        user_agent: User agent string
        query: Query parameters
        **additional_metadata: Any additional fields to include
    """
    log_data: dict[str, Any] = {
        "request_id": request_id,
        "method": method,
        "path": path,
        "client_ip": client_ip,
        "user_agent": user_agent,
        "query": query,
        "event_type": "request_start",
        "timestamp": time.time(),
    }

    # Add any additional metadata
    log_data.update(additional_metadata)

    # Remove None values
    log_data = {k: v for k, v in log_data.items() if v is not None}

    logger.debug("access_log_start", **log_data)


async def log_provider_access(
    request_id: str,
    provider: str,
    method: str,
    url: str,
    status_code: int | None = None,
    duration_ms: float | None = None,
    error_message: str | None = None,
    **additional_metadata: Any,
) -> None:
    """Log provider access information.

    Args:
        request_id: Request identifier
        provider: Provider name
        method: HTTP method
        url: Provider URL
        status_code: Response status code
        duration_ms: Request duration in milliseconds
        error_message: Error message if applicable
        **additional_metadata: Any additional fields to include
    """
    log_data: dict[str, Any] = {
        "request_id": request_id,
        "provider": provider,
        "method": method,
        "url": url,
        "status_code": status_code,
        "duration_ms": duration_ms,
        "duration_seconds": duration_ms / 1000 if duration_ms else None,
        "error_message": error_message,
        "event_type": "provider_access",
    }

    # Add token and cost metrics if available
    token_fields = [
        "tokens_input",
        "tokens_output",
        "cache_read_tokens",
        "cache_write_tokens",
        "cost_usd",
        "model",
    ]

    for field in token_fields:
        value = additional_metadata.get(field)
        if value is not None:
            log_data[field] = value

    # Add any additional metadata
    log_data.update(additional_metadata)

    # Remove None values
    log_data = {k: v for k, v in log_data.items() if v is not None}

    # Log with appropriate level
    bound_logger = logger.bind(**log_data)

    if error_message:
        bound_logger.warning(
            "provider_access_log", exc_info=additional_metadata.get("error")
        )
    else:
        bound_logger.info("provider_access_log")
