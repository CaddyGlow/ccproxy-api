"""Endpoint mapping definitions for llms adapters.

This module documents and centralizes the key field mappings between:
- OpenAI Chat Completions <-> Anthropic Messages
- OpenAI Responses <-> Anthropic Messages
- OpenAI Chat Completions <-> OpenAI Responses

It is intentionally minimal and focused on the core, broadly supported fields
that we can map reliably across formats. Adapters implement the behavior and
use these helpers to keep mapping rules consistent.
"""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Finish reason mapping (Anthropic -> OpenAI ChatCompletions)
# ---------------------------------------------------------------------------

ANTHROPIC_TO_OPENAI_FINISH_REASON: Final[dict[str, str]] = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    # Anthropic-specific values mapped to closest reasonable OpenAI value
    "pause_turn": "stop",
    "refusal": "stop",
}


OPENAI_TO_ANTHROPIC_STOP_REASON: Final[dict[str, str]] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
}


# ---------------------------------------------------------------------------
# Error type mapping (OpenAI <-> Anthropic)
# ---------------------------------------------------------------------------

OPENAI_TO_ANTHROPIC_ERROR_TYPE: Final[dict[str, str]] = {
    "invalid_request_error": "invalid_request_error",
    "authentication_error": "invalid_request_error",
    "permission_error": "invalid_request_error",
    "not_found_error": "invalid_request_error",
    "rate_limit_error": "rate_limit_error",
    "internal_server_error": "api_error",
    "overloaded_error": "api_error",
}

ANTHROPIC_TO_OPENAI_ERROR_TYPE: Final[dict[str, str]] = {
    "invalid_request_error": "invalid_request_error",
    "authentication_error": "authentication_error",
    "permission_error": "permission_error",
    "not_found_error": "invalid_request_error",  # OpenAI doesn't have not_found_error
    "rate_limit_error": "rate_limit_error",
    "api_error": "api_error",
    "overloaded_error": "api_error",  # Map to generic api_error
    "billing_error": "invalid_request_error",  # Map to invalid_request_error
    "timeout_error": "api_error",  # Map to generic api_error
}


# ---------------------------------------------------------------------------
# Error conversion helper functions
# ---------------------------------------------------------------------------


def convert_openai_error_to_anthropic(error: BaseModel) -> BaseModel:
    """Convert OpenAI error to Anthropic error format."""
    from ccproxy.llms.anthropic.models import (
        APIError,
        ErrorType,
        InvalidRequestError,
        RateLimitError,
    )
    from ccproxy.llms.anthropic.models import (
        ErrorResponse as AnthropicErrorResponse,
    )
    from ccproxy.llms.openai.models import ErrorResponse as OpenAIErrorResponse

    # Handle OpenAI ErrorResponse format
    if isinstance(error, OpenAIErrorResponse):
        openai_error = error.error
        error_message = openai_error.message
        openai_error_type = openai_error.type or "api_error"

        # Map to Anthropic error type
        anthropic_error_type = OPENAI_TO_ANTHROPIC_ERROR_TYPE.get(
            openai_error_type, "api_error"
        )

        # Create appropriate Anthropic error model
        anthropic_error: ErrorType
        if anthropic_error_type == "invalid_request_error":
            anthropic_error = InvalidRequestError(message=error_message)
        elif anthropic_error_type == "rate_limit_error":
            anthropic_error = RateLimitError(message=error_message)
        else:
            anthropic_error = APIError(message=error_message)

        return AnthropicErrorResponse(error=anthropic_error)

    # Handle generic BaseModel errors or malformed errors
    if hasattr(error, "error") and hasattr(error.error, "message"):
        # Try to extract message from nested error structure
        error_message = error.error.message
        fallback_error: ErrorType = APIError(message=error_message)
        return AnthropicErrorResponse(error=fallback_error)

    # Fallback for unknown error formats
    error_message = "Unknown error occurred"
    if hasattr(error, "message"):
        error_message = error.message
    elif hasattr(error, "model_dump"):
        # Try to extract any available message from model dump
        error_dict = error.model_dump()
        error_message = str(error_dict.get("message", error_dict))

    generic_error: ErrorType = APIError(message=error_message)
    return AnthropicErrorResponse(error=generic_error)


def convert_anthropic_error_to_openai(error: BaseModel) -> BaseModel:
    """Convert Anthropic error to OpenAI error format."""
    from ccproxy.llms.anthropic.models import ErrorResponse as AnthropicErrorResponse
    from ccproxy.llms.openai.models import ErrorDetail
    from ccproxy.llms.openai.models import ErrorResponse as OpenAIErrorResponse

    # Handle Anthropic ErrorResponse format
    if isinstance(error, AnthropicErrorResponse):
        anthropic_error = error.error
        error_message = anthropic_error.message

        # Get error type, defaulting to "api_error"
        anthropic_error_type = "api_error"
        if hasattr(anthropic_error, "type"):
            anthropic_error_type = anthropic_error.type

        # Map to OpenAI error type
        openai_error_type = ANTHROPIC_TO_OPENAI_ERROR_TYPE.get(
            anthropic_error_type, "api_error"
        )

        return OpenAIErrorResponse(
            error=ErrorDetail(
                message=error_message,
                type=openai_error_type,
                code=None,
                param=None,
            )
        )

    # Handle generic BaseModel errors or malformed errors
    if hasattr(error, "error") and hasattr(error.error, "message"):
        # Try to extract message from nested error structure
        error_message = error.error.message
        return OpenAIErrorResponse(
            error=ErrorDetail(
                message=error_message,
                type="api_error",
                code=None,
                param=None,
            )
        )

    # Fallback for unknown error formats
    error_message = "Unknown error occurred"
    if hasattr(error, "message"):
        error_message = error.message
    elif hasattr(error, "model_dump"):
        # Try to extract any available message from model dump
        error_dict = error.model_dump()
        if isinstance(error_dict, dict):
            error_message = error_dict.get("message", str(error_dict))

    return OpenAIErrorResponse(
        error=ErrorDetail(
            message=error_message,
            type="api_error",
            code=None,
            param=None,
        )
    )


def normalize_openai_error(error: BaseModel) -> BaseModel:
    """Normalize and validate OpenAI error format."""
    from ccproxy.llms.openai.models import ErrorDetail, ErrorResponse

    # If it's already a proper OpenAI ErrorResponse, return as-is
    if isinstance(error, ErrorResponse):
        return error

    # Try to create a proper ErrorResponse from malformed input
    if hasattr(error, "error") and hasattr(error.error, "message"):
        # Extract details from nested error structure
        nested_error = error.error
        return ErrorResponse(
            error=ErrorDetail(
                message=nested_error.message,
                code=getattr(nested_error, "code", None),
                param=getattr(nested_error, "param", None),
                type=getattr(nested_error, "type", None),
            )
        )

    # Fallback for unknown error formats - create generic ErrorResponse
    error_message = "Unknown error occurred"
    if hasattr(error, "message"):
        error_message = error.message
    elif hasattr(error, "model_dump"):
        # Try to extract any available message from model dump
        error_dict = error.model_dump()
        if isinstance(error_dict, dict):
            error_message = error_dict.get("message", str(error_dict))

    return ErrorResponse(
        error=ErrorDetail(
            message=error_message,
            code=None,
            param=None,
            type="api_error",
        )
    )


# ---------------------------------------------------------------------------
# Minimal defaults used by adapters
# ---------------------------------------------------------------------------

DEFAULT_MAX_TOKENS: Final[int] = 1024


__all__ = [
    "ANTHROPIC_TO_OPENAI_FINISH_REASON",
    "OPENAI_TO_ANTHROPIC_STOP_REASON",
    "OPENAI_TO_ANTHROPIC_ERROR_TYPE",
    "ANTHROPIC_TO_OPENAI_ERROR_TYPE",
    "convert_openai_error_to_anthropic",
    "convert_anthropic_error_to_openai",
    "normalize_openai_error",
    "DEFAULT_MAX_TOKENS",
]
