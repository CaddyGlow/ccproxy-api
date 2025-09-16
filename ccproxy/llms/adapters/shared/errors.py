"""Error conversion helpers shared across adapters."""

from __future__ import annotations

from pydantic import BaseModel

from ccproxy.llms.adapters.shared.constants import (
    ANTHROPIC_TO_OPENAI_ERROR_TYPE,
    OPENAI_TO_ANTHROPIC_ERROR_TYPE,
)


def convert_openai_error_to_anthropic(error: BaseModel) -> BaseModel:
    """Convert an OpenAI error payload to the Anthropic envelope."""
    from ccproxy.llms.anthropic.models import (
        APIError,
        ErrorResponse as AnthropicErrorResponse,
        ErrorType,
        InvalidRequestError,
        RateLimitError,
    )
    from ccproxy.llms.openai.models import ErrorResponse as OpenAIErrorResponse

    if isinstance(error, OpenAIErrorResponse):
        openai_error = error.error
        error_message = openai_error.message
        openai_error_type = openai_error.type or "api_error"
        anthropic_error_type = OPENAI_TO_ANTHROPIC_ERROR_TYPE.get(
            openai_error_type, "api_error"
        )

        anthropic_error: ErrorType
        if anthropic_error_type == "invalid_request_error":
            anthropic_error = InvalidRequestError(message=error_message)
        elif anthropic_error_type == "rate_limit_error":
            anthropic_error = RateLimitError(message=error_message)
        else:
            anthropic_error = APIError(message=error_message)

        return AnthropicErrorResponse(error=anthropic_error)

    if hasattr(error, "error") and hasattr(error.error, "message"):
        error_message = error.error.message
        fallback_error: ErrorType = APIError(message=error_message)
        return AnthropicErrorResponse(error=fallback_error)

    error_message = "Unknown error occurred"
    if hasattr(error, "message"):
        error_message = error.message
    elif hasattr(error, "model_dump"):
        error_dict = error.model_dump()
        error_message = str(error_dict.get("message", error_dict))

    generic_error: ErrorType = APIError(message=error_message)
    return AnthropicErrorResponse(error=generic_error)


def convert_anthropic_error_to_openai(error: BaseModel) -> BaseModel:
    """Convert an Anthropic error payload to the OpenAI envelope."""
    from ccproxy.llms.anthropic.models import ErrorResponse as AnthropicErrorResponse
    from ccproxy.llms.openai.models import ErrorDetail
    from ccproxy.llms.openai.models import ErrorResponse as OpenAIErrorResponse

    if isinstance(error, AnthropicErrorResponse):
        anthropic_error = error.error
        error_message = anthropic_error.message
        anthropic_error_type = "api_error"
        if hasattr(anthropic_error, "type"):
            anthropic_error_type = anthropic_error.type

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

    if hasattr(error, "error") and hasattr(error.error, "message"):
        error_message = error.error.message
        return OpenAIErrorResponse(
            error=ErrorDetail(
                message=error_message,
                type="api_error",
                code=None,
                param=None,
            )
        )

    error_message = "Unknown error occurred"
    if hasattr(error, "message"):
        error_message = error.message
    elif hasattr(error, "model_dump"):
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
    """Normalize malformed OpenAI error payloads to the canonical model."""
    from ccproxy.llms.openai.models import ErrorDetail, ErrorResponse

    if isinstance(error, ErrorResponse):
        return error

    if hasattr(error, "error") and hasattr(error.error, "message"):
        nested_error = error.error
        return ErrorResponse(
            error=ErrorDetail(
                message=nested_error.message,
                code=getattr(nested_error, "code", None),
                param=getattr(nested_error, "param", None),
                type=getattr(nested_error, "type", None),
            )
        )

    error_message = "Unknown error occurred"
    if hasattr(error, "message"):
        error_message = error.message
    elif hasattr(error, "model_dump"):
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


__all__ = [
    "convert_openai_error_to_anthropic",
    "convert_anthropic_error_to_openai",
    "normalize_openai_error",
]
