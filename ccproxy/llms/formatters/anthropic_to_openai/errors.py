"""Anthropic→OpenAI error conversion entry point."""

from __future__ import annotations

from pydantic import BaseModel

from ccproxy.llms.formatters.constants import ANTHROPIC_TO_OPENAI_ERROR_TYPE
from ccproxy.llms.models.openai import ErrorDetail
from ccproxy.llms.models.openai import ErrorResponse as OpenAIErrorResponse


def convert__anthropic_to_openai__error(error: BaseModel) -> BaseModel:
    """Convert an Anthropic error payload to the OpenAI envelope."""
    from ccproxy.llms.models.anthropic import ErrorResponse as AnthropicErrorResponse

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


__all__ = ["convert__anthropic_to_openai__error"]
