"""OpenAI→Anthropic error conversion entry point."""

from __future__ import annotations

from pydantic import BaseModel

from ccproxy.llms.formatters.constants import OPENAI_TO_ANTHROPIC_ERROR_TYPE
from ccproxy.llms.models import anthropic as anthropic_models
from ccproxy.llms.models import openai as openai_models


def convert__openai_to_anthropic__error(error: BaseModel) -> BaseModel:
    """Convert an OpenAI error payload to the Anthropic envelope."""
    if isinstance(error, openai_models.ErrorResponse):
        openai_error = error.error
        error_message = openai_error.message
        openai_error_type = openai_error.type or "api_error"
        anthropic_error_type = OPENAI_TO_ANTHROPIC_ERROR_TYPE.get(
            openai_error_type, "api_error"
        )

        if anthropic_error_type == "invalid_request_error":
            anthropic_error: anthropic_models.ErrorType = (
                anthropic_models.InvalidRequestError(message=error_message)
            )
        elif anthropic_error_type == "rate_limit_error":
            anthropic_error = anthropic_models.RateLimitError(message=error_message)
        else:
            anthropic_error = anthropic_models.APIError(message=error_message)

        return anthropic_models.ErrorResponse(error=anthropic_error)

    if hasattr(error, "error") and hasattr(error.error, "message"):
        error_message = error.error.message
        fallback_error: anthropic_models.ErrorType = anthropic_models.APIError(
            message=error_message
        )
        return anthropic_models.ErrorResponse(error=fallback_error)

    error_message = "Unknown error occurred"
    if hasattr(error, "message"):
        error_message = error.message
    elif hasattr(error, "model_dump"):
        error_dict = error.model_dump()
        error_message = str(error_dict.get("message", error_dict))

    generic_error: anthropic_models.ErrorType = anthropic_models.APIError(
        message=error_message
    )
    return anthropic_models.ErrorResponse(error=generic_error)


__all__ = ["convert__openai_to_anthropic__error"]
