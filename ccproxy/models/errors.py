"""Error response models for Anthropic API compatibility."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    """Error detail information."""

    type: Annotated[str, Field(description="Error type identifier")]
    message: Annotated[str, Field(description="Human-readable error message")]


class AnthropicError(BaseModel):
    """Anthropic API error response format."""

    type: Annotated[Literal["error"], Field(description="Error type")] = "error"
    error: Annotated[ErrorDetail, Field(description="Error details")]


# Note: Specific error model classes were removed as they were unused.
# Error responses are now forwarded directly from the upstream Claude API
# to preserve the exact error format and headers.


def create_error_response(
    error_type: str, message: str, status_code: int = 500
) -> tuple[dict[str, Any], int]:
    """
    Create a standardized error response.

    Args:
        error_type: Type of error (e.g., "invalid_request_error")
        message: Human-readable error message
        status_code: HTTP status code

    Returns:
        Tuple of (error_dict, status_code)
    """
    error_response = AnthropicError(error=ErrorDetail(type=error_type, message=message))
    return error_response.model_dump(), status_code
