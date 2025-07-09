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


class InvalidRequestError(AnthropicError):
    """Invalid request error (400)."""

    error: Annotated[ErrorDetail, Field(description="Error details")] = ErrorDetail(
        type="invalid_request_error", message="Invalid request"
    )


class AuthenticationError(AnthropicError):
    """Authentication error (401)."""

    error: Annotated[ErrorDetail, Field(description="Error details")] = ErrorDetail(
        type="authentication_error", message="Authentication failed"
    )


class PermissionError(AnthropicError):
    """Permission error (403)."""

    error: Annotated[ErrorDetail, Field(description="Error details")] = ErrorDetail(
        type="permission_error", message="Permission denied"
    )


class NotFoundError(AnthropicError):
    """Not found error (404)."""

    error: Annotated[ErrorDetail, Field(description="Error details")] = ErrorDetail(
        type="not_found_error", message="Resource not found"
    )


class RateLimitError(AnthropicError):
    """Rate limit error (429)."""

    error: Annotated[ErrorDetail, Field(description="Error details")] = ErrorDetail(
        type="rate_limit_error", message="Rate limit exceeded"
    )


class InternalServerError(AnthropicError):
    """Internal server error (500)."""

    error: Annotated[ErrorDetail, Field(description="Error details")] = ErrorDetail(
        type="internal_server_error", message="Internal server error"
    )


class ServiceUnavailableError(AnthropicError):
    """Service unavailable error (503)."""

    error: Annotated[ErrorDetail, Field(description="Error details")] = ErrorDetail(
        type="service_unavailable_error", message="Service temporarily unavailable"
    )


# Streaming error format
class StreamingError(BaseModel):
    """Streaming error message format."""

    type: Annotated[Literal["error"], Field(description="Error type")] = "error"
    error: Annotated[ErrorDetail, Field(description="Error details")]


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
