"""Custom exceptions for Claude Proxy API Server."""

from typing import Any


class ClaudeProxyError(Exception):
    """Base exception for Claude Proxy errors."""

    def __init__(
        self,
        message: str,
        error_type: str = "internal_server_error",
        status_code: int = 500,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.status_code = status_code
        self.details = details or {}


class ValidationError(ClaudeProxyError):
    """Validation error (400)."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            error_type="invalid_request_error",
            status_code=400,
            details=details,
        )


class AuthenticationError(ClaudeProxyError):
    """Authentication error (401)."""

    def __init__(self, message: str = "Authentication failed") -> None:
        super().__init__(
            message=message, error_type="authentication_error", status_code=401
        )


class PermissionError(ClaudeProxyError):
    """Permission error (403)."""

    def __init__(self, message: str = "Permission denied") -> None:
        super().__init__(
            message=message, error_type="permission_error", status_code=403
        )


class NotFoundError(ClaudeProxyError):
    """Not found error (404)."""

    def __init__(self, message: str = "Resource not found") -> None:
        super().__init__(message=message, error_type="not_found_error", status_code=404)


class RateLimitError(ClaudeProxyError):
    """Rate limit error (429)."""

    def __init__(self, message: str = "Rate limit exceeded") -> None:
        super().__init__(
            message=message, error_type="rate_limit_error", status_code=429
        )


class ModelNotFoundError(ClaudeProxyError):
    """Model not found error (404)."""

    def __init__(self, model: str) -> None:
        super().__init__(
            message=f"Model '{model}' not found",
            error_type="not_found_error",
            status_code=404,
        )


class TimeoutError(ClaudeProxyError):
    """Request timeout error (408)."""

    def __init__(self, message: str = "Request timeout") -> None:
        super().__init__(message=message, error_type="timeout_error", status_code=408)


class ServiceUnavailableError(ClaudeProxyError):
    """Service unavailable error (503)."""

    def __init__(self, message: str = "Service temporarily unavailable") -> None:
        super().__init__(
            message=message, error_type="service_unavailable_error", status_code=503
        )
