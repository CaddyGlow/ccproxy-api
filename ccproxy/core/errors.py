"""Core error types for the proxy system."""

from typing import Any

from fastapi import HTTPException


class ProxyHTTPException(HTTPException):
    pass


class ProxyError(Exception):
    """Base exception for all proxy-related errors."""

    def __init__(self, message: str, cause: Exception | None = None):
        """Initialize with a message and optional cause.

        Args:
            message: The error message
            cause: The underlying exception that caused this error
        """
        super().__init__(message)
        self.cause = cause
        if cause:
            # Use Python's exception chaining
            self.__cause__ = cause


class TransformationError(ProxyError):
    """Error raised during data transformation."""

    def __init__(self, message: str, data: Any = None, cause: Exception | None = None):
        """Initialize with a message, optional data, and cause.

        Args:
            message: The error message
            data: The data that failed to transform
            cause: The underlying exception
        """
        super().__init__(message, cause)
        self.data = data


class MiddlewareError(ProxyError):
    """Error raised during middleware execution."""

    def __init__(
        self,
        message: str,
        middleware_name: str | None = None,
        cause: Exception | None = None,
    ):
        """Initialize with a message, middleware name, and cause.

        Args:
            message: The error message
            middleware_name: The name of the middleware that failed
            cause: The underlying exception
        """
        super().__init__(message, cause)
        self.middleware_name = middleware_name


class ProxyConnectionError(ProxyError):
    """Error raised when proxy connection fails."""

    def __init__(
        self, message: str, url: str | None = None, cause: Exception | None = None
    ):
        """Initialize with a message, URL, and cause.

        Args:
            message: The error message
            url: The URL that failed to connect
            cause: The underlying exception
        """
        super().__init__(message, cause)
        self.url = url


class ProxyTimeoutError(ProxyError):
    """Error raised when proxy operation times out."""

    def __init__(
        self,
        message: str,
        timeout: float | None = None,
        cause: Exception | None = None,
    ):
        """Initialize with a message, timeout value, and cause.

        Args:
            message: The error message
            timeout: The timeout value in seconds
            cause: The underlying exception
        """
        super().__init__(message, cause)
        self.timeout = timeout


class ProxyAuthenticationError(ProxyError):
    """Error raised when proxy authentication fails."""

    def __init__(
        self,
        message: str,
        auth_type: str | None = None,
        cause: Exception | None = None,
    ):
        """Initialize with a message, auth type, and cause.

        Args:
            message: The error message
            auth_type: The type of authentication that failed
            cause: The underlying exception
        """
        super().__init__(message, cause)
        self.auth_type = auth_type


# API-level exceptions (consolidated from exceptions.py)
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


class DockerError(ClaudeProxyError):
    """Docker operation error."""

    def __init__(
        self,
        message: str,
        command: str | None = None,
        cause: Exception | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        error_details = details or {}
        if command:
            error_details["command"] = command
        if cause:
            error_details["cause"] = str(cause)
            error_details["cause_type"] = type(cause).__name__

        super().__init__(
            message=message,
            error_type="docker_error",
            status_code=500,
            details=error_details,
        )


class PermissionRequestError(ClaudeProxyError):
    """Base exception for permission request-related errors."""

    pass


class PermissionNotFoundError(PermissionRequestError):
    """Raised when permission request is not found."""

    def __init__(self, confirmation_id: str) -> None:
        super().__init__(
            message=f"Permission request '{confirmation_id}' not found",
            error_type="not_found_error",
            status_code=404,
        )


class PermissionExpiredError(PermissionRequestError):
    """Raised when permission request has expired."""

    def __init__(self, confirmation_id: str) -> None:
        super().__init__(
            message=f"Permission request '{confirmation_id}' has expired",
            error_type="expired_error",
            status_code=410,
        )


class PermissionAlreadyResolvedError(PermissionRequestError):
    """Raised when trying to resolve an already resolved request."""

    def __init__(self, confirmation_id: str, status: str) -> None:
        super().__init__(
            message=f"Permission request '{confirmation_id}' already resolved with status: {status}",
            error_type="conflict_error",
            status_code=409,
        )


class PluginResourceError(ProxyError):
    """Error raised when a plugin resource is unavailable or misconfigured.

    This is a general exception for plugins to use when required resources
    (like configuration, external services, or dependencies) are not available.
    """

    def __init__(
        self,
        message: str,
        plugin_name: str | None = None,
        resource_type: str | None = None,
        cause: Exception | None = None,
    ):
        """Initialize with a message and optional details.

        Args:
            message: The error message
            plugin_name: Name of the plugin encountering the error
            resource_type: Type of resource that's unavailable (e.g., "instructions", "config", "auth")
            cause: The underlying exception
        """
        super().__init__(message, cause)
        self.plugin_name = plugin_name
        self.resource_type = resource_type


class PluginLoadError(ProxyError):
    """Error raised when plugin loading fails.

    This exception is used when plugins cannot be loaded due to import errors,
    missing dependencies, missing classes, or other loading-related issues.
    """

    def __init__(
        self,
        message: str,
        plugin_name: str | None = None,
        cause: Exception | None = None,
    ):
        """Initialize with a message and optional details.

        Args:
            message: The error message
            plugin_name: Name of the plugin that failed to load
            cause: The underlying exception
        """
        super().__init__(message, cause)
        self.plugin_name = plugin_name


__all__ = [
    # Core proxy errors
    "ProxyError",
    "TransformationError",
    "MiddlewareError",
    "ProxyConnectionError",
    "ProxyTimeoutError",
    "ProxyAuthenticationError",
    "PluginResourceError",
    "PluginLoadError",
    # API-level errors
    "ClaudeProxyError",
    "ValidationError",
    "AuthenticationError",
    "PermissionError",
    "NotFoundError",
    "RateLimitError",
    "ModelNotFoundError",
    "TimeoutError",
    "ServiceUnavailableError",
    "DockerError",
    # Permission errors
    "PermissionRequestError",
    "PermissionNotFoundError",
    "PermissionExpiredError",
    "PermissionAlreadyResolvedError",
]
