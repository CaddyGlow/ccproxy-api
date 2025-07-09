"""Core error types for the proxy system."""

from typing import Any, Optional


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
