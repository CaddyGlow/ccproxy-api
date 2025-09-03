"""OAuth error handling utilities and decorators."""

import functools
import json
from collections.abc import Callable
from typing import Any, TypeVar

import httpx
from pydantic import ValidationError

from ccproxy.auth.exceptions import (
    CredentialsInvalidError,
    CredentialsStorageError,
    OAuthError,
    OAuthTokenRefreshError,
)
from ccproxy.core.logging import get_logger


logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def oauth_error_handler(operation: str) -> Callable[[F], F]:
    """Decorator for consistent OAuth error handling.

    This decorator provides unified error handling for OAuth operations,
    catching common exceptions and converting them to appropriate OAuth errors.

    Args:
        operation: Name of the operation for logging (e.g., "token_exchange")

    Returns:
        Decorated function with error handling

    Example:
        @oauth_error_handler("token_exchange")
        async def exchange_tokens(self, code: str) -> dict:
            # OAuth token exchange logic
            pass
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)

            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                error_detail = _extract_http_error_detail(e.response)

                logger.error(
                    f"{operation}_http_error",
                    status_code=status_code,
                    error_detail=error_detail,
                    operation=operation,
                    exc_info=e,
                    category="auth",
                )

                if status_code == 401:
                    raise OAuthError(
                        f"{operation} failed: Unauthorized - {error_detail}"
                    ) from e
                elif status_code == 403:
                    raise OAuthError(
                        f"{operation} failed: Forbidden - {error_detail}"
                    ) from e
                elif status_code >= 500:
                    raise OAuthError(
                        f"{operation} failed: Server error - {error_detail}"
                    ) from e
                else:
                    raise OAuthError(f"{operation} failed: {error_detail}") from e

            except httpx.TimeoutException as e:
                logger.error(
                    f"{operation}_timeout",
                    operation=operation,
                    error=str(e),
                    exc_info=e,
                    category="auth",
                )
                raise OAuthError(f"{operation} timed out") from e

            except httpx.ConnectError as e:
                logger.error(
                    f"{operation}_connection_error",
                    operation=operation,
                    error=str(e),
                    exc_info=e,
                    category="auth",
                )
                raise OAuthError(f"{operation} failed: Connection error") from e

            except httpx.HTTPError as e:
                logger.error(
                    f"{operation}_http_error",
                    operation=operation,
                    error=str(e),
                    exc_info=e,
                    category="auth",
                )
                raise OAuthError(f"{operation} failed: Network error - {e}") from e

            except json.JSONDecodeError as e:
                logger.error(
                    f"{operation}_json_decode_error",
                    operation=operation,
                    error=str(e),
                    line=e.lineno,
                    exc_info=e,
                    category="auth",
                )
                raise OAuthError(f"{operation} failed: Invalid JSON response") from e

            except ValidationError as e:
                logger.error(
                    f"{operation}_validation_error",
                    operation=operation,
                    error=str(e),
                    exc_info=e,
                    category="auth",
                )
                raise OAuthError(
                    f"{operation} failed: Invalid data format - {e}"
                ) from e

            except CredentialsStorageError as e:
                logger.error(
                    f"{operation}_storage_error",
                    operation=operation,
                    error=str(e),
                    exc_info=e,
                    category="auth",
                )
                raise  # Re-raise storage errors as-is

            except CredentialsInvalidError as e:
                logger.error(
                    f"{operation}_credentials_invalid",
                    operation=operation,
                    error=str(e),
                    exc_info=e,
                    category="auth",
                )
                raise  # Re-raise credential errors as-is

            except OAuthError:
                raise  # Re-raise OAuth errors as-is

            except Exception as e:
                logger.error(
                    f"{operation}_unexpected_error",
                    operation=operation,
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=e,
                    category="auth",
                )
                raise OAuthError(f"{operation} failed: Unexpected error - {e}") from e

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)

            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                error_detail = _extract_http_error_detail(e.response)

                logger.error(
                    f"{operation}_http_error",
                    status_code=status_code,
                    error_detail=error_detail,
                    operation=operation,
                    exc_info=e,
                    category="auth",
                )

                if status_code == 401:
                    raise OAuthError(
                        f"{operation} failed: Unauthorized - {error_detail}"
                    ) from e
                elif status_code == 403:
                    raise OAuthError(
                        f"{operation} failed: Forbidden - {error_detail}"
                    ) from e
                elif status_code >= 500:
                    raise OAuthError(
                        f"{operation} failed: Server error - {error_detail}"
                    ) from e
                else:
                    raise OAuthError(f"{operation} failed: {error_detail}") from e

            except httpx.TimeoutException as e:
                logger.error(
                    f"{operation}_timeout",
                    operation=operation,
                    error=str(e),
                    exc_info=e,
                    category="auth",
                )
                raise OAuthError(f"{operation} timed out") from e

            except httpx.HTTPError as e:
                logger.error(
                    f"{operation}_http_error",
                    operation=operation,
                    error=str(e),
                    exc_info=e,
                    category="auth",
                )
                raise OAuthError(f"{operation} failed: Network error - {e}") from e

            except json.JSONDecodeError as e:
                logger.error(
                    f"{operation}_json_decode_error",
                    operation=operation,
                    error=str(e),
                    exc_info=e,
                    category="auth",
                )
                raise OAuthError(f"{operation} failed: Invalid JSON response") from e

            except ValidationError as e:
                logger.error(
                    f"{operation}_validation_error",
                    operation=operation,
                    error=str(e),
                    exc_info=e,
                    category="auth",
                )
                raise OAuthError(
                    f"{operation} failed: Invalid data format - {e}"
                ) from e

            except OAuthError:
                raise  # Re-raise OAuth errors as-is

            except Exception as e:
                logger.error(
                    f"{operation}_unexpected_error",
                    operation=operation,
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=e,
                    category="auth",
                )
                raise OAuthError(f"{operation} failed: Unexpected error - {e}") from e

        # Return appropriate wrapper based on function type
        import asyncio
        import inspect

        if asyncio.iscoroutinefunction(func) or inspect.isasyncgenfunction(func):
            return async_wrapper  # type: ignore
        else:
            return sync_wrapper  # type: ignore

    return decorator


def _extract_http_error_detail(response: httpx.Response) -> str:
    """Extract error detail from HTTP response.

    Args:
        response: HTTP response object

    Returns:
        Human-readable error detail string
    """
    try:
        error_data = response.json()

        # Common OAuth error response formats
        if isinstance(error_data, dict):
            # Standard OAuth error response
            if "error_description" in error_data:
                return str(error_data["error_description"])
            if "error" in error_data:
                error = error_data["error"]
                if isinstance(error, dict) and "message" in error:
                    return str(error["message"])
                return str(error)
            # Generic message field
            if "message" in error_data:
                return str(error_data["message"])
            if "detail" in error_data:
                return str(error_data["detail"])

        # If we can't parse a specific error, return truncated response
        return _truncate_error_text(str(error_data))

    except (json.JSONDecodeError, KeyError, TypeError):
        # Fall back to text response if JSON parsing fails
        return _truncate_error_text(response.text)


def _truncate_error_text(text: str, max_length: int = 200) -> str:
    """Truncate error text to reasonable length.

    Args:
        text: Error text to truncate
        max_length: Maximum length (default 200)

    Returns:
        Truncated error text
    """
    if len(text) <= max_length:
        return text

    # For long errors, show beginning and end
    if len(text) > max_length * 2:
        return f"{text[:max_length]}...{text[-50:]}"
    else:
        return f"{text[:max_length]}..."


class OAuthErrorContext:
    """Context manager for OAuth error handling.

    Provides a context where OAuth errors are handled consistently.

    Example:
        async with OAuthErrorContext("token_refresh"):
            await refresh_tokens()
    """

    def __init__(self, operation: str):
        """Initialize error context.

        Args:
            operation: Name of the operation for logging
        """
        self.operation = operation

    async def __aenter__(self) -> "OAuthErrorContext":
        """Enter async context."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Exit async context with error handling."""
        if exc_type is None:
            return False

        # Handle specific exception types
        if isinstance(exc_val, httpx.HTTPStatusError):
            error_detail = _extract_http_error_detail(exc_val.response)
            logger.error(
                f"{self.operation}_http_error",
                status_code=exc_val.response.status_code,
                error_detail=error_detail,
                operation=self.operation,
                exc_info=exc_val,
            )
            raise OAuthError(f"{self.operation} failed: {error_detail}") from exc_val

        elif isinstance(exc_val, httpx.TimeoutException):
            logger.error(
                f"{self.operation}_timeout",
                operation=self.operation,
                error=str(exc_val),
                exc_info=exc_val,
            )
            raise OAuthError(f"{self.operation} timed out") from exc_val

        elif isinstance(exc_val, httpx.HTTPError):
            logger.error(
                f"{self.operation}_network_error",
                operation=self.operation,
                error=str(exc_val),
                exc_info=exc_val,
            )
            raise OAuthError(f"{self.operation} failed: Network error") from exc_val

        elif isinstance(exc_val, json.JSONDecodeError):
            logger.error(
                f"{self.operation}_json_error",
                operation=self.operation,
                error=str(exc_val),
                exc_info=exc_val,
            )
            raise OAuthError(
                f"{self.operation} failed: Invalid JSON response"
            ) from exc_val

        elif isinstance(exc_val, ValidationError):
            logger.error(
                f"{self.operation}_validation_error",
                operation=self.operation,
                error=str(exc_val),
                exc_info=exc_val,
            )
            raise OAuthError(
                f"{self.operation} failed: Invalid data format"
            ) from exc_val

        elif isinstance(exc_val, OAuthError | OAuthTokenRefreshError):
            # Re-raise OAuth errors as-is
            return False

        else:
            logger.error(
                f"{self.operation}_unexpected_error",
                operation=self.operation,
                error=str(exc_val),
                error_type=type(exc_val).__name__,
                exc_info=exc_val,
            )
            raise OAuthError(f"{self.operation} failed: {exc_val}") from exc_val
