"""Generic validation utilities for the CCProxy API."""

import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ccproxy.core.constants import EMAIL_PATTERN, URL_PATTERN, UUID_PATTERN


class ValidationError(Exception):
    """Base class for validation errors."""

    pass


def validate_email(email: str) -> str:
    """Validate email format.

    Args:
        email: Email address to validate

    Returns:
        The validated email address

    Raises:
        ValidationError: If email format is invalid
    """
    if not isinstance(email, str):
        raise ValidationError("Email must be a string")

    if not re.match(EMAIL_PATTERN, email):
        raise ValidationError(f"Invalid email format: {email}")

    return email.strip().lower()


def validate_url(url: str) -> str:
    """Validate URL format.

    Args:
        url: URL to validate

    Returns:
        The validated URL

    Raises:
        ValidationError: If URL format is invalid
    """
    if not isinstance(url, str):
        raise ValidationError("URL must be a string")

    if not re.match(URL_PATTERN, url):
        raise ValidationError(f"Invalid URL format: {url}")

    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ValidationError(f"Invalid URL format: {url}")
    except (ValueError, TypeError) as e:
        raise ValidationError(f"Invalid URL format: {url}") from e
    except Exception as e:
        raise ValidationError(f"Invalid URL format: {url}") from e

    return url.strip()


def validate_uuid(uuid_str: str) -> str:
    """Validate UUID format.

    Args:
        uuid_str: UUID string to validate

    Returns:
        The validated UUID string

    Raises:
        ValidationError: If UUID format is invalid
    """
    if not isinstance(uuid_str, str):
        raise ValidationError("UUID must be a string")

    if not re.match(UUID_PATTERN, uuid_str.lower()):
        raise ValidationError(f"Invalid UUID format: {uuid_str}")

    return uuid_str.strip().lower()


def validate_path(path: str | Path, must_exist: bool = True) -> Path:
    """Validate file system path.

    Args:
        path: Path to validate
        must_exist: Whether the path must exist

    Returns:
        The validated Path object

    Raises:
        ValidationError: If path is invalid
    """
    if isinstance(path, str):
        path = Path(path)
    elif not isinstance(path, Path):
        raise ValidationError("Path must be a string or Path object")

    if must_exist and not path.exists():
        raise ValidationError(f"Path does not exist: {path}")

    return path.resolve()


def validate_port(port: int | str) -> int:
    """Validate port number.

    Args:
        port: Port number to validate

    Returns:
        The validated port number

    Raises:
        ValidationError: If port is invalid
    """
    if isinstance(port, str):
        try:
            port = int(port)
        except ValueError as e:
            raise ValidationError(f"Port must be a valid integer: {port}") from e

    if not isinstance(port, int):
        raise ValidationError(f"Port must be an integer: {port}")

    if port < 1 or port > 65535:
        raise ValidationError(f"Port must be between 1 and 65535: {port}")

    return port


def validate_timeout(timeout: float | int | str) -> float:
    """Validate timeout value.

    Args:
        timeout: Timeout value to validate

    Returns:
        The validated timeout value

    Raises:
        ValidationError: If timeout is invalid
    """
    if isinstance(timeout, str):
        try:
            timeout = float(timeout)
        except ValueError as e:
            raise ValidationError(f"Timeout must be a valid number: {timeout}") from e

    if not isinstance(timeout, int | float):
        raise ValidationError(f"Timeout must be a number: {timeout}")

    if timeout < 0:
        raise ValidationError(f"Timeout must be non-negative: {timeout}")

    return float(timeout)


def validate_non_empty_string(value: str, name: str = "value") -> str:
    """Validate that a string is not empty.

    Args:
        value: String value to validate
        name: Name of the field for error messages

    Returns:
        The validated string

    Raises:
        ValidationError: If string is empty or not a string
    """
    if not isinstance(value, str):
        raise ValidationError(f"{name} must be a string")

    if not value.strip():
        raise ValidationError(f"{name} cannot be empty")

    return value.strip()


def validate_dict(value: Any, required_keys: list[str] | None = None) -> dict[str, Any]:
    """Validate dictionary and required keys.

    Args:
        value: Value to validate as dictionary
        required_keys: List of required keys

    Returns:
        The validated dictionary

    Raises:
        ValidationError: If not a dictionary or missing required keys
    """
    if not isinstance(value, dict):
        raise ValidationError("Value must be a dictionary")

    if required_keys:
        missing_keys = [key for key in required_keys if key not in value]
        if missing_keys:
            raise ValidationError(f"Missing required keys: {missing_keys}")

    return value


def validate_list(
    value: Any, min_length: int = 0, max_length: int | None = None
) -> list[Any]:
    """Validate list and length constraints.

    Args:
        value: Value to validate as list
        min_length: Minimum list length
        max_length: Maximum list length

    Returns:
        The validated list

    Raises:
        ValidationError: If not a list or length constraints are violated
    """
    if not isinstance(value, list):
        raise ValidationError("Value must be a list")

    if len(value) < min_length:
        raise ValidationError(f"List must have at least {min_length} items")

    if max_length is not None and len(value) > max_length:
        raise ValidationError(f"List cannot have more than {max_length} items")

    return value


def validate_choice(value: Any, choices: list[Any], name: str = "value") -> Any:
    """Validate that value is one of the allowed choices.

    Args:
        value: Value to validate
        choices: List of allowed choices
        name: Name of the field for error messages

    Returns:
        The validated value

    Raises:
        ValidationError: If value is not in choices
    """
    if value not in choices:
        raise ValidationError(f"{name} must be one of {choices}, got: {value}")

    return value


def validate_range(
    value: float | int,
    min_value: float | int | None = None,
    max_value: float | int | None = None,
    name: str = "value",
) -> float | int:
    """Validate that a numeric value is within a specified range.

    Args:
        value: Numeric value to validate
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        name: Name of the field for error messages

    Returns:
        The validated value

    Raises:
        ValidationError: If value is outside the allowed range
    """
    if not isinstance(value, int | float):
        raise ValidationError(f"{name} must be a number")

    if min_value is not None and value < min_value:
        raise ValidationError(f"{name} must be at least {min_value}")

    if max_value is not None and value > max_value:
        raise ValidationError(f"{name} must be at most {max_value}")

    return value
