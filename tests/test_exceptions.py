"""Tests for custom exceptions."""

import pytest

from ccproxy.exceptions import (
    AuthenticationError,
    ClaudeProxyError,
    ModelNotFoundError,
    NotFoundError,
    PermissionError,
    RateLimitError,
    ServiceUnavailableError,
    TimeoutError,
    ValidationError,
)


@pytest.mark.unit
class TestClaudeProxyError:
    """Test ClaudeProxyError base exception."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default values."""
        error = ClaudeProxyError("Test message")
        assert str(error) == "Test message"
        assert error.message == "Test message"
        assert error.error_type == "internal_server_error"
        assert error.status_code == 500
        assert error.details == {}

    def test_init_with_custom_values(self) -> None:
        """Test initialization with custom values."""
        details = {"field": "value"}
        error = ClaudeProxyError(
            message="Custom message",
            error_type="custom_error",
            status_code=400,
            details=details,
        )
        assert str(error) == "Custom message"
        assert error.message == "Custom message"
        assert error.error_type == "custom_error"
        assert error.status_code == 400
        assert error.details == details

    def test_init_with_none_details(self) -> None:
        """Test initialization with None details."""
        error = ClaudeProxyError("Test message", details=None)
        assert error.details == {}


@pytest.mark.unit
class TestValidationError:
    """Test ValidationError exception."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default values."""
        error = ValidationError("Validation failed")
        assert str(error) == "Validation failed"
        assert error.message == "Validation failed"
        assert error.error_type == "invalid_request_error"
        assert error.status_code == 400
        assert error.details == {}

    def test_init_with_details(self) -> None:
        """Test initialization with details."""
        details = {"field": "required"}
        error = ValidationError("Validation failed", details=details)
        assert error.details == details


@pytest.mark.unit
class TestAuthenticationError:
    """Test AuthenticationError exception."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default message."""
        error = AuthenticationError()
        assert str(error) == "Authentication failed"
        assert error.message == "Authentication failed"
        assert error.error_type == "authentication_error"
        assert error.status_code == 401

    def test_init_with_custom_message(self) -> None:
        """Test initialization with custom message."""
        error = AuthenticationError("Invalid API key")
        assert str(error) == "Invalid API key"
        assert error.message == "Invalid API key"


@pytest.mark.unit
class TestPermissionError:
    """Test PermissionError exception."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default message."""
        error = PermissionError()
        assert str(error) == "Permission denied"
        assert error.message == "Permission denied"
        assert error.error_type == "permission_error"
        assert error.status_code == 403

    def test_init_with_custom_message(self) -> None:
        """Test initialization with custom message."""
        error = PermissionError("Access denied")
        assert str(error) == "Access denied"
        assert error.message == "Access denied"


@pytest.mark.unit
class TestNotFoundError:
    """Test NotFoundError exception."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default message."""
        error = NotFoundError()
        assert str(error) == "Resource not found"
        assert error.message == "Resource not found"
        assert error.error_type == "not_found_error"
        assert error.status_code == 404

    def test_init_with_custom_message(self) -> None:
        """Test initialization with custom message."""
        error = NotFoundError("Endpoint not found")
        assert str(error) == "Endpoint not found"
        assert error.message == "Endpoint not found"


@pytest.mark.unit
class TestRateLimitError:
    """Test RateLimitError exception."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default message."""
        error = RateLimitError()
        assert str(error) == "Rate limit exceeded"
        assert error.message == "Rate limit exceeded"
        assert error.error_type == "rate_limit_error"
        assert error.status_code == 429

    def test_init_with_custom_message(self) -> None:
        """Test initialization with custom message."""
        error = RateLimitError("Too many requests")
        assert str(error) == "Too many requests"
        assert error.message == "Too many requests"


@pytest.mark.unit
class TestModelNotFoundError:
    """Test ModelNotFoundError exception."""

    def test_init_with_model_name(self) -> None:
        """Test initialization with model name."""
        error = ModelNotFoundError("claude-opus-4-20250514")
        assert str(error) == "Model 'claude-opus-4-20250514' not found"
        assert error.message == "Model 'claude-opus-4-20250514' not found"
        assert error.error_type == "not_found_error"
        assert error.status_code == 404


@pytest.mark.unit
class TestTimeoutError:
    """Test TimeoutError exception."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default message."""
        error = TimeoutError()
        assert str(error) == "Request timeout"
        assert error.message == "Request timeout"
        assert error.error_type == "timeout_error"
        assert error.status_code == 408

    def test_init_with_custom_message(self) -> None:
        """Test initialization with custom message."""
        error = TimeoutError("Connection timeout")
        assert str(error) == "Connection timeout"
        assert error.message == "Connection timeout"


@pytest.mark.unit
class TestServiceUnavailableError:
    """Test ServiceUnavailableError exception."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default message."""
        error = ServiceUnavailableError()
        assert str(error) == "Service temporarily unavailable"
        assert error.message == "Service temporarily unavailable"
        assert error.error_type == "service_unavailable_error"
        assert error.status_code == 503

    def test_init_with_custom_message(self) -> None:
        """Test initialization with custom message."""
        error = ServiceUnavailableError("Service down for maintenance")
        assert str(error) == "Service down for maintenance"
        assert error.message == "Service down for maintenance"
