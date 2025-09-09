"""Unit tests for CLI OAuth error taxonomy."""

from ccproxy.auth.oauth.cli_errors import (
    AuthError,
    AuthProviderError,
    AuthTimedOutError,
    AuthUserAbortedError,
    NetworkError,
    PortBindError,
)


class TestAuthErrorHierarchy:
    """Test authentication error hierarchy."""

    def test_auth_error_base_class(self) -> None:
        """Test AuthError base class."""
        error = AuthError("Base auth error")
        assert str(error) == "Base auth error"
        assert isinstance(error, Exception)

    def test_auth_timeout_error(self) -> None:
        """Test AuthTimedOutError."""
        error = AuthTimedOutError("Authentication timed out")
        assert isinstance(error, AuthError)
        assert str(error) == "Authentication timed out"

    def test_auth_user_aborted_error(self) -> None:
        """Test AuthUserAbortedError."""
        error = AuthUserAbortedError("User cancelled authentication")
        assert isinstance(error, AuthError)
        assert str(error) == "User cancelled authentication"

    def test_auth_provider_error(self) -> None:
        """Test AuthProviderError."""
        error = AuthProviderError("Provider-specific error")
        assert isinstance(error, AuthError)
        assert str(error) == "Provider-specific error"

    def test_network_error(self) -> None:
        """Test NetworkError."""
        error = NetworkError("Network connectivity error")
        assert isinstance(error, AuthError)
        assert str(error) == "Network connectivity error"

    def test_port_bind_error(self) -> None:
        """Test PortBindError."""
        error = PortBindError("Failed to bind to port 8080")
        assert isinstance(error, AuthError)
        assert str(error) == "Failed to bind to port 8080"

    def test_error_inheritance_chain(self) -> None:
        """Test that all errors inherit from AuthError."""
        errors = [
            AuthTimedOutError("timeout"),
            AuthUserAbortedError("aborted"),
            AuthProviderError("provider"),
            NetworkError("network"),
            PortBindError("port"),
        ]

        for error in errors:
            assert isinstance(error, AuthError)
            assert isinstance(error, Exception)

    def test_error_exception_chaining(self) -> None:
        """Test exception chaining with 'raise from' pattern."""
        original_error = ValueError("Original error")

        try:
            raise AuthProviderError("Provider error") from original_error
        except AuthProviderError as e:
            assert e.__cause__ is original_error
            assert str(e) == "Provider error"

    def test_port_bind_error_with_errno(self) -> None:
        """Test PortBindError with errno context."""
        import errno

        original_os_error = OSError("Address already in use")
        original_os_error.errno = errno.EADDRINUSE

        try:
            raise PortBindError("Port 8080 unavailable") from original_os_error
        except PortBindError as e:
            assert e.__cause__ is original_os_error
            assert e.__cause__.errno == errno.EADDRINUSE
