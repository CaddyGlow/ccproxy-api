"""Error taxonomy for CLI authentication flows."""


class AuthError(Exception):
    """Base class for authentication errors."""

    pass


class AuthTimedOutError(AuthError):
    """Authentication process timed out."""

    pass


class AuthUserAbortedError(AuthError):
    """User cancelled authentication."""

    pass


class AuthProviderError(AuthError):
    """Provider-specific authentication error."""

    pass


class NetworkError(AuthError):
    """Network connectivity error."""

    pass


class PortBindError(AuthError):
    """Failed to bind to required port."""

    pass
