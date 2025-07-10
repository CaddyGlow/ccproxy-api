"""Authentication exceptions."""


class AuthenticationError(Exception):
    """Base authentication error."""

    pass


class AuthenticationRequiredError(AuthenticationError):
    """Authentication is required but not provided."""

    pass


class InvalidTokenError(AuthenticationError):
    """Invalid or expired token."""

    pass


class InsufficientPermissionsError(AuthenticationError):
    """Insufficient permissions for the requested operation."""

    pass
