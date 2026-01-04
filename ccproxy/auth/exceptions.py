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


class CredentialsError(AuthenticationError):
    """Base credentials error."""

    pass


class CredentialsNotFoundError(CredentialsError):
    """Credentials not found error."""

    pass


class CredentialsExpiredError(CredentialsError):
    """Credentials expired error."""

    pass


class CredentialsInvalidError(CredentialsError):
    """Credentials are invalid or malformed."""

    pass


class CredentialsStorageError(CredentialsError):
    """Error occurred during credentials storage operations."""

    pass


class OAuthError(AuthenticationError):
    """Base OAuth error."""

    pass


class OAuthTokenRefreshError(OAuthError):
    """OAuth token refresh failed."""

    pass
