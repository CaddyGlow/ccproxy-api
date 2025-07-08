"""Custom exceptions for credential handling."""


class CredentialsError(Exception):
    """Base exception for all credential-related errors."""

    pass


class CredentialsNotFoundError(CredentialsError):
    """Raised when credentials cannot be found in any configured location."""

    pass


class CredentialsInvalidError(CredentialsError):
    """Raised when credentials are found but invalid or corrupted."""

    pass


class CredentialsExpiredError(CredentialsError):
    """Raised when credentials have expired."""

    pass


class CredentialsStorageError(CredentialsError):
    """Raised when there's an error reading or writing credentials."""

    pass


class OAuthError(CredentialsError):
    """Base exception for OAuth-related errors."""

    pass


class OAuthLoginError(OAuthError):
    """Raised when OAuth login fails."""

    pass


class OAuthTokenRefreshError(OAuthError):
    """Raised when token refresh fails."""

    pass


class OAuthCallbackError(OAuthError):
    """Raised when OAuth callback processing fails."""

    pass
