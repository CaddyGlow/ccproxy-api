"""Authentication module for centralized auth handling."""

from ccproxy.auth.bearer import BearerTokenAuthManager
from ccproxy.auth.dependencies import (
    AccessTokenDep,
    AuthManagerDep,
    RequiredAuthDep,
    get_access_token,
    get_auth_manager,
    require_auth,
)
from ccproxy.auth.exceptions import (
    AuthenticationError,
    AuthenticationRequiredError,
    CredentialsError,
    CredentialsExpiredError,
    CredentialsInvalidError,
    CredentialsNotFoundError,
    CredentialsStorageError,
    InsufficientPermissionsError,
    InvalidTokenError,
    OAuthError,
    OAuthTokenRefreshError,
)
from ccproxy.auth.manager import AuthManager
from ccproxy.auth.storage import (
    JsonFileTokenStorage,
    KeyringTokenStorage,
    TokenStorage,
)


__all__ = [
    # Manager interface
    "AuthManager",
    # Implementations
    "BearerTokenAuthManager",
    # Storage interfaces and implementations
    "TokenStorage",
    "JsonFileTokenStorage",
    "KeyringTokenStorage",
    # Exceptions
    "AuthenticationError",
    "AuthenticationRequiredError",
    "CredentialsError",
    "CredentialsExpiredError",
    "CredentialsInvalidError",
    "CredentialsNotFoundError",
    "CredentialsStorageError",
    "InvalidTokenError",
    "InsufficientPermissionsError",
    "OAuthError",
    "OAuthTokenRefreshError",
    # Dependencies
    "get_auth_manager",
    "require_auth",
    "get_access_token",
    # Type aliases
    "AuthManagerDep",
    "RequiredAuthDep",
    "AccessTokenDep",
]
