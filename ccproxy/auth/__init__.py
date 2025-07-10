"""Authentication module for centralized auth handling."""

from ccproxy.auth.bearer import BearerTokenAuthManager
from ccproxy.auth.credentials_adapter import CredentialsAuthManager
from ccproxy.auth.dependencies import (
    AccessTokenDep,
    AuthManagerDep,
    RequiredAuthDep,
    get_access_token,
    get_auth_manager,
    get_bearer_auth_manager,
    get_credentials_auth_manager,
    require_auth,
)
from ccproxy.auth.exceptions import (
    AuthenticationError,
    AuthenticationRequiredError,
    InsufficientPermissionsError,
    InvalidTokenError,
)
from ccproxy.auth.manager import AuthManager, BaseAuthManager
from ccproxy.auth.storage import (
    JsonFileTokenStorage,
    KeyringTokenStorage,
    TokenStorage,
)


__all__ = [
    # Manager interfaces
    "AuthManager",
    "BaseAuthManager",
    # Implementations
    "BearerTokenAuthManager",
    "CredentialsAuthManager",
    # Storage interfaces and implementations
    "TokenStorage",
    "JsonFileTokenStorage",
    "KeyringTokenStorage",
    # Exceptions
    "AuthenticationError",
    "AuthenticationRequiredError",
    "InvalidTokenError",
    "InsufficientPermissionsError",
    # Dependencies
    "get_auth_manager",
    "get_bearer_auth_manager",
    "get_credentials_auth_manager",
    "require_auth",
    "get_access_token",
    # Type aliases
    "AuthManagerDep",
    "RequiredAuthDep",
    "AccessTokenDep",
]
