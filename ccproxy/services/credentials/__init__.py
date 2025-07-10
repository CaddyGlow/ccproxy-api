"""Credentials management package."""

from ccproxy.auth.exceptions import (
    CredentialsError,
    CredentialsExpiredError,
    CredentialsInvalidError,
    CredentialsNotFoundError,
    CredentialsStorageError,
    OAuthCallbackError,
    OAuthError,
    OAuthLoginError,
    OAuthTokenRefreshError,
)
from ccproxy.auth.models import (
    AccountInfo,
    ClaudeCredentials,
    OAuthToken,
    OrganizationInfo,
    UserProfile,
)
from ccproxy.auth.storage import JsonFileTokenStorage as JsonFileStorage
from ccproxy.auth.storage import TokenStorage as CredentialsStorageBackend
from ccproxy.services.credentials.config import CredentialsConfig, OAuthConfig
from ccproxy.services.credentials.manager import CredentialsManager
from ccproxy.services.credentials.oauth_client import OAuthClient


__all__ = [
    # Manager
    "CredentialsManager",
    # Config
    "CredentialsConfig",
    "OAuthConfig",
    # Models
    "ClaudeCredentials",
    "OAuthToken",
    "OrganizationInfo",
    "AccountInfo",
    "UserProfile",
    # Storage
    "CredentialsStorageBackend",
    "JsonFileStorage",
    # OAuth
    "OAuthClient",
    # Exceptions
    "CredentialsError",
    "CredentialsNotFoundError",
    "CredentialsInvalidError",
    "CredentialsExpiredError",
    "CredentialsStorageError",
    "OAuthError",
    "OAuthLoginError",
    "OAuthTokenRefreshError",
    "OAuthCallbackError",
]
