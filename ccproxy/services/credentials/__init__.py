"""Credentials management package."""

from ccproxy.services.credentials.config import CredentialsConfig, OAuthConfig
from ccproxy.services.credentials.exceptions import (
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
from ccproxy.services.credentials.json_storage import JsonFileStorage
from ccproxy.services.credentials.manager import CredentialsManager
from ccproxy.services.credentials.models import (
    AccountInfo,
    ClaudeCredentials,
    OAuthToken,
    OrganizationInfo,
    UserProfile,
)
from ccproxy.services.credentials.oauth_client import OAuthClient
from ccproxy.services.credentials.storage import CredentialsStorageBackend


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
