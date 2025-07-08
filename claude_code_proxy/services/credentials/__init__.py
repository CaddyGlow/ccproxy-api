"""Credentials management package."""

from claude_code_proxy.services.credentials.config import CredentialsConfig, OAuthConfig
from claude_code_proxy.services.credentials.exceptions import (
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
from claude_code_proxy.services.credentials.json_storage import JsonFileStorage
from claude_code_proxy.services.credentials.manager import CredentialsManager
from claude_code_proxy.services.credentials.models import (
    AccountInfo,
    ClaudeCredentials,
    OAuthToken,
    OrganizationInfo,
    UserProfile,
)
from claude_code_proxy.services.credentials.oauth_client import OAuthClient
from claude_code_proxy.services.credentials.storage import CredentialsStorageBackend


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
