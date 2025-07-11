"""Configuration module for Claude Proxy API Server."""

from .auth import AuthSettings, CredentialStorageSettings, OAuthSettings
from .docker_settings import DockerSettings
from .loader import ConfigLoader, load_config
from .reverse_proxy import ReverseProxySettings
from .settings import Settings, get_settings
from .validators import (
    ConfigValidationError,
    validate_config_dict,
    validate_cors_origins,
    validate_host,
    validate_log_level,
    validate_path,
    validate_port,
    validate_timeout,
    validate_url,
)


__all__ = [
    "Settings",
    "get_settings",
    "AuthSettings",
    "OAuthSettings",
    "CredentialStorageSettings",
    "ReverseProxySettings",
    "DockerSettings",
    "ConfigLoader",
    "load_config",
    "ConfigValidationError",
    "validate_config_dict",
    "validate_cors_origins",
    "validate_host",
    "validate_log_level",
    "validate_path",
    "validate_port",
    "validate_timeout",
    "validate_url",
]
