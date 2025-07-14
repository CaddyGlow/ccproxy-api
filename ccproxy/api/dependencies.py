"""Shared dependencies for Claude Code Proxy API Server."""

from typing import Annotated

from fastapi import Depends

from ccproxy.auth.dependencies import AuthManagerDep, get_auth_manager
from ccproxy.config.settings import Settings, get_settings
from ccproxy.core.http import BaseProxyClient
from ccproxy.core.logging import get_logger
from ccproxy.observability import PrometheusMetrics, get_metrics
from ccproxy.services.claude_sdk_service import ClaudeSDKService
from ccproxy.services.credentials.manager import CredentialsManager
from ccproxy.services.proxy_service import ProxyService


logger = get_logger(__name__)

# Type aliases for dependency injection
SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_claude_service(
    auth_manager: AuthManagerDep,
) -> ClaudeSDKService:
    """Get Claude SDK service instance.

    Args:
        auth_manager: Authentication manager dependency

    Returns:
        Claude SDK service instance
    """
    logger.debug("Creating Claude SDK service instance")
    # Get global metrics instance
    metrics = get_metrics()

    return ClaudeSDKService(
        auth_manager=auth_manager,
        metrics=metrics,
    )


def get_credentials_manager(
    settings: SettingsDep,
) -> CredentialsManager:
    """Get credentials manager instance.

    Args:
        settings: Application settings dependency

    Returns:
        Credentials manager instance
    """
    logger.debug("Creating credentials manager instance")
    return CredentialsManager(config=settings.auth)


def get_proxy_service(
    settings: SettingsDep,
    credentials_manager: Annotated[
        CredentialsManager, Depends(get_credentials_manager)
    ],
) -> ProxyService:
    """Get proxy service instance.

    Args:
        settings: Application settings dependency
        credentials_manager: Credentials manager dependency

    Returns:
        Proxy service instance
    """
    logger.debug("Creating proxy service instance")
    # Create HTTP client for proxy
    from ccproxy.core.http import HTTPXClient

    http_client = HTTPXClient()
    proxy_client = BaseProxyClient(http_client)

    # Get global metrics instance
    metrics = get_metrics()

    return ProxyService(
        proxy_client=proxy_client,
        credentials_manager=credentials_manager,
        proxy_mode="full",
        target_base_url=settings.reverse_proxy.target_url,
        metrics=metrics,
    )


def get_observability_metrics() -> PrometheusMetrics:
    """Get observability metrics instance.

    Returns:
        PrometheusMetrics instance
    """
    logger.debug("Getting observability metrics instance")
    return get_metrics()


# Type aliases for service dependencies
ClaudeServiceDep = Annotated[ClaudeSDKService, Depends(get_claude_service)]
ProxyServiceDep = Annotated[ProxyService, Depends(get_proxy_service)]
ObservabilityMetricsDep = Annotated[
    PrometheusMetrics, Depends(get_observability_metrics)
]
