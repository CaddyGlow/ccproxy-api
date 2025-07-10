"""Shared dependencies for Claude Code Proxy API Server."""

from typing import Annotated

from fastapi import Depends

from ccproxy.auth.dependencies import AuthManagerDep, get_auth_manager
from ccproxy.config.settings import Settings, get_settings
from ccproxy.core.http import BaseProxyClient
from ccproxy.core.logging import get_logger
from ccproxy.metrics.collector import MetricsCollector
from ccproxy.metrics.storage.memory import InMemoryMetricsStorage
from ccproxy.services.claude_sdk_service import ClaudeSDKService
from ccproxy.services.credentials.manager import CredentialsManager
from ccproxy.services.hybrid_service import HybridService
from ccproxy.services.metrics_service import MetricsService
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
    return ClaudeSDKService(
        auth_manager=auth_manager,
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
    return CredentialsManager(config=settings.credentials)


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
    proxy_client = HTTPXClient()

    return ProxyService(
        proxy_client=proxy_client,
        credentials_manager=credentials_manager,
        proxy_mode="full",
        target_base_url=settings.reverse_proxy_target_url,
    )


def get_metrics_collector() -> MetricsCollector:
    """Get metrics collector instance.

    Returns:
        Metrics collector instance
    """
    logger.debug("Creating metrics collector instance")
    # Use in-memory storage for now
    storage = InMemoryMetricsStorage()
    return MetricsCollector(storage=storage)


def get_metrics_service() -> MetricsService:
    """Get metrics service instance.

    Returns:
        Metrics service instance
    """
    logger.debug("Creating metrics service instance")
    return MetricsService()


def get_hybrid_service(
    claude_service: Annotated[ClaudeSDKService, Depends(get_claude_service)],
    proxy_service: Annotated[ProxyService, Depends(get_proxy_service)],
) -> HybridService:
    """Get hybrid service instance.

    Args:
        claude_service: Claude SDK service dependency
        proxy_service: Proxy service dependency

    Returns:
        Hybrid service instance
    """
    logger.debug("Creating hybrid service instance")
    return HybridService(
        claude_sdk_service=claude_service,
        proxy_service=proxy_service,
    )


# Type aliases for service dependencies
ClaudeServiceDep = Annotated[ClaudeSDKService, Depends(get_claude_service)]
ProxyServiceDep = Annotated[ProxyService, Depends(get_proxy_service)]
HybridServiceDep = Annotated[HybridService, Depends(get_hybrid_service)]
MetricsServiceDep = Annotated[MetricsService, Depends(get_metrics_service)]
MetricsCollectorDep = Annotated[MetricsCollector, Depends(get_metrics_collector)]
