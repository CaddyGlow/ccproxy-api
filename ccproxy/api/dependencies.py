"""Shared dependencies for Claude Code Proxy API Server."""

from typing import Annotated

from fastapi import Depends

from ccproxy.auth.dependencies import AuthManagerDep, get_auth_manager
from ccproxy.config.settings import Settings, get_settings
from ccproxy.core.logging import get_logger
from ccproxy.metrics.collector import MetricsCollector
from ccproxy.services.claude_sdk_service import ClaudeSDKService
from ccproxy.services.metrics_service import MetricsService
from ccproxy.services.proxy_service import ProxyService


logger = get_logger(__name__)

# Type aliases for dependency injection
SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_claude_service(
    settings: SettingsDep,
    auth_manager: AuthManagerDep,
) -> ClaudeSDKService:
    """Get Claude SDK service instance.

    Args:
        settings: Application settings dependency
        auth_manager: Authentication manager dependency

    Returns:
        Claude SDK service instance
    """
    logger.debug("Creating Claude SDK service instance")
    return ClaudeSDKService(
        settings=settings,
        auth_manager=auth_manager,
    )


def get_proxy_service(
    settings: SettingsDep,
    auth_manager: AuthManagerDep,
) -> ProxyService:
    """Get proxy service instance.

    Args:
        settings: Application settings dependency
        auth_manager: Authentication manager dependency

    Returns:
        Proxy service instance
    """
    logger.debug("Creating proxy service instance")
    return ProxyService(
        settings=settings,
        auth_manager=auth_manager,
    )


def get_metrics_collector(
    settings: SettingsDep,
) -> MetricsCollector:
    """Get metrics collector instance.

    Args:
        settings: Application settings dependency

    Returns:
        Metrics collector instance
    """
    logger.debug("Creating metrics collector instance")
    return MetricsCollector(settings=settings)


def get_metrics_service(
    metrics_collector: Annotated[MetricsCollector, Depends(get_metrics_collector)],
    settings: SettingsDep,
) -> MetricsService:
    """Get metrics service instance.

    Args:
        metrics_collector: Metrics collector dependency
        settings: Application settings dependency

    Returns:
        Metrics service instance
    """
    logger.debug("Creating metrics service instance")
    return MetricsService(
        collector=metrics_collector,
        settings=settings,
    )


# Type aliases for service dependencies
ClaudeServiceDep = Annotated[ClaudeSDKService, Depends(get_claude_service)]
ProxyServiceDep = Annotated[ProxyService, Depends(get_proxy_service)]
MetricsServiceDep = Annotated[MetricsService, Depends(get_metrics_service)]
MetricsCollectorDep = Annotated[MetricsCollector, Depends(get_metrics_collector)]
