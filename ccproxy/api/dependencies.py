"""Shared dependencies for CCProxy API Server."""

from typing import Annotated

from fastapi import Depends, Request
from structlog import get_logger

from ccproxy.auth.dependencies import AuthManagerDep
from ccproxy.config.settings import Settings, get_settings
from ccproxy.core.http import BaseProxyClient
from ccproxy.observability import PrometheusMetrics, get_metrics
from ccproxy.observability.storage.duckdb_simple import SimpleDuckDBStorage
from ccproxy.services.claude_sdk_service import ClaudeSDKService
from ccproxy.services.credentials.manager import CredentialsManager
from ccproxy.services.proxy_service import ProxyService


logger = get_logger(__name__)


def get_cached_settings(request: Request) -> Settings:
    """Get cached settings from app state.

    This avoids recomputing settings on every request by using the
    settings instance computed during application startup.

    Args:
        request: FastAPI request object

    Returns:
        Settings instance from app state

    Raises:
        RuntimeError: If settings are not available in app state
    """
    settings = getattr(request.app.state, "settings", None)
    if settings is None:
        # Fallback to get_settings() for safety, but this should not happen
        # in normal operation after lifespan startup
        logger.warning(
            "Settings not found in app state, falling back to get_settings()"
        )
        settings = get_settings()
    return settings


def get_cached_claude_service(request: Request) -> ClaudeSDKService:
    """Get cached ClaudeSDKService from app state.

    This avoids recreating the ClaudeSDKService on every request by using the
    service instance created during application startup.

    Args:
        request: FastAPI request object

    Returns:
        ClaudeSDKService instance from app state

    Raises:
        RuntimeError: If ClaudeSDKService is not available in app state
    """
    claude_service = getattr(request.app.state, "claude_service", None)
    if claude_service is None:
        # Fallback to get_claude_service() for safety, but this should not happen
        # in normal operation after lifespan startup
        logger.warning(
            "ClaudeSDKService not found in app state, falling back to get_claude_service()"
        )
        # Get dependencies manually for fallback
        settings = get_cached_settings(request)
        # Create a simple auth manager for fallback
        from ccproxy.auth.credentials_adapter import CredentialsAuthManager

        auth_manager = CredentialsAuthManager()
        claude_service = get_claude_service(settings, auth_manager)
    return claude_service


# Type aliases for dependency injection
SettingsDep = Annotated[Settings, Depends(get_cached_settings)]


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
    # Get global metrics instance
    metrics = get_metrics()

    # Check if pooling should be enabled from configuration
    use_pool = settings.claude.use_client_pool

    if use_pool:
        logger.info(
            "claude_sdk_pool_enabled",
            message="Using Claude SDK client pooling for improved performance",
            pool_size=settings.claude.pool_settings.pool_size,
            max_pool_size=settings.claude.pool_settings.max_pool_size,
        )

    return ClaudeSDKService(
        auth_manager=auth_manager,
        metrics=metrics,
        settings=settings,
        use_pool=use_pool,
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
        settings=settings,
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


async def get_log_storage(request: Request) -> SimpleDuckDBStorage | None:
    """Get log storage from app state.

    Args:
        request: FastAPI request object

    Returns:
        SimpleDuckDBStorage instance if available, None otherwise
    """
    return getattr(request.app.state, "log_storage", None)


async def get_duckdb_storage(request: Request) -> SimpleDuckDBStorage | None:
    """Get DuckDB storage from app state (backward compatibility).

    Args:
        request: FastAPI request object

    Returns:
        SimpleDuckDBStorage instance if available, None otherwise
    """
    # Try new name first, then fall back to old name for backward compatibility
    storage = getattr(request.app.state, "log_storage", None)
    if storage is None:
        storage = getattr(request.app.state, "duckdb_storage", None)
    return storage


# Type aliases for service dependencies
ClaudeServiceDep = Annotated[ClaudeSDKService, Depends(get_cached_claude_service)]
ProxyServiceDep = Annotated[ProxyService, Depends(get_proxy_service)]
ObservabilityMetricsDep = Annotated[
    PrometheusMetrics, Depends(get_observability_metrics)
]
LogStorageDep = Annotated[SimpleDuckDBStorage | None, Depends(get_log_storage)]
DuckDBStorageDep = Annotated[SimpleDuckDBStorage | None, Depends(get_duckdb_storage)]
