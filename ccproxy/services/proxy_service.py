"""Simplified ProxyService - provides core services for adapters."""

from typing import TYPE_CHECKING

import httpx
import structlog

from ccproxy.config.settings import Settings
from ccproxy.core.http import BaseProxyClient
from ccproxy.observability.metrics import PrometheusMetrics
from ccproxy.services.config import ProxyConfiguration
from ccproxy.services.streaming import StreamingHandler
from ccproxy.services.tracing import RequestTracer


if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)


class ProxyService:
    """Provides core services and resources for adapters.

    This service manages shared resources like HTTP clients, connection pools,
    and service dependencies that adapters need to function. Individual request
    handling is now done directly by adapters, with hooks handled by HooksMiddleware.
    """

    def __init__(
        self,
        # Core dependencies
        proxy_client: BaseProxyClient,
        settings: Settings,
        # Injected services
        request_tracer: RequestTracer,
        streaming_handler: StreamingHandler,
        config: ProxyConfiguration,
        http_client: httpx.AsyncClient,  # Shared HTTP client for centralized management
        metrics: PrometheusMetrics | None = None,
    ) -> None:
        """Initialize with core services for adapters.

        Only includes services that are actively used by adapters:
        - http_client: Used by BaseHTTPAdapter
        - request_tracer: Used for request tracing
        - streaming_handler: Required for streaming requests
        - config: Used for CORS settings
        - metrics: Used by Claude SDK adapter
        """
        # Core dependencies
        self.proxy_client = proxy_client
        self.settings = settings

        # Injected services (only those actually used)
        self.request_tracer = request_tracer
        self.streaming_handler = streaming_handler
        self.config = config
        self.metrics = metrics

        # Shared HTTP client (injected for centralized management)
        self.http_client = http_client

        logger.debug(
            "ProxyService initialized with core services for adapter use",
            category="lifecycle",
        )


    async def close(self) -> None:
        """Clean up resources on shutdown.

        - Closes proxy client
        - Does NOT close HTTP client (managed by ServiceContainer)
        """
        try:
            # Close proxy client
            if hasattr(self.proxy_client, "close"):
                await self.proxy_client.close()

            logger.info("proxy_service_cleanup_complete", category="lifecycle")

        except (AttributeError, TypeError) as e:
            logger.error(
                "cleanup_attribute_error",
                error=str(e),
                exc_info=e,
                category="lifecycle",
            )
        except Exception as e:
            logger.error(
                "error_during_cleanup", error=str(e), exc_info=e, category="lifecycle"
            )
