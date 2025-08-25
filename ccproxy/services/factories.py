"""Concrete service factory implementations.

This module provides concrete implementations of service factories that
create and configure service instances according to their interfaces.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import structlog

from ccproxy.adapters.openai.adapter import OpenAIAdapter
from ccproxy.core.http_client import HTTPClientFactory
from ccproxy.services.config import ProxyConfiguration
from ccproxy.services.mocking import MockResponseHandler
from ccproxy.services.streaming import StreamingHandler
from ccproxy.services.tracing import CoreRequestTracer
from ccproxy.testing import RealisticMockResponseGenerator
from plugins.pricing.service import PricingService


if TYPE_CHECKING:
    from ccproxy.config.settings import Settings
    from ccproxy.observability.metrics import PrometheusMetrics


logger = structlog.get_logger(__name__)


class ConcreteServiceFactory:
    """Concrete implementation of service factory.

    This factory creates concrete service instances and configures them
    according to their interfaces. It follows the factory pattern to
    centralize service creation logic and ensure consistent configuration.
    """

    def create_request_tracer(self, settings: Settings) -> CoreRequestTracer:
        """Create request tracer instance.

        Args:
            settings: Application settings

        Returns:
            Configured request tracer instance
        """
        tracer = CoreRequestTracer(
            verbose_api=settings.server.verbose_api,
            request_log_dir=settings.server.request_log_dir,
        )
        return tracer

    def create_mock_handler(self, settings: Settings) -> MockResponseHandler:
        """Create mock handler instance.

        Args:
            settings: Application settings

        Returns:
            Configured mock handler instance
        """
        mock_generator = RealisticMockResponseGenerator()
        openai_adapter = OpenAIAdapter()

        handler = MockResponseHandler(
            mock_generator=mock_generator,
            openai_adapter=openai_adapter,
            error_rate=0.05,
            latency_range=(0.5, 2.0),
        )
        return handler

    def create_streaming_handler(
        self,
        settings: Settings,
        metrics: PrometheusMetrics | None = None,
        request_tracer: CoreRequestTracer | None = None,
        pricing_service: PricingService | None = None,
    ) -> StreamingHandler:
        """Create streaming handler instance.

        Args:
            settings: Application settings
            metrics: Optional metrics service
            request_tracer: Optional request tracer
            service_container: Optional service container for pricing service access

        Returns:
            Configured streaming handler instance with header preservation
        """
        handler = StreamingHandler(
            metrics=metrics,
            verbose_streaming=settings.server.verbose_api,
            request_tracer=request_tracer,
            pricing_service=pricing_service,
        )
        return handler

    def create_proxy_config(self) -> ProxyConfiguration:
        """Create proxy configuration instance.

        Returns:
            Configured proxy configuration instance
        """
        config = ProxyConfiguration()
        return config

    def create_http_client(self, settings: Settings) -> httpx.AsyncClient:
        """Create HTTP client instance.

        Args:
            settings: Application settings

        Returns:
            Configured HTTP client instance
        """
        client = HTTPClientFactory.create_shared_client(settings)
        logger.debug("created_shared_http_client", category="lifecycle")
        return client
