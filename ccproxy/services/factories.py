"""Concrete service factory implementations.

This module provides concrete implementations of service factories that
create and configure service instances according to their interfaces.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import structlog

from ccproxy.adapters.openai.adapter import OpenAIAdapter
from ccproxy.config.settings import Settings
from ccproxy.core.http_client import HTTPClientFactory
from ccproxy.services.cli_detection import CLIDetectionService
from ccproxy.services.config import ProxyConfiguration
from ccproxy.services.mocking import MockResponseHandler
from ccproxy.services.streaming import StreamingHandler
from ccproxy.hooks import HookManager
from ccproxy.testing import RealisticMockResponseGenerator


if TYPE_CHECKING:
    from ccproxy.services.container import ServiceContainer

logger = structlog.get_logger(__name__)


class ConcreteServiceFactory:
    """Concrete implementation of service factory."""

    def __init__(self, container: ServiceContainer) -> None:
        """Initialize the service factory."""
        self._container = container

    def register_services(self) -> None:
        """Register all services with the container."""
        self._container.register_service(
            MockResponseHandler, factory=self.create_mock_handler
        )
        self._container.register_service(
            StreamingHandler, factory=self.create_streaming_handler
        )
        self._container.register_service(
            ProxyConfiguration, factory=self.create_proxy_config
        )
        self._container.register_service(
            httpx.AsyncClient, factory=self.create_http_client
        )
        self._container.register_service(
            CLIDetectionService, factory=self.create_cli_detection_service
        )

    def create_mock_handler(self) -> MockResponseHandler:
        """Create mock handler instance."""
        mock_generator = RealisticMockResponseGenerator()
        openai_adapter = OpenAIAdapter()

        handler = MockResponseHandler(
            mock_generator=mock_generator,
            openai_adapter=openai_adapter,
            error_rate=0.05,
            latency_range=(0.5, 2.0),
        )
        return handler

    def create_streaming_handler(self) -> StreamingHandler:
        """Create streaming handler instance."""
        # Try to use HookManager if already registered, otherwise defer
        hook_manager = None
        try:
            hook_manager = self._container.get_service(HookManager)
        except Exception:
            hook_manager = None

        handler = StreamingHandler(hook_manager=hook_manager)
        return handler

    def create_proxy_config(self) -> ProxyConfiguration:
        """Create proxy configuration instance."""
        config = ProxyConfiguration()
        return config

    def create_http_client(self) -> httpx.AsyncClient:
        """Create HTTP client instance."""
        settings = self._container.get_service(Settings)
        client = HTTPClientFactory.create_client(settings=settings)
        logger.debug("http_client_created", category="lifecycle")
        return client

    def create_cli_detection_service(self) -> CLIDetectionService:
        """Create CLI detection service instance."""
        settings = self._container.get_service(Settings)
        return CLIDetectionService(settings)
