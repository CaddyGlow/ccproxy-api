"""Tests for ProxyService functionality.

This module provides working tests for ProxyService that improve test coverage
without breaking the build.
"""

from typing import Any
from unittest.mock import Mock

import pytest

from ccproxy.config.settings import Settings
from ccproxy.services.proxy_service import ProxyService


class TestProxyService:
    """Test the ProxyService implementation."""

    @pytest.mark.unit
    def test_init(self, test_settings: Settings) -> None:
        """Test ProxyService initialization."""
        from ccproxy.metrics.collector import MetricsCollector
        from ccproxy.metrics.storage.memory import InMemoryMetricsStorage
        from ccproxy.services.credentials.manager import CredentialsManager

        credentials_manager = CredentialsManager(test_settings.auth)
        storage = InMemoryMetricsStorage()
        metrics_collector = MetricsCollector(storage=storage)

        # Create a mock proxy client
        mock_proxy_client = Mock()

        proxy_service = ProxyService(
            proxy_client=mock_proxy_client,
            credentials_manager=credentials_manager,
            proxy_mode="direct",
            target_base_url="https://api.anthropic.com",
            metrics_collector=metrics_collector,
        )

        assert proxy_service.proxy_mode == "direct"
        assert proxy_service.target_base_url == "https://api.anthropic.com"
        assert proxy_service.credentials_manager is not None
        assert proxy_service.metrics_collector is not None
        assert proxy_service.request_transformer is not None
        assert proxy_service.response_transformer is not None
        assert proxy_service.openai_adapter is not None

    @pytest.mark.unit
    def test_should_stream_response_false(self, test_settings: Settings) -> None:
        """Test stream detection when streaming is not requested."""
        from ccproxy.metrics.collector import MetricsCollector
        from ccproxy.metrics.storage.memory import InMemoryMetricsStorage
        from ccproxy.services.credentials.manager import CredentialsManager

        credentials_manager = CredentialsManager(test_settings.auth)
        storage = InMemoryMetricsStorage()
        metrics_collector = MetricsCollector(storage=storage)

        # Create a mock proxy client
        mock_proxy_client = Mock()

        proxy_service = ProxyService(
            proxy_client=mock_proxy_client,
            credentials_manager=credentials_manager,
            proxy_mode="direct",
            target_base_url="https://api.anthropic.com",
            metrics_collector=metrics_collector,
        )

        # Test with stream=False
        request_data: dict[str, Any] = {"stream": False, "messages": []}
        assert not proxy_service._should_stream_response(request_data)

        # Test with missing stream key
        request_data = {"messages": []}
        assert not proxy_service._should_stream_response(request_data)
