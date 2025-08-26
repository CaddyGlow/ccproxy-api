"""Test pricing service integration with claude_api adapter."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from httpx import AsyncClient

from ccproxy.plugins.factory import PluginRegistry
from plugins.claude_api.adapter import ClaudeAPIAdapter
from plugins.pricing.service import PricingService


@pytest.fixture
def mock_pricing_service():
    """Create a mock pricing service."""
    service = AsyncMock(spec=PricingService)
    service.calculate_cost = AsyncMock(return_value=0.0105)
    return service


@pytest.fixture
def plugin_registry_with_pricing(mock_pricing_service):
    """Create a plugin registry with pricing service."""
    registry = PluginRegistry()

    # Patch the get_service method to return our mock pricing service
    with patch.object(registry, "get_service", return_value=mock_pricing_service):
        yield registry


@pytest.fixture
def adapter_with_pricing(plugin_registry_with_pricing):
    """Create a ClaudeAPIAdapter with pricing service access."""
    context = {
        "plugin_registry": plugin_registry_with_pricing,
        "settings": Mock(),
        "http_client": AsyncClient(),
        "logger": Mock(),
    }

    adapter = ClaudeAPIAdapter(
        proxy_service=None,
        auth_manager=Mock(),
        detection_service=Mock(),
        http_client=AsyncClient(),
        context=context,
    )

    return adapter


class TestClaudeAPIPricingIntegration:
    """Test pricing service integration in claude_api adapter."""

    def test_adapter_stores_context(self, adapter_with_pricing):
        """Test that adapter stores the context passed to it."""
        assert hasattr(adapter_with_pricing, "context")
        assert isinstance(adapter_with_pricing.context, dict)
        assert "plugin_registry" in adapter_with_pricing.context

    def test_get_pricing_service_with_registry(
        self, adapter_with_pricing, mock_pricing_service
    ):
        """Test that adapter can get pricing service through plugin registry."""
        service = adapter_with_pricing._get_pricing_service()

        assert service is not None
        assert service is mock_pricing_service

    def test_get_pricing_service_without_registry(self):
        """Test that adapter returns None when no plugin registry is available."""
        adapter = ClaudeAPIAdapter(
            proxy_service=None,
            auth_manager=Mock(),
            detection_service=Mock(),
            http_client=AsyncClient(),
            context={},  # Empty context, no plugin_registry
        )

        service = adapter._get_pricing_service()
        assert service is None

    def test_get_pricing_service_with_missing_runtime(self):
        """Test graceful handling when pricing service is not available."""
        registry = PluginRegistry()

        context = {"plugin_registry": registry}
        adapter = ClaudeAPIAdapter(
            proxy_service=None,
            auth_manager=Mock(),
            detection_service=Mock(),
            http_client=AsyncClient(),
            context=context,
        )

        # Mock get_service to return None (service not available)
        with patch.object(registry, "get_service", return_value=None):
            service = adapter._get_pricing_service()
            assert service is None

    @pytest.mark.asyncio
    async def test_extract_usage_with_pricing(
        self, adapter_with_pricing, mock_pricing_service
    ):
        """Test that cost calculation uses pricing service when available."""
        import time

        from ccproxy.observability.context import RequestContext

        # Create a mock request context with required arguments
        request_context = RequestContext(
            request_id="test-123", start_time=time.time(), logger=Mock()
        )
        request_context.metadata["model"] = "claude-3-5-sonnet-20241022"

        # Simulate usage data already extracted in processor
        request_context.metadata.update(
            {
                "tokens_input": 1000,
                "tokens_output": 500,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
            }
        )

        # Calculate cost with pricing service
        await adapter_with_pricing._calculate_cost_for_usage(request_context)

        # Verify pricing service was called
        mock_pricing_service.calculate_cost.assert_called_once_with(
            model_name="claude-3-5-sonnet-20241022",
            input_tokens=1000,
            output_tokens=500,
            cache_read_tokens=0,
            cache_write_tokens=0,
        )

        # Verify cost was added to metadata
        assert "cost_usd" in request_context.metadata
        assert request_context.metadata["cost_usd"] == 0.0105

    @pytest.mark.asyncio
    async def test_extract_usage_without_pricing(self):
        """Test that usage extraction works without pricing service."""
        import time

        from ccproxy.observability.context import RequestContext

        # Create adapter without pricing service
        adapter = ClaudeAPIAdapter(
            proxy_service=None,
            auth_manager=Mock(),
            detection_service=Mock(),
            http_client=AsyncClient(),
            context={},  # No plugin_registry
        )

        # Create a mock request context with required arguments
        request_context = RequestContext(
            request_id="test-456", start_time=time.time(), logger=Mock()
        )
        request_context.metadata["model"] = "claude-3-5-sonnet-20241022"

        # Simulate usage data already extracted in processor
        request_context.metadata.update(
            {
                "tokens_input": 1000,
                "tokens_output": 500,
                "tokens_total": 1500,
            }
        )

        # Calculate cost (should not fail even without pricing service)
        await adapter._calculate_cost_for_usage(request_context)

        # Verify tokens are still in metadata
        assert request_context.metadata["tokens_input"] == 1000
        assert request_context.metadata["tokens_output"] == 500
        assert request_context.metadata["tokens_total"] == 1500

        # Cost should not be set when pricing service is not available
        assert "cost_usd" not in request_context.metadata
