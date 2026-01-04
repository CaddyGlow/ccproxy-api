"""Tests for Claude API pricing hook integration."""

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import Mock

import pytest

from ccproxy.core.plugins.hooks import HookContext, HookEvent
from ccproxy.plugins.claude_api.hooks import ClaudeAPIStreamingMetricsHook


@pytest.fixture
def pricing_service() -> Mock:
    """Return a mock pricing service with synchronous calculator."""
    service = Mock()
    service.calculate_cost_sync.return_value = Decimal("0.0105")
    return service


@pytest.mark.unit
class TestClaudeAPIPricingHook:
    """Behavioral tests for the Claude streaming metrics hook."""

    def test_get_pricing_service_direct(self, pricing_service: Mock) -> None:
        """When provided explicitly, the hook should reuse the pricing service."""
        hook = ClaudeAPIStreamingMetricsHook(pricing_service=pricing_service)
        assert hook._get_pricing_service() is pricing_service

    def test_get_pricing_service_lazy(self, pricing_service: Mock) -> None:
        """Hook should resolve the pricing service lazily via the registry."""
        registry = Mock()
        registry.get_service.return_value = pricing_service

        hook = ClaudeAPIStreamingMetricsHook(plugin_registry=registry)
        resolved = hook._get_pricing_service()

        registry.get_service.assert_called_once()
        assert resolved is pricing_service

    def test_get_pricing_service_missing(self) -> None:
        """When no service is available hook should return None."""
        hook = ClaudeAPIStreamingMetricsHook()
        assert hook._get_pricing_service() is None

    @pytest.mark.asyncio
    async def test_cost_calculation_with_pricing(self, pricing_service: Mock) -> None:
        """Hook should compute cost when pricing service is configured."""
        hook = ClaudeAPIStreamingMetricsHook(pricing_service=pricing_service)
        request_id = "req-123"

        # message_start chunk provides model + input tokens
        start_chunk = {
            "type": "message_start",
            "message": {
                "model": "claude-3-5-sonnet-20241022",
                "usage": {
                    "input_tokens": 1000,
                    "output_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
            },
        }
        start_context = HookContext(
            event=HookEvent.PROVIDER_STREAM_CHUNK,
            timestamp=datetime.now(UTC),
            data={"chunk": start_chunk},
            metadata={"request_id": request_id},
            provider="claude_api",
            plugin="claude_api",
        )

        # message_delta chunk provides final output tokens
        delta_chunk = {
            "type": "message_delta",
            "usage": {
                "output_tokens": 500,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        }
        delta_context = HookContext(
            event=HookEvent.PROVIDER_STREAM_CHUNK,
            timestamp=datetime.now(UTC),
            data={"chunk": delta_chunk},
            metadata={"request_id": request_id},
            provider="claude_api",
            plugin="claude_api",
        )

        end_context = HookContext(
            event=HookEvent.PROVIDER_STREAM_END,
            timestamp=datetime.now(UTC),
            data={},
            metadata={"request_id": request_id},
            provider="claude_api",
            plugin="claude_api",
        )

        await hook(start_context)
        await hook(delta_context)
        await hook(end_context)

        pricing_service.calculate_cost_sync.assert_called_once_with(
            model_name="claude-3-5-sonnet-20241022",
            input_tokens=1000,
            output_tokens=500,
            cache_read_tokens=0,
            cache_write_tokens=0,
        )
        assert end_context.data["usage_metrics"]["cost_usd"] == float(Decimal("0.0105"))

    @pytest.mark.asyncio
    async def test_cost_calculation_without_pricing(self) -> None:
        """Hook should skip cost calculation gracefully when service missing."""
        hook = ClaudeAPIStreamingMetricsHook()
        request_id = "req-999"

        start_chunk = {
            "type": "message_start",
            "message": {
                "model": "claude-3-opus",
                "usage": {"input_tokens": 50},
            },
        }
        delta_chunk = {
            "type": "message_delta",
            "usage": {"output_tokens": 25},
        }

        start_context = HookContext(
            event=HookEvent.PROVIDER_STREAM_CHUNK,
            timestamp=datetime.now(UTC),
            data={"chunk": start_chunk},
            metadata={"request_id": request_id},
            provider="claude_api",
            plugin="claude_api",
        )
        delta_context = HookContext(
            event=HookEvent.PROVIDER_STREAM_CHUNK,
            timestamp=datetime.now(UTC),
            data={"chunk": delta_chunk},
            metadata={"request_id": request_id},
            provider="claude_api",
            plugin="claude_api",
        )
        end_context = HookContext(
            event=HookEvent.PROVIDER_STREAM_END,
            timestamp=datetime.now(UTC),
            data={},
            metadata={"request_id": request_id},
            provider="claude_api",
            plugin="claude_api",
        )

        await hook(start_context)
        await hook(delta_context)
        await hook(end_context)

        assert "usage_metrics" in end_context.data
        assert "cost_usd" not in end_context.data["usage_metrics"]
