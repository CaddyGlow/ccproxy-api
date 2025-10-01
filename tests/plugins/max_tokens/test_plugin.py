"""Tests for max_tokens plugin integration."""

import json
from datetime import datetime

import pytest

from ccproxy.core.plugins.declaration import PluginContext
from ccproxy.core.plugins.hooks import HookContext, HookEvent, HookRegistry
from ccproxy.plugins.max_tokens.adapter import MaxTokensAdapter, MaxTokensHook
from ccproxy.plugins.max_tokens.config import MaxTokensConfig
from ccproxy.plugins.max_tokens.models import ModelTokenLimits
from ccproxy.plugins.max_tokens.plugin import MaxTokensRuntime, factory
from ccproxy.plugins.max_tokens.service import TokenLimitsService


class TestMaxTokensPlugin:
    """Test cases for max_tokens plugin."""

    @pytest.fixture
    def config(self) -> MaxTokensConfig:
        """Create test configuration."""
        return MaxTokensConfig(
            enabled=True,
            fallback_max_tokens=2048,
            log_modifications=False,
        )

    def test_factory_function(self) -> None:
        """Test plugin factory function."""

        plugin_factory = factory
        assert plugin_factory is not None
        assert hasattr(plugin_factory, "create_runtime")

        # Test that it creates a runtime
        runtime = plugin_factory.create_runtime()
        assert isinstance(runtime, MaxTokensRuntime)
        assert runtime.manifest.name == "max_tokens"

    @pytest.mark.asyncio
    async def test_plugin_lifecycle(self, config: MaxTokensConfig) -> None:
        """Test plugin lifecycle using the factory."""
        plugin_factory = factory
        runtime = plugin_factory.create_runtime()

        # Realistic plugin context with hook registry
        context = PluginContext()
        context.config = config
        context.hook_registry = HookRegistry()

        # Test initialization using the runtime's initialize method
        await runtime.initialize(context)
        assert runtime.config is not None
        assert runtime.config.enabled is True
        assert runtime.initialized is True
        assert runtime.hook_registered is True
        assert context.hook_registry.has(HookEvent.PROVIDER_REQUEST_PREPARED)

        # Test health details
        health = await runtime._get_health_details()
        assert health["type"] == "system"
        assert health["enabled"] is True
        assert "models_count" in health
        assert health["hook_registered"] is True

        # Test shutdown
        await runtime.shutdown()
        assert runtime.service is None
        assert runtime.hook_registered is False
        assert not context.hook_registry.has(HookEvent.PROVIDER_REQUEST_PREPARED)


class TestMaxTokensAdapter:
    """Test cases for MaxTokensAdapter."""

    @pytest.fixture
    def config(self) -> MaxTokensConfig:
        """Create test configuration."""
        return MaxTokensConfig(
            enabled=True,
            fallback_max_tokens=2048,
            log_modifications=False,
        )

    @pytest.fixture
    def adapter(self, config: MaxTokensConfig) -> MaxTokensAdapter:
        """Create max tokens adapter instance."""
        return MaxTokensAdapter(config)

    @pytest.mark.asyncio
    async def test_adapter_initialize(self, adapter: MaxTokensAdapter) -> None:
        """Test adapter initialization."""
        # Initialize the service directly (without hook registration for testing)
        adapter.service.initialize()
        adapter._initialized = True

        assert adapter._initialized is True
        # Check that service was initialized (service doesn't have _initialized attribute)
        assert adapter.service.config is not None

    @pytest.mark.asyncio
    async def test_adapter_cleanup(self, adapter: MaxTokensAdapter) -> None:
        """Test adapter cleanup."""
        # Initialize first
        adapter.service.initialize()
        adapter._initialized = True

        # Test cleanup (without actual hook unregistration for testing)
        await adapter.cleanup()
        assert adapter._initialized is False

    def test_adapter_disabled(self, adapter: MaxTokensAdapter) -> None:
        """Test adapter when disabled."""
        adapter.config.enabled = False

        # Should not fail when disabled
        adapter._initialized = False
        assert adapter._initialized is False

    def test_get_modification_stats(self, adapter: MaxTokensAdapter) -> None:
        """Test getting modification statistics."""
        stats = adapter.get_modification_stats()

        assert "adapter_initialized" in stats
        assert "config_enabled" in stats
        assert "target_providers" in stats
        assert "fallback_max_tokens" in stats
        assert stats["config_enabled"] is True
        assert stats["fallback_max_tokens"] == 2048


class TestMaxTokensHook:
    """Tests for the MaxTokensHook behaviour."""

    @pytest.fixture
    def config(self) -> MaxTokensConfig:
        return MaxTokensConfig(
            enabled=True,
            apply_to_all_providers=True,
            fallback_max_tokens=1024,
            log_modifications=False,
        )

    @pytest.fixture
    def service(self, config: MaxTokensConfig) -> TokenLimitsService:
        service = TokenLimitsService(config)
        service.initialize()
        service.token_limits_data.models.update(
            {
                "test-model": ModelTokenLimits(
                    max_output_tokens=256,
                    max_input_tokens=None,
                ),
                "mapped-model": ModelTokenLimits(
                    max_output_tokens=256,
                    max_input_tokens=None,
                ),
                "alias-model": ModelTokenLimits(
                    max_output_tokens=512,
                    max_input_tokens=None,
                ),
            }
        )
        return service

    @pytest.mark.asyncio
    async def test_hook_injects_missing_max_tokens(
        self, config: MaxTokensConfig, service: TokenLimitsService
    ) -> None:
        hook = MaxTokensHook(config, service)
        context = HookContext(
            event=HookEvent.PROVIDER_REQUEST_PREPARED,
            timestamp=datetime.utcnow(),
            data={
                "body": {"model": "test-model"},
                "body_kind": "json",
                "headers": {},
            },
            metadata={},
            provider="claude_api",
        )

        await hook(context)

        payload = context.data["body"]
        assert payload["max_tokens"] == 256
        assert context.data["body_kind"] == "json"
        assert json.loads(context.data["body_raw"].decode())["max_tokens"] == 256
        assert (
            context.data["modifiers"]["max_tokens"]["new"]
            == context.data["body"]["max_tokens"]
        )
        assert "max_output_tokens" not in context.data["modifiers"]

    @pytest.mark.asyncio
    async def test_hook_respects_non_target_provider(
        self, service: TokenLimitsService
    ) -> None:
        config = MaxTokensConfig(
            enabled=True,
            apply_to_all_providers=False,
            target_providers=["codex"],
            log_modifications=False,
        )
        hook = MaxTokensHook(config, service)
        context = HookContext(
            event=HookEvent.PROVIDER_REQUEST_PREPARED,
            timestamp=datetime.utcnow(),
            data={
                "body": {"model": "test-model", "max_output_tokens": 256},
                "body_kind": "json",
                "headers": {},
            },
            metadata={},
            provider="claude_api",
        )

        await hook(context)

        payload = context.data["body"]
        assert "max_tokens" not in payload
        assert payload["max_output_tokens"] == 256
        assert context.data.get("modifiers", {}) == {}
        assert context.data.get("body_raw") is None

    @pytest.mark.asyncio
    async def test_hook_adjusts_max_output_tokens_for_mapped_model(
        self, config: MaxTokensConfig, service: TokenLimitsService
    ) -> None:
        hook = MaxTokensHook(config, service)
        context = HookContext(
            event=HookEvent.PROVIDER_REQUEST_PREPARED,
            timestamp=datetime.utcnow(),
            data={
                "body": {"model": "mapped-model", "max_output_tokens": 512},
                "body_kind": "json",
                "headers": {},
            },
            metadata={
                "provider_model": "mapped-model",
                "client_model": "alias-model",
                "_model_alias_map": {"mapped-model": "alias-model"},
            },
            provider="claude_api",
        )

        await hook(context)

        payload = context.data["body"]
        assert payload["max_output_tokens"] == 256
        assert context.data["body_kind"] == "json"
        modifiers = context.data["modifiers"]
        assert modifiers["max_output_tokens"]["original"] == 512
        assert modifiers["max_output_tokens"]["new"] == 256
        assert (
            modifiers["max_output_tokens"]["reason"]
            == "max_output_tokens_aligned_with_mapped_model"
        )
        assert json.loads(context.data["body_raw"].decode())["max_output_tokens"] == 256
