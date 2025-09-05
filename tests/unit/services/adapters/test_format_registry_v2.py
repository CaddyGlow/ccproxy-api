"""Enhanced unit tests for format adapter registry v2.

This module provides comprehensive tests for the format adapter registry
including manifest registration, conflict resolution, and feature flag control.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from ccproxy.services.adapters.format_registry import FormatAdapterRegistry
from ccproxy.core.plugins.declaration import FormatAdapterSpec, PluginManifest, FormatPair
from ccproxy.adapters.base import APIAdapter


class MockAPIAdapter(APIAdapter):
    """Mock API adapter for testing."""

    async def adapt_request(self, request_data):
        return {"adapted": "request"}

    async def adapt_response(self, response_data):
        return {"adapted": "response"}

    async def adapt_stream(self, stream):
        async def mock_stream():
            yield {"adapted": "stream"}
        return mock_stream()


class TestFormatAdapterRegistryV2:
    """Enhanced tests for format adapter registry v2."""

    @pytest.fixture
    def registry(self):
        return FormatAdapterRegistry(conflict_mode="fail_fast")

    @pytest.fixture
    def priority_registry(self):
        return FormatAdapterRegistry(conflict_mode="priority")

    @pytest.mark.asyncio
    async def test_manifest_registration_with_feature_flag(self, registry):
        """Test registration from plugin manifest."""
        adapter_factory = lambda: MockAPIAdapter()
        spec = FormatAdapterSpec(
            from_format="test_from",
            to_format="test_to",
            adapter_factory=adapter_factory,
            priority=100
        )

        manifest = PluginManifest(
            name="test_plugin",
            version="1.0.0",
            format_adapters=[spec]
        )
        await registry.register_from_manifest(manifest, "test_plugin")

        assert "test_plugin" in registry._registered_plugins
        assert ("test_from", "test_to") in registry._adapter_specs

    @pytest.mark.asyncio
    async def test_conflict_resolution_priority_mode(self, priority_registry):
        """Test priority-based conflict resolution."""
        # Register conflicting adapters with different priorities
        high_priority_spec = FormatAdapterSpec(
            from_format="openai", to_format="anthropic",
            adapter_factory=lambda: MockAPIAdapter(),
            priority=10  # Higher priority (lower number)
        )
        low_priority_spec = FormatAdapterSpec(
            from_format="openai", to_format="anthropic",
            adapter_factory=lambda: MockAPIAdapter(),
            priority=50  # Lower priority
        )

        manifest1 = PluginManifest(
            name="plugin1", version="1.0.0", format_adapters=[high_priority_spec]
        )
        manifest2 = PluginManifest(
            name="plugin2", version="1.0.0", format_adapters=[low_priority_spec]
        )

        await priority_registry.register_from_manifest(manifest1, "plugin1")
        await priority_registry.register_from_manifest(manifest2, "plugin2")

        await priority_registry.resolve_conflicts_and_finalize()

        # High priority adapter should win
        assert priority_registry._adapter_specs[("openai", "anthropic")] == high_priority_spec

    @pytest.mark.asyncio
    async def test_requirement_validation(self, registry):
        """Test format adapter requirement validation."""
        # Pre-register a core adapter
        core_adapter = MockAPIAdapter()
        registry._adapters[("core", "adapter")] = core_adapter

        # Create manifest with requirements
        manifest = PluginManifest(
            name="test_plugin",
            version="1.0.0",
            requires_format_adapters=[
                ("core", "adapter"),  # Available
                ("missing", "adapter"),  # Missing
            ]
        )

        missing = registry.validate_requirements({"test_plugin": manifest})
        assert "test_plugin" in missing
        assert ("missing", "adapter") in missing["test_plugin"]
        assert ("core", "adapter") not in missing["test_plugin"]

    @pytest.mark.asyncio
    async def test_finalization_prevents_modifications(self, registry):
        """Test registry cannot be modified after finalization."""
        await registry.resolve_conflicts_and_finalize()

        manifest = PluginManifest(
            name="late_plugin",
            version="1.0.0",
            format_adapters=[
                FormatAdapterSpec(
                    from_format="openai", to_format="anthropic",
                    adapter_factory=lambda: MockAPIAdapter()
                )
            ]
        )

        # Should be ignored with warning log
        await registry.register_from_manifest(manifest, "late_plugin")
        assert "late_plugin" not in registry._registered_plugins

    @pytest.mark.asyncio
    async def test_async_adapter_factory_support(self, registry):
        """Test support for async adapter factories."""
        async def async_factory():
            return MockAPIAdapter()

        spec = FormatAdapterSpec(
            from_format="async", to_format="anthropic",
            adapter_factory=async_factory
        )

        manifest = PluginManifest(
            name="async_plugin", version="1.0.0", format_adapters=[spec]
        )
        await registry.register_from_manifest(manifest, "async_plugin")
        await registry.resolve_conflicts_and_finalize()

        assert ("async", "anthropic") in registry._adapters

    @pytest.mark.asyncio
    async def test_conflict_detection_behavior(self, priority_registry):
        """Test that conflicts are properly detected and resolved."""
        spec1 = FormatAdapterSpec(
            from_format="openai", to_format="anthropic",
            adapter_factory=lambda: MockAPIAdapter(),
            priority=10
        )
        spec2 = FormatAdapterSpec(
            from_format="openai", to_format="anthropic",
            adapter_factory=lambda: MockAPIAdapter(),
            priority=20
        )

        manifest1 = PluginManifest(
            name="plugin1", version="1.0.0", format_adapters=[spec1]
        )
        manifest2 = PluginManifest(
            name="plugin2", version="1.0.0", format_adapters=[spec2]
        )

        await priority_registry.register_from_manifest(manifest1, "plugin1")
        await priority_registry.register_from_manifest(manifest2, "plugin2")

        # Should detect conflict and resolve with priority
        assert ("openai", "anthropic") in priority_registry._conflicts
        
        await priority_registry.resolve_conflicts_and_finalize()
        
        # Higher priority should win
        winning_spec = priority_registry._adapter_specs[("openai", "anthropic")]
        assert winning_spec.priority == 10

    @pytest.mark.asyncio
    async def test_fail_fast_mode_raises_on_conflicts(self, registry):
        """Test fail fast mode raises ValueError on conflicts."""
        spec1 = FormatAdapterSpec(
            from_format="openai", to_format="anthropic",
            adapter_factory=lambda: MockAPIAdapter()
        )
        spec2 = FormatAdapterSpec(
            from_format="openai", to_format="anthropic",
            adapter_factory=lambda: MockAPIAdapter()
        )

        manifest1 = PluginManifest(
            name="plugin1", version="1.0.0", format_adapters=[spec1]
        )
        manifest2 = PluginManifest(
            name="plugin2", version="1.0.0", format_adapters=[spec2]
        )

        await registry.register_from_manifest(manifest1, "plugin1")
        await registry.register_from_manifest(manifest2, "plugin2")

        with pytest.raises(ValueError, match="Format adapter conflicts detected"):
            await registry.resolve_conflicts_and_finalize()

    def test_format_adapter_spec_validation(self):
        """Test FormatAdapterSpec validation."""
        # Test empty format names
        with pytest.raises(ValueError, match="Format names cannot be empty"):
            FormatAdapterSpec(
                from_format="",
                to_format="test",
                adapter_factory=lambda: MockAPIAdapter()
            )

        # Test same format names
        with pytest.raises(ValueError, match="from_format and to_format cannot be the same"):
            FormatAdapterSpec(
                from_format="same",
                to_format="same",
                adapter_factory=lambda: MockAPIAdapter()
            )

    def test_format_pair_property(self):
        """Test format_pair property returns correct tuple."""
        spec = FormatAdapterSpec(
            from_format="from_test",
            to_format="to_test",
            adapter_factory=lambda: MockAPIAdapter()
        )
        assert spec.format_pair == ("from_test", "to_test")

    @pytest.mark.asyncio
    async def test_adapter_factory_error_handling(self, registry):
        """Test error handling for failing adapter factories."""
        def failing_factory():
            raise RuntimeError("Factory failed")

        spec = FormatAdapterSpec(
            from_format="openai", to_format="anthropic",
            adapter_factory=failing_factory
        )

        manifest = PluginManifest(
            name="failing_plugin", version="1.0.0", format_adapters=[spec]
        )
        await registry.register_from_manifest(manifest, "failing_plugin")

        with pytest.raises(ValueError, match="Failed to instantiate adapter"):
            await registry.resolve_conflicts_and_finalize()

    @pytest.mark.asyncio
    async def test_multiple_plugins_registration(self, registry):
        """Test registering multiple plugins with different adapters."""
        plugins = {}
        for i in range(3):
            spec = FormatAdapterSpec(
                from_format=f"from_{i}",
                to_format=f"to_{i}",
                adapter_factory=lambda: MockAPIAdapter(),
                priority=i * 10
            )
            plugins[f"plugin_{i}"] = PluginManifest(
                name=f"plugin_{i}",
                version="1.0.0",
                format_adapters=[spec]
            )

        # Register all plugins
        for name, manifest in plugins.items():
            await registry.register_from_manifest(manifest, name)

        # Validate all are registered
        for name in plugins.keys():
            assert name in registry._registered_plugins

        # Finalize and check all adapters are available
        await registry.resolve_conflicts_and_finalize()
        assert len(registry._adapters) == 3

    def test_plugin_manifest_validation(self):
        """Test PluginManifest format adapter requirement validation."""
        manifest = PluginManifest(
            name="test", 
            version="1.0.0",
            requires_format_adapters=[("req1", "req2"), ("req3", "req4")]
        )
        
        available = {("req1", "req2"), ("req5", "req6")}
        missing = manifest.validate_format_adapter_requirements(available)
        
        assert ("req3", "req4") in missing
        assert ("req1", "req2") not in missing