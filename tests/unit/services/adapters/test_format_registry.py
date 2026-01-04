"""Unit tests for format adapter registry.

This module provides tests for the format adapter registry
including manifest registration and requirement validation.
"""

import pytest

from ccproxy.core.plugins import (
    FormatAdapterSpec,
    PluginManifest,
)
from ccproxy.services.adapters.format_adapter import DictFormatAdapter
from ccproxy.services.adapters.format_registry import FormatRegistry


def create_mock_adapter():
    """Create a mock format adapter for testing."""
    return DictFormatAdapter(
        name="test_adapter",
        request=lambda data: {"adapted": "request"},
        response=lambda data: {"adapted": "response"},
    )


class TestFormatRegistry:
    """Tests for format adapter registry."""

    @pytest.fixture
    def registry(self):
        return FormatRegistry()

    @pytest.mark.asyncio
    async def test_manifest_registration_with_feature_flag(self, registry):
        """Test registration from plugin manifest."""

        def adapter_factory():
            return create_mock_adapter()

        spec = FormatAdapterSpec(
            from_format="test_from",
            to_format="test_to",
            adapter_factory=adapter_factory,
            priority=100,
        )

        manifest = PluginManifest(
            name="test_plugin", version="1.0.0", format_adapters=[spec]
        )
        await registry.register_from_manifest(manifest, "test_plugin")

        assert ("test_from", "test_to") in registry._registered_plugins
        assert registry._registered_plugins[("test_from", "test_to")] == "test_plugin"

    @pytest.mark.asyncio
    async def test_conflict_detection_first_wins(self, registry):
        """Test that first registered adapter wins conflicts."""
        # Register two conflicting adapters
        spec1 = FormatAdapterSpec(
            from_format="openai",
            to_format="anthropic",
            adapter_factory=lambda: create_mock_adapter(),
            priority=10,
        )
        spec2 = FormatAdapterSpec(
            from_format="openai",
            to_format="anthropic",
            adapter_factory=lambda: create_mock_adapter(),
            priority=50,
        )

        manifest1 = PluginManifest(
            name="plugin1", version="1.0.0", format_adapters=[spec1]
        )
        manifest2 = PluginManifest(
            name="plugin2", version="1.0.0", format_adapters=[spec2]
        )

        await registry.register_from_manifest(manifest1, "plugin1")
        await registry.register_from_manifest(manifest2, "plugin2")

        # First adapter should be registered
        assert ("openai", "anthropic") in registry._adapters
        assert registry._registered_plugins[("openai", "anthropic")] == "plugin1"

    @pytest.mark.asyncio
    async def test_requirement_validation(self, registry):
        """Test format adapter requirement validation."""
        # Pre-register a core adapter
        core_adapter = create_mock_adapter()
        registry._adapters[("core", "adapter")] = core_adapter

        # Create manifest with requirements
        manifest = PluginManifest(
            name="test_plugin",
            version="1.0.0",
            requires_format_adapters=[
                ("core", "adapter"),  # Available
                ("missing", "adapter"),  # Missing
            ],
        )

        missing = registry.validate_requirements({"test_plugin": manifest})
        assert "test_plugin" in missing
        assert ("missing", "adapter") in missing["test_plugin"]
        assert ("core", "adapter") not in missing["test_plugin"]

    @pytest.mark.asyncio
    async def test_get_adapter_success(self, registry):
        """Test successful adapter retrieval."""
        adapter = create_mock_adapter()
        registry.register(
            from_format="test_from",
            to_format="test_to",
            adapter=adapter,
            plugin_name="test_plugin",
        )

        retrieved = registry.get("test_from", "test_to")
        assert retrieved is adapter

    @pytest.mark.asyncio
    async def test_async_adapter_factory_support(self, registry):
        """Test support for async adapter factories."""

        async def async_factory():
            return create_mock_adapter()

        spec = FormatAdapterSpec(
            from_format="async", to_format="anthropic", adapter_factory=async_factory
        )

        manifest = PluginManifest(
            name="async_plugin", version="1.0.0", format_adapters=[spec]
        )
        await registry.register_from_manifest(manifest, "async_plugin")

        assert ("async", "anthropic") in registry._adapters

    @pytest.mark.asyncio
    async def test_get_adapter_missing(self, registry):
        """Test adapter retrieval when adapter is missing."""
        with pytest.raises(ValueError, match="No adapter registered"):
            registry.get("missing", "adapter")

    @pytest.mark.asyncio
    async def test_get_if_exists_success(self, registry):
        """Test get_if_exists returns adapter when present."""
        adapter = create_mock_adapter()
        registry.register(
            from_format="test_from",
            to_format="test_to",
            adapter=adapter,
            plugin_name="test_plugin",
        )

        retrieved = registry.get_if_exists("test_from", "test_to")
        assert retrieved is adapter

    @pytest.mark.asyncio
    async def test_get_if_exists_missing(self, registry):
        """Test get_if_exists returns None when adapter is missing."""
        result = registry.get_if_exists("missing", "adapter")
        assert result is None

    def test_format_adapter_spec_validation(self):
        """Test FormatAdapterSpec validation."""
        # Test empty format names
        with pytest.raises(ValueError, match="Format names cannot be empty"):
            FormatAdapterSpec(
                from_format="",
                to_format="test",
                adapter_factory=lambda: create_mock_adapter(),
            )

        # Test same format names
        with pytest.raises(
            ValueError, match="from_format and to_format cannot be the same"
        ):
            FormatAdapterSpec(
                from_format="same",
                to_format="same",
                adapter_factory=lambda: create_mock_adapter(),
            )

    def test_format_pair_property(self):
        """Test format_pair property returns correct tuple."""
        spec = FormatAdapterSpec(
            from_format="from_test",
            to_format="to_test",
            adapter_factory=lambda: create_mock_adapter(),
        )
        assert spec.format_pair == ("from_test", "to_test")

    @pytest.mark.asyncio
    async def test_adapter_factory_error_handling(self, registry):
        """Test error handling for failing adapter factories."""

        def failing_factory():
            raise RuntimeError("Factory failed")

        spec = FormatAdapterSpec(
            from_format="openai", to_format="anthropic", adapter_factory=failing_factory
        )

        manifest = PluginManifest(
            name="failing_plugin", version="1.0.0", format_adapters=[spec]
        )

        with pytest.raises(RuntimeError, match="Factory failed"):
            await registry.register_from_manifest(manifest, "failing_plugin")

    @pytest.mark.asyncio
    async def test_multiple_plugins_registration(self, registry):
        """Test registering multiple plugins with different adapters."""
        plugins = {}
        for i in range(3):
            spec = FormatAdapterSpec(
                from_format=f"from_{i}",
                to_format=f"to_{i}",
                adapter_factory=lambda: create_mock_adapter(),
                priority=i * 10,
            )
            plugins[f"plugin_{i}"] = PluginManifest(
                name=f"plugin_{i}", version="1.0.0", format_adapters=[spec]
            )

        # Register all plugins
        for name, manifest in plugins.items():
            await registry.register_from_manifest(manifest, name)

        # Validate all are registered
        for name in plugins:
            assert name in registry.get_registered_plugins()

        # Check all adapters are available
        assert len(registry._adapters) == 3

    def test_plugin_manifest_validation(self):
        """Test PluginManifest format adapter requirement validation."""
        manifest = PluginManifest(
            name="test",
            version="1.0.0",
            requires_format_adapters=[("req1", "req2"), ("req3", "req4")],
        )

        available = {("req1", "req2"), ("req5", "req6")}
        missing = manifest.validate_format_adapter_requirements(available)

        assert ("req3", "req4") in missing
        assert ("req1", "req2") not in missing

    def test_core_adapter_registration(self, registry):
        """Test that core adapters can be registered."""
        adapter = create_mock_adapter()
        registry.register(
            from_format="anthropic.messages",
            to_format="openai.responses",
            adapter=adapter,
            plugin_name="core",
        )

        assert ("anthropic.messages", "openai.responses") in registry._adapters
        assert (
            registry._registered_plugins[("anthropic.messages", "openai.responses")]
            == "core"
        )

    def test_list_pairs(self, registry):
        """Test format pair listing."""
        adapter = create_mock_adapter()
        registry.register(
            from_format="from1",
            to_format="to1",
            adapter=adapter,
            plugin_name="test",
        )

        pairs = registry.list_pairs()
        assert "from1->to1" in pairs
