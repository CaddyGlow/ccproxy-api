"""Integration tests for plugin format adapter system v2.

This module tests the end-to-end integration of the format adapter system
with actual plugins and the service container.
"""

import pytest
import os
from unittest.mock import patch, MagicMock

from ccproxy.core.plugins.factory import PluginRegistry
from ccproxy.services.container import ServiceContainer
from ccproxy.config.settings import Settings
from ccproxy.core.plugins.declaration import FormatAdapterSpec, PluginManifest
from ccproxy.adapters.base import APIAdapter


class TestAdapter(APIAdapter):
    """Test adapter for integration tests."""
    
    async def adapt_request(self, request_data):
        return {"test": "request"}
    
    async def adapt_response(self, response_data):
        return {"test": "response"}
    
    async def adapt_stream(self, stream):
        async def test_stream():
            yield {"test": "stream"}
        return test_stream()


@pytest.mark.integration
class TestPluginFormatAdapterIntegrationV2:
    """Integration tests for plugin format adapter system v2."""

    @pytest.fixture
    def settings_with_manifest_enabled(self):
        """Settings with manifest format adapters enabled."""
        settings = Settings()
        settings.features.manifest_format_adapters = True
        settings.features.deprecate_manual_format_setup = True
        return settings

    @pytest.fixture
    def settings_with_manifest_disabled(self):
        """Settings with manifest format adapters disabled."""
        settings = Settings()
        settings.features.manifest_format_adapters = False
        settings.features.deprecate_manual_format_setup = False
        return settings

    @pytest.mark.asyncio
    async def test_feature_flag_controlled_registration(self, settings_with_manifest_enabled):
        """Test format adapters automatically registered when feature flag enabled."""
        container = ServiceContainer(settings_with_manifest_enabled)
        plugin_registry = PluginRegistry()
        
        # Create a mock plugin factory with format adapters
        from ccproxy.core.plugins.base_factory import BaseProviderPluginFactory
        from ccproxy.core.plugins.runtime import ProviderPluginRuntime
        
        class TestPluginRuntime(ProviderPluginRuntime):
            async def _setup_format_registry(self):
                # Should be skipped when manifest system is enabled
                pass
        
        class TestPluginFactory(BaseProviderPluginFactory):
            plugin_name = "test_plugin"
            plugin_description = "Test plugin"
            runtime_class = TestPluginRuntime
            adapter_class = TestAdapter
            config_class = Settings
            
            format_adapters = [
                FormatAdapterSpec(
                    from_format="test_from",
                    to_format="test_to",
                    adapter_factory=lambda: TestAdapter(),
                    priority=50
                )
            ]
        
        # Register the test plugin
        test_factory = TestPluginFactory()
        plugin_registry.register_factory(test_factory)
        
        # Get format registry
        format_registry = container.get_format_registry()
        
        # Simulate plugin initialization
        manifests = plugin_registry.get_all_manifests()
        for name, manifest in manifests.items():
            if manifest.format_adapters:
                await format_registry.register_from_manifest(manifest, name)
        
        # Verify adapter was registered from manifest
        assert ("test_from", "test_to") in format_registry._adapter_specs
        assert "test_plugin" in format_registry._registered_plugins

    @pytest.mark.asyncio
    async def test_dependency_validation_skips_invalid_plugins(self):
        """Test plugins with missing format adapter requirements are skipped."""
        plugin_registry = PluginRegistry()
        
        # Create manifest with missing requirements
        manifest = PluginManifest(
            name="invalid_plugin",
            version="1.0.0",
            requires_format_adapters=[("missing", "adapter")]
        )
        
        # Mock factory for the invalid plugin
        from ccproxy.core.plugins.factory import BasePluginFactory
        from ccproxy.core.plugins.runtime import BasePluginRuntime
        
        class InvalidPluginFactory(BasePluginFactory):
            def __init__(self):
                super().__init__(manifest, BasePluginRuntime)
                
            def create_context(self, core_services):
                from ccproxy.core.plugins.declaration import PluginContext
                return PluginContext()
        
        plugin_registry.register_factory(InvalidPluginFactory())
        
        # Resolve dependencies with feature flag enabled
        with patch.dict(os.environ, {"FEATURES__MANIFEST_FORMAT_ADAPTERS": "true"}):
            order = plugin_registry.resolve_dependencies()
            # Plugin should be skipped due to missing format adapter requirements
            assert "invalid_plugin" not in order

    @pytest.mark.asyncio
    async def test_migration_safety_with_dual_path(self):
        """Test both manual and manifest registration work during transition."""
        # Test with feature flag disabled - manual registration works
        settings_manual = Settings()
        settings_manual.features.manifest_format_adapters = False
        
        container_manual = ServiceContainer(settings_manual)
        format_registry_manual = container_manual.get_format_registry()
        
        # Manual registration should work
        test_adapter = TestAdapter()
        format_registry_manual.register("manual", "test", test_adapter, "manual_plugin")
        assert format_registry_manual.get_adapter_if_exists("manual", "test") is not None
        
        # Test with feature flag enabled - manifest registration works  
        settings_manifest = Settings()
        settings_manifest.features.manifest_format_adapters = True
        
        container_manifest = ServiceContainer(settings_manifest)
        format_registry_manifest = container_manifest.get_format_registry()
        
        # Manifest registration should work
        spec = FormatAdapterSpec(
            from_format="manifest",
            to_format="test",
            adapter_factory=lambda: TestAdapter(),
            priority=50
        )
        manifest = PluginManifest(
            name="manifest_plugin",
            version="1.0.0", 
            format_adapters=[spec]
        )
        
        await format_registry_manifest.register_from_manifest(manifest, "manifest_plugin")
        await format_registry_manifest.resolve_conflicts_and_finalize()
        
        assert format_registry_manifest.get_adapter_if_exists("manifest", "test") is not None

    @pytest.mark.asyncio
    async def test_conflict_resolution_in_real_scenario(self):
        """Test conflict resolution with realistic plugin scenarios."""
        settings = Settings()
        settings.features.manifest_format_adapters = True
        
        container = ServiceContainer(settings)
        format_registry = container.get_format_registry()
        
        # Simulate two plugins registering conflicting adapters
        spec1 = FormatAdapterSpec(
            from_format="openai",
            to_format="anthropic",
            adapter_factory=lambda: TestAdapter(),
            priority=40  # Higher priority
        )
        
        spec2 = FormatAdapterSpec(
            from_format="openai", 
            to_format="anthropic",
            adapter_factory=lambda: TestAdapter(),
            priority=60  # Lower priority
        )
        
        manifest1 = PluginManifest(
            name="claude_sdk", version="1.0.0", format_adapters=[spec1]
        )
        manifest2 = PluginManifest(
            name="claude_api", version="1.0.0", format_adapters=[spec2]
        )
        
        # Register both plugins
        await format_registry.register_from_manifest(manifest1, "claude_sdk")
        await format_registry.register_from_manifest(manifest2, "claude_api")
        
        # Should detect conflict
        assert ("openai", "anthropic") in format_registry._conflicts
        
        # Resolve with priority mode
        await format_registry.resolve_conflicts_and_finalize(enable_priority_mode=True)
        
        # Higher priority (lower number) should win
        winning_spec = format_registry._adapter_specs[("openai", "anthropic")]
        assert winning_spec.priority == 40

    @pytest.mark.asyncio
    async def test_performance_with_many_plugins(self):
        """Test performance with many plugins registering adapters."""
        settings = Settings()
        settings.features.manifest_format_adapters = True
        
        container = ServiceContainer(settings)
        format_registry = container.get_format_registry()
        
        # Create many plugins with adapters
        import time
        start_time = time.time()
        
        for i in range(50):
            spec = FormatAdapterSpec(
                from_format=f"from_{i}",
                to_format=f"to_{i}",
                adapter_factory=lambda: TestAdapter(),
                priority=i
            )
            manifest = PluginManifest(
                name=f"plugin_{i}",
                version="1.0.0",
                format_adapters=[spec]
            )
            await format_registry.register_from_manifest(manifest, f"plugin_{i}")
        
        registration_time = time.time() - start_time
        
        # Finalize all adapters
        start_finalize = time.time()
        await format_registry.resolve_conflicts_and_finalize()
        finalization_time = time.time() - start_finalize
        
        # Performance assertions (should be reasonable)
        assert registration_time < 1.0  # Should register 50 plugins in under 1 second
        assert finalization_time < 2.0  # Should finalize in under 2 seconds
        
        # Verify all adapters are available
        assert len(format_registry._adapters) == 50

    @pytest.mark.asyncio
    async def test_logging_and_monitoring(self, caplog):
        """Test logging output for monitoring and observability."""
        settings = Settings()
        settings.features.manifest_format_adapters = True
        
        container = ServiceContainer(settings)
        format_registry = container.get_format_registry()
        
        # Register a plugin with adapters
        spec = FormatAdapterSpec(
            from_format="monitored",
            to_format="test",
            adapter_factory=lambda: TestAdapter(),
            priority=50
        )
        manifest = PluginManifest(
            name="monitored_plugin",
            version="1.0.0",
            format_adapters=[spec]
        )
        
        await format_registry.register_from_manifest(manifest, "monitored_plugin")
        await format_registry.resolve_conflicts_and_finalize()
        
        # Check for expected log entries with category="format"
        log_messages = [record.message for record in caplog.records]
        
        # Should have finalization log
        assert any("format_adapter_registry_finalized" in msg for msg in log_messages)
        
        # Should have category="format" in structured logs
        format_logs = [
            record for record in caplog.records 
            if hasattr(record, 'category') and record.category == "format"
        ]
        assert len(format_logs) > 0

    def test_feature_flag_configuration(self):
        """Test feature flag configuration works correctly."""
        # Test default values
        settings = Settings()
        assert settings.features.manifest_format_adapters is False
        assert settings.features.deprecate_manual_format_setup is False
        
        # Test environment variable override
        with patch.dict(os.environ, {
            "FEATURES__MANIFEST_FORMAT_ADAPTERS": "true",
            "FEATURES__DEPRECATE_MANUAL_FORMAT_SETUP": "true"
        }):
            settings_env = Settings()
            assert settings_env.features.manifest_format_adapters is True
            assert settings_env.features.deprecate_manual_format_setup is True