"""Integration tests for Copilot plugin lifecycle."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccproxy.config.settings import Settings
from ccproxy.plugins.copilot.plugin import CopilotPluginFactory


@pytest.mark.integration
class TestCopilotPluginLifecycle:
    """Test Copilot plugin lifecycle integration."""

    @pytest.mark.asyncio
    async def test_plugin_factory_initialization(self) -> None:
        """Test plugin factory initialization."""
        factory = CopilotPluginFactory()

        # Test factory properties
        assert factory.plugin_name == "copilot"
        assert (
            factory.plugin_description
            == "GitHub Copilot provider plugin with OAuth authentication"
        )
        assert factory.cli_safe is False
        assert factory.route_prefix == "/copilot"
        assert len(factory.format_adapters) == 2

        # Test format adapters are defined
        adapter_names = [
            (spec.from_format, spec.to_format) for spec in factory.format_adapters
        ]
        assert ("openai", "copilot") in adapter_names
        assert ("copilot", "openai") in adapter_names

    @pytest.mark.asyncio
    async def test_plugin_context_creation(self) -> None:
        """Test plugin context creation with core services."""
        factory = CopilotPluginFactory()

        # Create mock core services
        mock_core_services = MagicMock()
        mock_core_services.get_http_client = MagicMock(return_value=MagicMock())
        mock_core_services.get_hook_manager = MagicMock(return_value=MagicMock())
        mock_core_services.get_cli_detection_service = MagicMock(
            return_value=MagicMock()
        )
        mock_core_services.get_metrics = MagicMock(return_value=MagicMock())

        # Add settings
        settings = Settings()
        mock_core_services.get_settings = MagicMock(return_value=settings)

        # Mock get_plugin_config to return None (no config override)
        mock_core_services.get_plugin_config = MagicMock(return_value=None)

        context = factory.create_context(mock_core_services)

        # Verify context contains expected components
        assert "config" in context
        assert "oauth_provider" in context
        assert "detection_service" in context
        assert "adapter" in context
        assert "router_factory" in context

        # Verify components are properly initialized
        from ccproxy.plugins.copilot.adapter import CopilotAdapter
        from ccproxy.plugins.copilot.config import CopilotConfig
        from ccproxy.plugins.copilot.detection_service import CopilotDetectionService
        from ccproxy.plugins.copilot.oauth.provider import CopilotOAuthProvider

        assert isinstance(context["config"], CopilotConfig)
        assert isinstance(context["oauth_provider"], CopilotOAuthProvider)
        assert isinstance(context["detection_service"], CopilotDetectionService)
        assert isinstance(context["adapter"], CopilotAdapter)

    @pytest.mark.asyncio
    async def test_plugin_runtime_initialization(self) -> None:
        """Test plugin runtime initialization."""
        factory = CopilotPluginFactory()
        manifest = factory.manifest

        runtime = factory.create_runtime()
        runtime.manifest = manifest

        # Create mock adapter with async initialize method
        mock_adapter = MagicMock()
        mock_adapter.initialize = AsyncMock()

        # Create mock detection service with async initialize_detection method
        mock_detection_service = MagicMock()
        mock_detection_service.initialize_detection = AsyncMock()

        # Create mock context
        mock_context = {
            "config": factory.config_class(),
            "oauth_provider": MagicMock(),
            "detection_service": mock_detection_service,
            "adapter": mock_adapter,
            "service_container": MagicMock(),
        }

        runtime.context = mock_context

        # Initialize runtime
        await runtime._on_initialize()

        # Verify initialization
        assert runtime.config is not None
        assert runtime.oauth_provider is not None
        assert runtime.detection_service is not None
        assert runtime.adapter is not None

        # Verify adapter was initialized
        mock_adapter.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_plugin_runtime_cleanup(self) -> None:
        """Test plugin runtime cleanup."""
        factory = CopilotPluginFactory()
        runtime = factory.create_runtime()

        # Create mock components
        mock_adapter = MagicMock()
        mock_adapter.cleanup = AsyncMock()
        mock_oauth_provider = MagicMock()
        mock_oauth_provider.cleanup = AsyncMock()

        runtime.adapter = mock_adapter
        runtime.oauth_provider = mock_oauth_provider

        # Test cleanup
        await runtime.cleanup()

        # Verify cleanup was called
        mock_adapter.cleanup.assert_called_once()
        mock_oauth_provider.cleanup.assert_called_once()

        # Verify components are cleared
        assert runtime.adapter is None
        assert runtime.oauth_provider is None

    @pytest.mark.asyncio
    async def test_plugin_runtime_cleanup_with_errors(self) -> None:
        """Test plugin runtime cleanup handles errors gracefully."""
        factory = CopilotPluginFactory()
        runtime = factory.create_runtime()

        # Create mock components that raise errors
        mock_adapter = MagicMock()
        mock_adapter.cleanup = AsyncMock(side_effect=Exception("Adapter cleanup error"))
        mock_oauth_provider = MagicMock()
        mock_oauth_provider.cleanup = AsyncMock(
            side_effect=Exception("OAuth cleanup error")
        )

        runtime.adapter = mock_adapter
        runtime.oauth_provider = mock_oauth_provider

        # Should not raise exception
        await runtime.cleanup()

        # Verify cleanup was attempted
        mock_adapter.cleanup.assert_called_once()
        mock_oauth_provider.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_format_registry_setup_legacy(self) -> None:
        """Test legacy format registry setup."""
        factory = CopilotPluginFactory()
        runtime = factory.create_runtime()

        # Create mock service container and format registry
        mock_registry = MagicMock()
        mock_service_container = MagicMock()
        mock_service_container.get_format_registry.return_value = mock_registry

        mock_context = {
            "service_container": mock_service_container,
        }
        runtime.context = mock_context

        # Mock settings to use legacy setup
        with patch("ccproxy.config.Settings") as mock_settings_class:
            mock_settings = MagicMock()
            mock_settings_class.return_value = mock_settings

            await runtime._setup_format_registry()

            # Verify format adapters were registered
            assert mock_registry.register.call_count == 2

            # Check that both adapters were registered
            calls = mock_registry.register.call_args_list
            registered_pairs = [(call[0][0], call[0][1]) for call in calls]
            assert ("openai", "copilot") in registered_pairs
            assert ("copilot", "openai") in registered_pairs

    @pytest.mark.asyncio
    async def test_oauth_provider_creation(self) -> None:
        """Test OAuth provider creation with proper dependencies."""
        factory = CopilotPluginFactory()

        # Create mock context with dependencies
        mock_context = {
            "http_client": MagicMock(),
            "hook_manager": MagicMock(),
            "cli_detection_service": MagicMock(),
        }

        oauth_provider = factory.create_oauth_provider(mock_context)

        assert oauth_provider is not None
        assert oauth_provider.http_client is mock_context["http_client"]
        assert oauth_provider.hook_manager is mock_context["hook_manager"]
        assert oauth_provider.detection_service is mock_context["cli_detection_service"]

    @pytest.mark.asyncio
    async def test_detection_service_creation(self) -> None:
        """Test detection service creation with proper dependencies."""
        factory = CopilotPluginFactory()

        # Create mock context with required services
        mock_settings = MagicMock()
        mock_cli_service = MagicMock()

        mock_context = {
            "settings": mock_settings,
            "cli_detection_service": mock_cli_service,
        }

        detection_service = factory.create_detection_service(mock_context)

        assert detection_service is not None
        # Would need to check internal state, but this verifies creation doesn't fail

    @pytest.mark.asyncio
    async def test_detection_service_creation_requires_context(self) -> None:
        """Test detection service creation requires context."""
        factory = CopilotPluginFactory()

        with pytest.raises(ValueError, match="Context required for detection service"):
            factory.create_detection_service(None)

    @pytest.mark.asyncio
    async def test_detection_service_creation_requires_dependencies(self) -> None:
        """Test detection service creation requires dependencies."""
        factory = CopilotPluginFactory()

        # Test with None context
        with pytest.raises(ValueError, match=r"Context required for detection service"):
            factory.create_detection_service(None)

        # Test with context missing required services
        mock_context = {
            "some_other_key": "value"
        }  # Non-empty but missing required keys
        with pytest.raises(
            ValueError, match=r"Settings and CLI detection service required"
        ):
            factory.create_detection_service(mock_context)

    @pytest.mark.asyncio
    async def test_adapter_creation(self) -> None:
        """Test main adapter creation with dependencies."""
        factory = CopilotPluginFactory()
        from ccproxy.plugins.copilot.config import CopilotConfig

        # Create mock context with dependencies
        mock_config = CopilotConfig()
        mock_oauth_provider = MagicMock()
        mock_detection_service = MagicMock()
        mock_metrics = MagicMock()
        mock_hook_manager = MagicMock()
        mock_http_client = MagicMock()

        mock_context = {
            "config": mock_config,
            "oauth_provider": mock_oauth_provider,
            "detection_service": mock_detection_service,
            "metrics": mock_metrics,
            "hook_manager": mock_hook_manager,
            "http_client": mock_http_client,
        }

        adapter = factory.create_adapter(mock_context)

        assert adapter is not None
        # Verify adapter was created with proper dependencies
        assert adapter.config is mock_config
        assert adapter.oauth_provider is mock_oauth_provider
        assert adapter.detection_service is mock_detection_service
        assert adapter.metrics is mock_metrics
        assert adapter.hook_manager is mock_hook_manager
        assert adapter.http_client is mock_http_client

    @pytest.mark.asyncio
    async def test_adapter_creation_requires_context(self) -> None:
        """Test adapter creation requires context."""
        factory = CopilotPluginFactory()

        with pytest.raises(ValueError, match="Context required for adapter"):
            factory.create_adapter(None)

    @pytest.mark.asyncio
    async def test_adapter_creation_with_missing_config(self) -> None:
        """Test adapter creation handles missing config."""
        factory = CopilotPluginFactory()

        # Context without config - should use default
        mock_context = {
            "oauth_provider": MagicMock(),
            "detection_service": MagicMock(),
            "metrics": MagicMock(),
            "hook_manager": MagicMock(),
            "http_client": MagicMock(),
        }

        adapter = factory.create_adapter(mock_context)

        assert adapter is not None
        # Should have created default config
        from ccproxy.plugins.copilot.config import CopilotConfig

        assert isinstance(adapter.config, CopilotConfig)

    @pytest.mark.asyncio
    async def test_router_factory_creation(self) -> None:
        """Test router factory is created in context."""
        factory = CopilotPluginFactory()

        # Create mock core services
        mock_core_services = MagicMock()
        mock_core_services.get_http_client = MagicMock(return_value=MagicMock())
        mock_core_services.get_hook_manager = MagicMock(return_value=MagicMock())
        mock_core_services.get_cli_detection_service = MagicMock(
            return_value=MagicMock()
        )
        mock_core_services.get_metrics = MagicMock(return_value=MagicMock())

        # Add settings
        settings = Settings()
        mock_core_services.get_settings = MagicMock(return_value=settings)

        # Mock get_plugin_config to return None (no config override)
        mock_core_services.get_plugin_config = MagicMock(return_value=None)

        context = factory.create_context(mock_core_services)

        # Verify router factory is present
        assert "router_factory" in context
        assert callable(context["router_factory"])

        # Test calling router factory
        router = context["router_factory"]()
        assert router is not None

    @pytest.mark.asyncio
    async def test_manifest_properties(self) -> None:
        """Test plugin manifest properties."""
        factory = CopilotPluginFactory()
        manifest = factory.manifest

        assert manifest.name == "copilot"
        assert (
            manifest.description
            == "GitHub Copilot provider plugin with OAuth authentication"
        )
        # Note: manifest doesn't have runtime_class attribute, it's on the factory
        assert len(manifest.format_adapters) == 2

        # Verify format adapter specs
        adapter_pairs = [
            (spec.from_format, spec.to_format) for spec in manifest.format_adapters
        ]
        assert ("openai", "copilot") in adapter_pairs
        assert ("copilot", "openai") in adapter_pairs

        # Check priorities
        for spec in manifest.format_adapters:
            assert spec.priority == 30
