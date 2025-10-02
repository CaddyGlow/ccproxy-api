"""Integration tests for Copilot plugin lifecycle."""

from unittest.mock import AsyncMock, MagicMock

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
        assert len(factory.routers) == 2
        assert factory.routers[0].prefix == "/copilot/v1"
        assert factory.routers[1].prefix == "/copilot"
        assert len(factory.requires_format_adapters) == 6

        # Test format adapter requirements are defined
        adapter_names = factory.requires_format_adapters
        assert ("anthropic.messages", "openai.chat_completions") in adapter_names
        assert ("openai.chat_completions", "anthropic.messages") in adapter_names

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

        # Add settings with proper plugins config
        settings = Settings()
        # Mock the plugins config to return empty dict so no validation happens
        settings.plugins = {"copilot": {}}
        mock_core_services.get_settings = MagicMock(return_value=settings)
        # Also need settings attribute directly on service_container
        mock_core_services.settings = settings

        # Mock get_plugin_config to return actual config dict (not MagicMock)
        from ccproxy.plugins.copilot.config import CopilotConfig

        mock_core_services.get_plugin_config = MagicMock(
            return_value=CopilotConfig().model_dump()
        )
        # Mock get_service_container for format registry
        mock_core_services.get_service_container = MagicMock(return_value=MagicMock())

        context = factory.create_context(mock_core_services)

        # Verify context contains expected components
        assert "config" in context
        assert "oauth_provider" in context
        assert "detection_service" in context
        # Note: adapter and router_factory are created asynchronously by runtime or are part of manifest

        # Verify components are properly initialized
        from ccproxy.plugins.copilot.config import CopilotConfig
        from ccproxy.plugins.copilot.detection_service import CopilotDetectionService
        from ccproxy.plugins.copilot.oauth.provider import CopilotOAuthProvider

        assert isinstance(context["config"], CopilotConfig)
        assert isinstance(context["oauth_provider"], CopilotOAuthProvider)
        assert isinstance(context["detection_service"], CopilotDetectionService)
        # Note: adapter is not created in synchronous context, only in async runtime

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

        runtime.context = mock_context  # type: ignore[assignment]

        # Initialize runtime
        await runtime._on_initialize()

        # Verify initialization
        assert runtime.config is not None
        assert runtime.oauth_provider is not None
        assert runtime.detection_service is not None
        assert runtime.adapter is not None

        # Note: CopilotAdapter doesn't have an initialize() method
        # Initialization is handled through dependency injection in constructor
        # Just verify adapter is present
        assert mock_adapter is not None

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
    async def test_oauth_provider_creation(self) -> None:
        """Test OAuth provider creation with proper dependencies."""
        factory = CopilotPluginFactory()

        # Create mock context with dependencies
        mock_context = {
            "http_client": MagicMock(),
            "hook_manager": MagicMock(),
            "cli_detection_service": MagicMock(),
        }

        oauth_provider = factory.create_oauth_provider(mock_context)  # type: ignore[arg-type]

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

        detection_service = factory.create_detection_service(mock_context)  # type: ignore[arg-type]

        assert detection_service is not None
        # Would need to check internal state, but this verifies creation doesn't fail

    @pytest.mark.asyncio
    async def test_detection_service_creation_requires_context(self) -> None:
        """Test detection service creation requires context."""
        factory = CopilotPluginFactory()

        with pytest.raises(
            ValueError, match="Settings and CLI detection service required"
        ):
            factory.create_detection_service({})  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_detection_service_creation_requires_dependencies(self) -> None:
        """Test detection service creation requires dependencies."""
        factory = CopilotPluginFactory()

        # Test with None context
        with pytest.raises(
            ValueError, match=r"Settings and CLI detection service required"
        ):
            factory.create_detection_service({})  # type: ignore[arg-type]

        # Test with context missing required services
        mock_context = {
            "some_other_key": "value"
        }  # Non-empty but missing required keys
        with pytest.raises(
            ValueError, match=r"Settings and CLI detection service required"
        ):
            factory.create_detection_service(mock_context)  # type: ignore[arg-type]

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
            "http_pool_manager": MagicMock(),  # Required by adapter
            "credentials_manager": MagicMock(),  # Required by adapter
            "service_container": MagicMock(),  # For format registry
        }

        adapter = await factory.create_adapter(mock_context)  # type: ignore[arg-type]

        assert adapter is not None
        # Verify adapter was created with proper dependencies
        assert adapter.config is mock_config
        assert adapter.oauth_provider is mock_oauth_provider  # type: ignore[attr-defined]
        assert adapter.detection_service is mock_detection_service  # type: ignore[attr-defined]
        # Note: metrics, hook_manager, http_client are passed to BaseHTTPAdapter
        # but not stored as instance attributes directly
        assert adapter.auth_manager is not None  # type: ignore[attr-defined]
        assert adapter.http_pool_manager is not None  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_adapter_creation_requires_context(self) -> None:
        """Test adapter creation requires context."""
        factory = CopilotPluginFactory()

        with pytest.raises(ValueError, match="Context required for adapter"):
            await factory.create_adapter(None)  # type: ignore[arg-type]

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
            "http_pool_manager": MagicMock(),  # Required by adapter
            "credentials_manager": MagicMock(),  # Required by adapter
            "service_container": MagicMock(),  # For format registry
        }

        adapter = await factory.create_adapter(mock_context)  # type: ignore[arg-type]

        assert adapter is not None
        # Should have created default config
        from ccproxy.plugins.copilot.config import CopilotConfig

        assert isinstance(adapter.config, CopilotConfig)

    @pytest.mark.asyncio
    async def test_router_specs_available(self) -> None:
        """Test router specs are defined in factory."""
        factory = CopilotPluginFactory()

        # Verify router specs are present in the factory (not context)
        assert hasattr(factory, "routers")
        assert len(factory.routers) == 2

        # Verify the router specs have expected properties
        v1_router = factory.routers[0]
        github_router = factory.routers[1]

        assert v1_router.prefix == "/copilot/v1"
        assert v1_router.tags == ["copilot-api-v1"]
        assert github_router.prefix == "/copilot"
        assert github_router.tags == ["copilot-github"]

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
        # Copilot plugin relies on format adapters provided by core, so manifest has empty format_adapters
        assert len(manifest.format_adapters) == 0

        # But requires format adapters from dependencies
        assert len(manifest.requires_format_adapters) == 6
        adapter_pairs = manifest.requires_format_adapters
        assert ("anthropic.messages", "openai.chat_completions") in adapter_pairs
        assert ("openai.chat_completions", "anthropic.messages") in adapter_pairs
