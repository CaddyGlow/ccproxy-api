"""Unit tests for ProxyService plugin integration."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccproxy.config.settings import Settings
from ccproxy.services.adapters.base import BaseAdapter
from ccproxy.services.proxy_service import ProxyService


class MockAdapter(BaseAdapter):
    """Mock adapter for testing."""

    async def handle_request(self, request, endpoint, method, **kwargs):
        return MagicMock()

    async def handle_streaming(self, request, endpoint, **kwargs):
        return MagicMock()

    async def cleanup(self) -> None:
        """Cleanup mock adapter resources."""
        pass


@pytest.fixture
def mock_settings():
    """Create mock settings."""
    settings = MagicMock(spec=Settings)
    settings.plugin_dir = "plugins"
    settings.enable_plugins = True
    return settings


@pytest.fixture
def mock_proxy_client():
    """Create mock proxy client."""
    client = MagicMock()
    client.close = AsyncMock()
    return client


@pytest.fixture
def mock_credentials_manager():
    """Create mock credentials manager."""
    manager = MagicMock()
    manager.shutdown = AsyncMock()
    return manager


@pytest.fixture
def proxy_service(mock_proxy_client, mock_credentials_manager, mock_settings):
    """Create ProxyService instance."""
    return ProxyService(
        proxy_client=mock_proxy_client,
        credentials_manager=mock_credentials_manager,
        settings=mock_settings,
    )


@pytest.mark.asyncio
async def test_initialize_plugins(proxy_service):
    """Test plugin initialization."""
    # Mock plugin registry
    mock_plugin = MagicMock()
    mock_adapter = MockAdapter()

    with (
        patch.object(
            proxy_service.plugin_registry, "discover", new_callable=AsyncMock
        ) as mock_discover,
        patch.object(
            proxy_service.plugin_registry, "list_plugins", return_value=["test_plugin"]
        ),
        patch.object(
            proxy_service.plugin_registry, "get_adapter", return_value=mock_adapter
        ),
    ):
        await proxy_service.initialize_plugins()

    # Check that plugins were discovered
    mock_discover.assert_called_once_with(Path("plugins"))

    # Check that plugin adapter was registered
    assert "test_plugin" in proxy_service._plugin_adapters
    assert proxy_service._plugin_adapters["test_plugin"] == mock_adapter
    assert proxy_service._plugins_initialized is True


@pytest.mark.asyncio
async def test_initialize_plugins_already_initialized(proxy_service):
    """Test that plugins are not re-initialized."""
    proxy_service._plugins_initialized = True

    with patch.object(
        proxy_service.plugin_registry, "discover", new_callable=AsyncMock
    ) as mock_discover:
        await proxy_service.initialize_plugins()

    # Discover should not be called
    mock_discover.assert_not_called()


@pytest.mark.asyncio
async def test_initialize_plugins_disabled(proxy_service, mock_settings):
    """Test plugin initialization when disabled."""
    mock_settings.enable_plugins = False

    # Since enable_plugins is checked in the app lifecycle, not in initialize_plugins,
    # we test that the method still works but with an empty plugin dir
    mock_settings.plugin_dir = "/nonexistent"

    with patch("pathlib.Path.exists", return_value=False):
        await proxy_service.initialize_plugins()

    # No plugins should be loaded
    assert len(proxy_service._plugin_adapters) == 0
    assert proxy_service._plugins_initialized is True


@pytest.mark.asyncio
async def test_get_plugin_adapter(proxy_service):
    """Test getting a plugin adapter."""
    mock_adapter = MockAdapter()
    proxy_service._plugin_adapters["test_plugin"] = mock_adapter

    # Get existing adapter
    adapter = proxy_service.get_plugin_adapter("test_plugin")
    assert adapter == mock_adapter

    # Get non-existent adapter
    adapter = proxy_service.get_plugin_adapter("nonexistent")
    assert adapter is None


@pytest.mark.asyncio
async def test_list_active_providers(proxy_service):
    """Test listing active providers."""
    # Add some plugin adapters
    proxy_service._plugin_adapters["openai"] = MockAdapter()
    proxy_service._plugin_adapters["custom"] = MockAdapter()

    providers = proxy_service.list_active_providers()

    # Should include built-in claude and plugin providers
    assert "claude" in providers
    assert "openai" in providers
    assert "custom" in providers
    assert len(providers) == 3


@pytest.mark.asyncio
async def test_proxy_service_close_with_plugins(proxy_service):
    """Test that proxy service properly closes with plugins."""
    # Add a plugin adapter
    proxy_service._plugin_adapters["test"] = MockAdapter()

    await proxy_service.close()

    # Check that proxy client and credentials manager are closed
    proxy_service.proxy_client.close.assert_called_once()
    proxy_service.credentials_manager.shutdown.assert_called_once()


@pytest.mark.asyncio
async def test_plugin_discovery_with_real_path(proxy_service, tmp_path):
    """Test plugin discovery with a real temporary directory."""
    # Create a temporary plugin directory
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()

    # Create a plugin file
    plugin_file = plugin_dir / "test_plugin.py"
    plugin_file.write_text("""
from ccproxy.services.adapters.base import BaseAdapter
from ccproxy.models.provider import ProviderConfig

class TestAdapter(BaseAdapter):
    async def handle_request(self, request, endpoint, method, **kwargs):
        return None

    async def handle_streaming(self, request, endpoint, **kwargs):
        return None

class TestPlugin:
    @property
    def name(self):
        return "test"

    @property
    def version(self):
        return "1.0.0"

    def create_adapter(self):
        return TestAdapter()

    def create_config(self):
        return ProviderConfig(
            name="test",
            base_url="https://test.com",
            supports_streaming=True,
            requires_auth=False,
        )

    async def validate(self):
        return True
""")

    # Update settings to use temp dir
    proxy_service.settings.plugin_dir = str(plugin_dir)

    # Initialize plugins
    await proxy_service.initialize_plugins()

    # Check that plugin was loaded
    assert "test" in proxy_service.plugin_registry.list_plugins()
    assert proxy_service.get_plugin_adapter("test") is not None
