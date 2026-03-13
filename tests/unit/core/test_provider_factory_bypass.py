"""Unit tests for provider factory bypass mode behaviour."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ccproxy.core.plugins import factories as plugin_factories
from ccproxy.plugins.codex.plugin import CodexFactory
from ccproxy.services.adapters.mock_adapter import MockAdapter


@pytest.mark.asyncio
async def test_create_adapter_logs_warning_in_bypass_mode() -> None:
    factory = CodexFactory()
    mock_handler = MagicMock()
    service_container = MagicMock()
    service_container.get_mock_handler.return_value = mock_handler
    context = {
        "settings": SimpleNamespace(server=SimpleNamespace(bypass_mode=True)),
        "service_container": service_container,
    }

    with patch.object(plugin_factories.logger, "warning") as warning:
        adapter = await factory.create_adapter(context)  # type: ignore[arg-type]

    assert isinstance(adapter, MockAdapter)
    assert adapter.mock_handler is mock_handler
    warning.assert_called_once_with(
        "plugin_bypass_mode_enabled",
        plugin="codex",
        adapter="CodexAdapter",
        category="lifecycle",
    )


@pytest.mark.asyncio
async def test_create_adapter_raises_clear_error_without_service_container() -> None:
    factory = CodexFactory()
    context = {
        "settings": SimpleNamespace(server=SimpleNamespace(bypass_mode=True)),
    }

    with pytest.raises(
        RuntimeError,
        match="Cannot initialize plugin 'codex' in bypass mode",
    ):
        await factory.create_adapter(context)  # type: ignore[arg-type]
