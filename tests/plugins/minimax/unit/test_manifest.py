"""Manifest and factory tests for the MiniMax provider plugin."""

import pytest


@pytest.mark.minimax
@pytest.mark.unit
def test_minimax_manifest_name_and_config() -> None:
    from ccproxy.plugins.minimax.plugin import factory

    manifest = factory.get_manifest()
    assert manifest.name == "minimax"
    assert manifest.version
    assert manifest.is_provider is True
    assert manifest.config_class is not None


@pytest.mark.minimax
@pytest.mark.unit
def test_minimax_factory_creates_runtime() -> None:
    from ccproxy.plugins.minimax.plugin import factory

    runtime = factory.create_runtime()
    assert runtime is not None
    assert not runtime.initialized


@pytest.mark.minimax
@pytest.mark.unit
def test_minimax_factory_has_no_auth_manager_dependency() -> None:
    """MiniMax uses static API keys, so it must not depend on an OAuth manager."""
    from ccproxy.plugins.minimax.plugin import factory

    assert factory.auth_manager_name is None
    assert factory.get_manifest().dependencies == []
