"""Test pricing plugin manifest and factory."""

import pytest


def test_pricing_manifest_name_and_config() -> None:
    """Test that pricing plugin has proper manifest configuration."""
    from ccproxy.plugins.pricing.plugin import factory

    manifest = factory.get_manifest()
    assert manifest.name == "pricing"
    assert manifest.version
    assert manifest.config_class is not None


@pytest.mark.unit
def test_factory_creates_runtime() -> None:
    """Test that pricing plugin factory can create runtime."""
    from ccproxy.plugins.pricing.plugin import factory

    runtime = factory.create_runtime()
    assert runtime is not None
    # Runtime is not initialized yet
    assert not runtime.initialized
