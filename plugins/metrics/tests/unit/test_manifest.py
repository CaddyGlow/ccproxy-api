import pytest


def test_metrics_manifest_name_and_config():
    # Import from the filesystem-discovered plugin
    from ccproxy.plugins.metrics.plugin import factory

    manifest = factory.get_manifest()
    assert manifest.name == "metrics"
    assert manifest.version
    assert manifest.config_class is not None


@pytest.mark.unit
def test_factory_creates_runtime():
    from ccproxy.plugins.metrics.plugin import factory

    runtime = factory.create_runtime()
    assert runtime is not None
    # runtime is not initialized yet
    assert not runtime.initialized
