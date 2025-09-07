import pytest


def test_codex_manifest_name_and_config() -> None:
    from ccproxy.plugins.codex.plugin import factory

    manifest = factory.get_manifest()
    assert manifest.name == "codex"
    assert manifest.version
    assert manifest.config_class is not None


@pytest.mark.unit
def test_factory_creates_runtime() -> None:
    from ccproxy.plugins.codex.plugin import factory

    runtime = factory.create_runtime()
    assert runtime is not None
    # Runtime is not initialized yet
    assert not runtime.initialized
