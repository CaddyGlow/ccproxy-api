from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, Field
from typer.testing import CliRunner

from ccproxy.cli.commands import plugins as plugins_cmd
from ccproxy.core.plugins.discovery import PluginFilter


class SampleConfig(BaseModel):
    required_field: str
    optional_field: int = 10
    items: list[str] = Field(default_factory=list, description="Sample list")


class DummyManifest:
    def __init__(self, name: str, config_class: type[BaseModel] | None = None):
        self.name = name
        self.version = "1.0.0"
        self.description = "Just a test plugin"
        self.config_class = config_class


class DummyFactory:
    def __init__(self, manifest: DummyManifest):
        self._manifest = manifest

    def get_manifest(self) -> DummyManifest:
        return self._manifest


@pytest.mark.unit
def test_describe_config_model_includes_values() -> None:
    instance = SampleConfig(required_field="hello", optional_field=42, items=["a", "b"])

    fields = plugins_cmd.describe_config_model(SampleConfig, instance)
    by_name = {field.name: field for field in fields}

    assert by_name["required_field"].required is True
    assert by_name["required_field"].default_label == "required"
    assert by_name["required_field"].value_label == '"hello"'
    assert by_name["optional_field"].default_label == "10"
    assert by_name["optional_field"].value_label == "42"
    assert by_name["items"].default_label.startswith("<factory:")
    assert by_name["items"].value_label == "['a', 'b']"
    assert by_name["items"].description == "Sample list"


@pytest.mark.unit
def test_gather_plugin_metadata_includes_status_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = SimpleNamespace(
        enable_plugins=True,
        plugin_discovery=SimpleNamespace(directories=[]),
        disabled_plugins=["beta"],
        enabled_plugins=["alpha"],
        plugins={
            "alpha": {"required_field": "value", "optional_field": 11},
            "beta": {"required_field": "value"},
        },
        plugins_disable_local_discovery=True,
    )

    factories = {
        "alpha": DummyFactory(DummyManifest("alpha", SampleConfig)),
        "beta": DummyFactory(DummyManifest("beta", SampleConfig)),
    }
    filter_config = PluginFilter(enabled_plugins=["alpha"], disabled_plugins=["beta"])

    monkeypatch.setattr(
        plugins_cmd,
        "_load_all_plugin_factories",
        lambda _settings: (factories, filter_config, {"beta"}),
    )

    metadata = plugins_cmd.gather_plugin_metadata(settings)

    alpha = next(item for item in metadata if item.name == "alpha")
    beta = next(item for item in metadata if item.name == "beta")

    assert alpha.enabled is True
    assert alpha.status_reason is None
    alpha_fields = {field.name: field for field in alpha.config_fields}
    assert alpha_fields["optional_field"].value_label == "11"

    assert beta.enabled is False
    assert beta.status_reason == "disabled via config"


@pytest.fixture()
def cli_runner() -> CliRunner:
    return CliRunner()


@pytest.mark.unit
def test_scaffold_creates_system_plugin(cli_runner: CliRunner, tmp_path: Path) -> None:
    destination = tmp_path / "plugins"
    result = cli_runner.invoke(
        plugins_cmd.app,
        [
            "scaffold",
            "demo_plugin",
            "--type",
            "system",
            "--description",
            "Demo plugin",
            "--version",
            "0.2.0",
            "--path",
            str(destination),
        ],
    )

    assert result.exit_code == 0, result.stdout

    plugin_dir = destination / "demo_plugin"
    assert plugin_dir.is_dir()
    assert (plugin_dir / "plugin.py").is_file()
    assert (plugin_dir / "config.py").is_file()


@pytest.mark.unit
def test_scaffold_rejects_invalid_name(cli_runner: CliRunner, tmp_path: Path) -> None:
    result = cli_runner.invoke(
        plugins_cmd.app,
        [
            "scaffold",
            "Bad-Name",
            "--path",
            str(tmp_path),
        ],
    )

    assert result.exit_code != 0
    output = result.stdout + result.stderr
    assert "must start with a letter" in output
