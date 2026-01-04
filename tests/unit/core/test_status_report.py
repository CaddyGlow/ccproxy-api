"""Unit tests for status report helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from ccproxy.config.settings import Settings
from ccproxy.core import status_report


class FakeManifest:
    def __init__(
        self, version: str | None = None, description: str | None = None
    ) -> None:
        self.version = version
        self.description = description


class FakeFactory:
    def __init__(
        self, manifest: FakeManifest | None = None, error: Exception | None = None
    ) -> None:
        self._manifest = manifest or FakeManifest()
        self._error = error

    def get_manifest(self) -> FakeManifest:
        if self._error:
            raise self._error
        return self._manifest


def build_settings(**overrides: object) -> Settings:
    data: dict[str, object] = {
        "server": {"host": "127.0.0.1", "port": 8080},
        "logging": {"level": "info"},
        "security": {"auth_token": None},
        "enable_plugins": True,
        "plugins_disable_local_discovery": False,
        "plugin_discovery": {"directories": []},
        "plugins": {
            "request_tracer": {
                "enabled": True,
                "json_logs_enabled": False,
                "raw_http_enabled": False,
            }
        },
    }
    logging_overrides = overrides.pop("logging", None)
    if logging_overrides is not None:
        data["logging"].update(logging_overrides)  # type: ignore[arg-type,attr-defined]

    data.update(overrides)
    return Settings.model_validate(data)


def test_collect_system_snapshot_includes_directory_status(tmp_path: Path) -> None:
    existing_dir = tmp_path / "existing"
    existing_dir.mkdir()

    missing_dir = tmp_path / "missing"

    settings = build_settings(
        server={"host": "0.0.0.0", "port": 9001},
        logging={"level": "debug"},
        security={"auth_token": "token"},
        plugin_discovery={"directories": [existing_dir, missing_dir]},
    )

    snapshot = status_report.collect_system_snapshot(settings)

    assert snapshot.host == "0.0.0.0"
    assert snapshot.port == 9001
    assert snapshot.log_level == "DEBUG"
    assert snapshot.auth_token_configured is True
    assert snapshot.plugins_enabled is True

    statuses = {entry.path: entry.exists for entry in snapshot.plugin_directories}
    assert statuses[existing_dir] is True
    assert statuses[missing_dir] is False


def test_collect_config_snapshot_reports_expected_paths(tmp_path: Path) -> None:
    config_file = tmp_path / "ccproxy.toml"
    config_file.write_text("[server]\nport = 9000\n", encoding="utf-8")

    snapshot = status_report.collect_config_snapshot(cwd=tmp_path)

    paths = [source.path for source in snapshot.sources]
    assert paths[0] == tmp_path / ".ccproxy.toml"
    assert paths[1] == config_file

    existence = {source.path: source.exists for source in snapshot.sources}
    assert existence[config_file] is True
    assert existence[tmp_path / ".ccproxy.toml"] is False


def test_collect_plugin_snapshot_handles_enabled_and_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    disabled_dir = plugin_dir / "beta"
    disabled_dir.mkdir()
    (disabled_dir / "plugin.py").write_text("# stub", encoding="utf-8")

    settings = build_settings(
        plugin_discovery={"directories": [plugin_dir]},
        enabled_plugins=["alpha"],
        disabled_plugins=["gamma"],
    )

    manifest = FakeManifest(version="1.2.3", description="Alpha plugin")
    factory_map = {"alpha": FakeFactory(manifest=manifest)}

    monkeypatch.setattr(
        status_report,
        "discover_and_load_plugins",
        lambda _settings: factory_map,
    )

    snapshot = status_report.collect_plugin_snapshot(settings)

    assert snapshot.plugin_system_enabled is True
    assert [info.name for info in snapshot.enabled_plugins] == ["alpha"]
    assert snapshot.enabled_plugins[0].state == "enabled"
    assert snapshot.disabled_plugins == ("beta",)
    assert set(snapshot.configuration_notes) == {
        "Explicitly disabled: 1",
        "Allow-list active: 1 allowed",
    }


def test_collect_plugin_snapshot_records_manifest_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()

    settings = build_settings(plugin_discovery={"directories": [plugin_dir]})

    error = RuntimeError("manifest failure for testing")
    factory_map = {"broken": FakeFactory(error=error)}

    monkeypatch.setattr(
        status_report,
        "discover_and_load_plugins",
        lambda _settings: factory_map,
    )

    snapshot = status_report.collect_plugin_snapshot(settings)

    assert snapshot.enabled_plugins[0].state == "error"
    assert snapshot.enabled_plugins[0].error == str(error)
    assert snapshot.disabled_plugins == ()


def test_collect_plugin_snapshot_when_disabled() -> None:
    settings = build_settings(enable_plugins=False)

    snapshot = status_report.collect_plugin_snapshot(settings)

    assert snapshot.plugin_system_enabled is False
    assert snapshot.enabled_plugins == ()
    assert snapshot.disabled_plugins == ()
