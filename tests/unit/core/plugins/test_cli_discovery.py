"""Tests for CLI discovery filtering logic."""

from types import SimpleNamespace

from ccproxy.core.plugins import cli_discovery


class FakeManifest:
    def __init__(self, name: str, with_cli: bool = True):
        self.name = name
        self.cli_commands = [object()] if with_cli else []
        self.cli_arguments = []


def test_discover_plugin_cli_extensions_filters_disabled(monkeypatch):
    monkeypatch.setattr(
        cli_discovery,
        "_discover_filesystem_cli_extensions",
        lambda: [("foo", FakeManifest("foo")), ("bar", FakeManifest("bar"))],
    )
    monkeypatch.setattr(
        cli_discovery, "_discover_entry_point_cli_extensions", lambda: []
    )

    settings = SimpleNamespace(
        enable_plugins=True,
        disabled_plugins={"bar"},
        enabled_plugins=None,
        plugins=None,
    )

    manifests = cli_discovery.discover_plugin_cli_extensions(settings)

    assert [name for name, _ in manifests] == ["foo"]


def test_discover_plugin_cli_extensions_respects_enabled_list(monkeypatch):
    monkeypatch.setattr(
        cli_discovery,
        "_discover_filesystem_cli_extensions",
        lambda: [("foo", FakeManifest("foo"))],
    )
    monkeypatch.setattr(
        cli_discovery, "_discover_entry_point_cli_extensions", lambda: []
    )

    settings = SimpleNamespace(
        enable_plugins=True,
        disabled_plugins=set(),
        enabled_plugins=["other"],  # allow-list that doesn't include foo
        plugins=None,
    )

    manifests = cli_discovery.discover_plugin_cli_extensions(settings)

    assert manifests == []
