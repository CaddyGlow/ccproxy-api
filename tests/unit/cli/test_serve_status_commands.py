"""Smoke coverage for serve and status CLI commands."""

import importlib
from types import SimpleNamespace

from typer.testing import CliRunner


cli_main = importlib.import_module("ccproxy.cli.main")
serve_module = importlib.import_module("ccproxy.cli.commands.serve")
status_module = importlib.import_module("ccproxy.cli.commands.status")


def test_cli_serve_invokes_local_server(monkeypatch) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    stub_settings = SimpleNamespace(
        server=SimpleNamespace(host="127.0.0.1", port=9000, reload=False, workers=1),
        security=SimpleNamespace(auth_token=None),
        logging=SimpleNamespace(format="text", level="INFO", file=None),
        plugins={},
        enable_plugins=False,
        plugin_discovery=SimpleNamespace(directories=[]),
        disabled_plugins=[],
        enabled_plugins=[],
        plugins_disable_local_discovery=False,
    )

    def fake_from_config(cls, *, config_path=None, cli_context=None):  # type: ignore[override]
        captured["cli_context"] = cli_context
        return stub_settings

    monkeypatch.setattr(
        serve_module.Settings,
        "from_config",
        classmethod(fake_from_config),
    )
    monkeypatch.setattr(serve_module, "setup_logging", lambda **_: None)
    monkeypatch.setattr(
        serve_module,
        "get_rich_toolkit",
        lambda: SimpleNamespace(
            print=lambda *_, **__: None,
            print_line=lambda: None,
            print_title=lambda *_, **__: None,
        ),
    )

    ran = {}

    def fake_run_local(settings):
        ran["settings"] = settings

    monkeypatch.setattr(serve_module, "_run_local_server", fake_run_local)

    result = runner.invoke(
        cli_main.app, ["serve", "--host", "0.0.0.0", "--port", "4321"]
    )

    assert result.exit_code == 0
    assert ran.get("settings") is stub_settings
    assert captured["cli_context"]["host"] == "0.0.0.0"
    assert captured["cli_context"]["port"] == 4321


def test_status_command_runs_with_stubbed_snapshots(monkeypatch) -> None:
    runner = CliRunner()

    stub_settings = object()
    monkeypatch.setattr(
        status_module.Settings,
        "from_config",
        classmethod(lambda cls: stub_settings),
    )

    fake_snapshot = status_module.SystemSnapshot(
        host="127.0.0.1",
        port=8000,
        log_level="INFO",
        auth_token_configured=False,
        plugins_enabled=False,
        plugin_directories=(),
    )

    monkeypatch.setattr(
        status_module,
        "collect_system_snapshot",
        lambda _settings: fake_snapshot,
    )

    result = runner.invoke(cli_main.app, ["status", "--no-plugins", "--no-config"])

    assert result.exit_code == 0
