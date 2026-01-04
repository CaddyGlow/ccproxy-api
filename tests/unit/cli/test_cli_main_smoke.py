"""Minimal smoke coverage for the CLI entrypoints."""

import importlib

import pytest
from typer.testing import CliRunner


cli_main = importlib.import_module("ccproxy.cli.main")


def _runner() -> CliRunner:
    """Return a new CliRunner instance for clarity in tests."""
    return CliRunner()


def test_cli_help_runs_without_error() -> None:
    runner = _runner()
    result = runner.invoke(cli_main.app, ["--help"])

    assert result.exit_code == 0


def test_cli_version_flag_exits_cleanly() -> None:
    runner = _runner()
    result = runner.invoke(cli_main.app, ["--version"])

    assert result.exit_code == 0


def test_cli_status_help_runs_without_error() -> None:
    runner = _runner()
    result = runner.invoke(cli_main.app, ["status", "--help"])

    assert result.exit_code == 0


def test_plugin_cli_registration_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli_main, "discover_plugin_cli_extensions", lambda _settings: []
    )
    monkeypatch.setattr(cli_main, "_plugins_registered", False, raising=False)

    cli_main.ensure_plugin_cli_extensions_registered(cli_main.app)
    cli_main.ensure_plugin_cli_extensions_registered(cli_main.app)

    assert cli_main._plugins_registered is True


def test_collect_relevant_env_masks_sensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PLUGINS__TOKEN", "super-secret-value")
    monkeypatch.setenv("SERVER__PORT", "8080")
    monkeypatch.setenv("UNRELATED_VAR", "ignore-me")

    env = cli_main._collect_relevant_env()

    assert env["PLUGINS__TOKEN"].startswith("***MASKED***")
    assert "UNRELATED_VAR" not in env
