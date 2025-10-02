"""Tests for CLI logging utilities."""

from types import SimpleNamespace

import pytest

from ccproxy.utils import cli_logging


def _make_cli_info(**overrides):
    info = {
        "name": "example",
        "is_available": True,
        "version": "1.0.0",
        "source": "path",
        "path": "/usr/bin/example",
        "command": ["example"],
        "package_manager": None,
    }
    info.update(overrides)
    return info


def test_log_cli_info_emits_debug_and_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    messages: list[tuple[str, dict[str, object]]] = []

    monkeypatch.setattr(
        cli_logging,
        "logger",
        SimpleNamespace(
            debug=lambda *args, **kwargs: messages.append((args[0], kwargs)),
            warning=lambda *args, **kwargs: messages.append((args[0], kwargs)),
        ),
    )

    cli_logging.log_cli_info(
        {
            "ok": _make_cli_info(name="ok"),
            "missing": _make_cli_info(
                name="missing", is_available=False, version="2.0"
            ),
        },
        context="startup",
    )

    assert messages[0][0] == "startup_cli_available"
    assert messages[1][0] == "startup_cli_unavailable"
    assert messages[1][1]["expected_version"] == "2.0"


def test_log_plugin_summary_logs_and_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    from typing import Any

    called: dict[str, Any] = {}

    def debug_handler(*args: object, **kwargs: object) -> None:
        if "debug" not in called:
            called["debug"] = []
        called["debug"].append((args, kwargs))

    monkeypatch.setattr(
        cli_logging,
        "logger",
        SimpleNamespace(debug=debug_handler),
    )
    monkeypatch.setattr(
        cli_logging,
        "log_cli_info",
        lambda cli_info, context: called.setdefault("delegate", (cli_info, context)),
    )

    cli_logging.log_plugin_summary(
        {
            "status": "enabled",
            "cli_info": {"bin": _make_cli_info(name="bin")},
        },
        plugin_name="myplugin",
    )

    assert called["debug"][0][0][0] == "plugin_summary"
    assert called["delegate"][1] == "myplugin_plugin"


def test_format_cli_info_for_display_handles_sources() -> None:
    unavailable = cli_logging.format_cli_info_for_display(
        _make_cli_info(name="foo", is_available=False)
    )
    assert unavailable["status"] == "unavailable"

    path_case = cli_logging.format_cli_info_for_display(
        _make_cli_info(name="bar", path="/usr/bin/bar")
    )
    assert path_case["path"] == "/usr/bin/bar"

    pkg_case = cli_logging.format_cli_info_for_display(
        _make_cli_info(
            name="baz",
            source="package_manager",
            package_manager="pipx",
            command=["pipx", "run", "baz"],
        )
    )
    assert pkg_case["package_manager"] == "pipx"
    assert pkg_case["command"] == "pipx run baz"


def test_create_cli_summary_table_formats_all_entries() -> None:
    summary = cli_logging.create_cli_summary_table(
        {
            "foo": _make_cli_info(name="foo"),
            "bar": _make_cli_info(name="bar", is_available=False),
        }
    )

    assert len(summary) == 2
    assert {entry["name"] for entry in summary} == {"foo", "bar"}
