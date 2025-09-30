"""Tests for config schema CLI commands."""

import pytest
import typer

from ccproxy.cli.commands.config import schema_commands


class DummyToolkit:
    def __init__(self):
        self.messages = []

    def print(self, message, tag=None):
        self.messages.append((tag, message))

    def print_line(self):
        self.messages.append(("line", ""))

    def print_title(self, *args, **kwargs):  # pragma: no cover - unused
        pass


def test_config_schema_success(monkeypatch, tmp_path):
    toolkit = DummyToolkit()
    monkeypatch.setattr(schema_commands, "get_rich_toolkit", lambda: toolkit)
    monkeypatch.setattr(
        schema_commands,
        "generate_schema_files",
        lambda output_dir: [output_dir / "schema.json"],
    )
    monkeypatch.setattr(
        schema_commands,
        "generate_taplo_config",
        lambda output_dir: output_dir / ".taplo.toml",
    )

    schema_commands.config_schema(output_dir=tmp_path)

    assert any("schema.json" in message for _, message in toolkit.messages)


def test_config_schema_failure(monkeypatch, tmp_path):
    toolkit = DummyToolkit()
    monkeypatch.setattr(schema_commands, "get_rich_toolkit", lambda: toolkit)
    monkeypatch.setattr(
        schema_commands,
        "generate_schema_files",
        lambda output_dir: (_ for _ in ()).throw(RuntimeError("oops")),
    )

    with pytest.raises(typer.Exit):
        schema_commands.config_schema(output_dir=tmp_path)


def test_config_validate_success(monkeypatch, tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text("key=value")
    toolkit = DummyToolkit()
    monkeypatch.setattr(schema_commands, "get_rich_toolkit", lambda: toolkit)
    monkeypatch.setattr(
        schema_commands, "validate_config_with_schema", lambda path: True
    )

    schema_commands.config_validate(config_file)

    assert any("Validating" in message for _, message in toolkit.messages)


def test_config_validate_failure(monkeypatch, tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text("key=value")
    toolkit = DummyToolkit()
    monkeypatch.setattr(schema_commands, "get_rich_toolkit", lambda: toolkit)
    monkeypatch.setattr(
        schema_commands, "validate_config_with_schema", lambda path: False
    )

    with pytest.raises(typer.Exit):
        schema_commands.config_validate(config_file)


def test_config_validate_missing_file(monkeypatch, tmp_path):
    toolkit = DummyToolkit()
    monkeypatch.setattr(schema_commands, "get_rich_toolkit", lambda: toolkit)

    missing = tmp_path / "missing.toml"
    with pytest.raises(typer.Exit):
        schema_commands.config_validate(missing)
