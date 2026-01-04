"""Tests for Claude CLI option validators."""

from pathlib import Path

import pytest
import typer

from ccproxy.cli.options import claude_options


def test_validate_max_thinking_tokens() -> None:
    assert claude_options.validate_max_thinking_tokens(None, None, None) is None  # type: ignore[arg-type]
    assert claude_options.validate_max_thinking_tokens(None, None, 10) == 10  # type: ignore[arg-type]
    with pytest.raises(typer.BadParameter):
        claude_options.validate_max_thinking_tokens(None, None, -1)  # type: ignore[arg-type]


def test_validate_max_turns() -> None:
    assert claude_options.validate_max_turns(None, None, 2) == 2  # type: ignore[arg-type]
    with pytest.raises(typer.BadParameter):
        claude_options.validate_max_turns(None, None, 0)  # type: ignore[arg-type]


def test_validate_paths(tmp_path: Path) -> None:
    path = tmp_path / "bin"
    path.touch()
    assert claude_options.validate_claude_cli_path(None, None, str(path)) == str(path)  # type: ignore[arg-type]

    directory = tmp_path / "work"
    directory.mkdir()
    assert claude_options.validate_cwd(None, None, str(directory)) == str(directory)  # type: ignore[arg-type]

    with pytest.raises(typer.BadParameter):
        claude_options.validate_cwd(None, None, str(path))  # type: ignore[arg-type]


def test_validate_sdk_message_mode() -> None:
    assert claude_options.validate_sdk_message_mode(None, None, "forward") == "forward"  # type: ignore[arg-type]
    with pytest.raises(typer.BadParameter):
        claude_options.validate_sdk_message_mode(None, None, "invalid")  # type: ignore[arg-type]


def test_validate_pool_size() -> None:
    assert claude_options.validate_pool_size(None, None, 5) == 5  # type: ignore[arg-type]
    with pytest.raises(typer.BadParameter):
        claude_options.validate_pool_size(None, None, 0)  # type: ignore[arg-type]
    with pytest.raises(typer.BadParameter):
        claude_options.validate_pool_size(None, None, 25)  # type: ignore[arg-type]


def test_validate_system_prompt_injection_mode() -> None:
    assert (
        claude_options.validate_system_prompt_injection_mode(None, None, "minimal")  # type: ignore[arg-type]
        == "minimal"
    )
    with pytest.raises(typer.BadParameter):
        claude_options.validate_system_prompt_injection_mode(None, None, "extra")  # type: ignore[arg-type]


def test_claude_options_container() -> None:
    options = claude_options.ClaudeOptions(max_thinking_tokens=50, sdk_pool=True)
    assert options.max_thinking_tokens == 50
    assert options.sdk_pool is True
