"""Tests for Claude CLI option validators."""

import pytest
import typer

from ccproxy.cli.options import claude_options


def test_validate_max_thinking_tokens():
    assert claude_options.validate_max_thinking_tokens(None, None, None) is None
    assert claude_options.validate_max_thinking_tokens(None, None, 10) == 10
    with pytest.raises(typer.BadParameter):
        claude_options.validate_max_thinking_tokens(None, None, -1)


def test_validate_max_turns():
    assert claude_options.validate_max_turns(None, None, 2) == 2
    with pytest.raises(typer.BadParameter):
        claude_options.validate_max_turns(None, None, 0)


def test_validate_paths(tmp_path):
    path = tmp_path / "bin"
    path.touch()
    assert claude_options.validate_claude_cli_path(None, None, str(path)) == str(path)

    directory = tmp_path / "work"
    directory.mkdir()
    assert claude_options.validate_cwd(None, None, str(directory)) == str(directory)

    with pytest.raises(typer.BadParameter):
        claude_options.validate_cwd(None, None, str(path))


def test_validate_sdk_message_mode():
    assert claude_options.validate_sdk_message_mode(None, None, "forward") == "forward"
    with pytest.raises(typer.BadParameter):
        claude_options.validate_sdk_message_mode(None, None, "invalid")


def test_validate_pool_size():
    assert claude_options.validate_pool_size(None, None, 5) == 5
    with pytest.raises(typer.BadParameter):
        claude_options.validate_pool_size(None, None, 0)
    with pytest.raises(typer.BadParameter):
        claude_options.validate_pool_size(None, None, 25)


def test_validate_system_prompt_injection_mode():
    assert (
        claude_options.validate_system_prompt_injection_mode(None, None, "minimal")
        == "minimal"
    )
    with pytest.raises(typer.BadParameter):
        claude_options.validate_system_prompt_injection_mode(None, None, "extra")


def test_claude_options_container():
    options = claude_options.ClaudeOptions(max_thinking_tokens=50, sdk_pool=True)
    assert options.max_thinking_tokens == 50
    assert options.sdk_pool is True
