"""Tests for CLI decorators metadata helpers."""

from ccproxy.cli import decorators


def test_needs_auth_provider_sets_flag() -> None:
    @decorators.needs_auth_provider()
    def sample() -> None:  # pragma: no cover - body unused
        pass

    assert decorators.get_command_auth_provider(sample) is True


def test_allows_plugins_attaches_list() -> None:
    plugins = ["metrics", "tracer"]

    @decorators.allows_plugins(plugins)
    def sample() -> None:  # pragma: no cover - body unused
        pass

    assert decorators.get_command_allowed_plugins(sample) == plugins


def test_getters_return_defaults_when_not_decorated() -> None:
    def plain() -> None:  # pragma: no cover - body unused
        pass

    assert decorators.get_command_auth_provider(plain) is False
    assert decorators.get_command_allowed_plugins(plain) == []
