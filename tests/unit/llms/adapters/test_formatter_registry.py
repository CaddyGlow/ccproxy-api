from __future__ import annotations

import pytest

from ccproxy.llms.formatters.anthropic_to_openai.helpers import (
    convert__anthropic_message_to_openai_responses__request,
)
from ccproxy.llms.formatters.formatter_registry import (
    FormatterRegistry,
    iter_registered_formatters,
    load_builtin_formatter_modules,
)


def test_formatter_registry_register_and_get() -> None:
    registry = FormatterRegistry()

    def _example(value: str) -> str:
        return value.upper()

    registry.register("source", "target", "operation", _example, "test_module")

    assert registry.get("source", "target", "operation")("hi") == "HI"
    assert registry.get_all("source", "target") == {"operation": _example}

    with pytest.raises(ValueError):
        registry.get("missing", "target", "operation")


def test_formatter_registry_bidirectional() -> None:
    registry = FormatterRegistry()

    registry.register("a", "b", "first", lambda: None, "mod1")
    registry.register("b", "a", "second", lambda: None, "mod2")

    result = registry.get_bidirectional("a", "b")
    assert set(result) == {"a_to_b", "b_to_a"}


def test_formatter_registry_loads_builtin_helpers() -> None:
    load_builtin_formatter_modules()

    registry = FormatterRegistry()
    registry.register_many(iter_registered_formatters())

    fn = registry.get("anthropic.messages", "openai.responses", "request")
    assert fn is convert__anthropic_message_to_openai_responses__request
