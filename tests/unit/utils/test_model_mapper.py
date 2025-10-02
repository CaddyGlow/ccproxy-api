from __future__ import annotations

from ccproxy.models.provider import ModelMappingRule
from ccproxy.utils.model_mapper import (
    ModelMapper,
    add_model_alias,
    restore_model_aliases,
)


def test_model_mapper_honors_rule_order_and_types() -> None:
    rules = [
        ModelMappingRule(match="gpt-4o-mini", target="haiku-latest", kind="prefix"),
        ModelMappingRule(match=r"^gpt-4", target="sonnet", kind="regex"),
    ]
    mapper = ModelMapper(rules)

    mini_match = mapper.map("gpt-4o-mini-2024-07-18")
    assert mini_match.mapped == "haiku-latest"
    assert mini_match.rule is rules[0]

    base_match = mapper.map("gpt-4-turbo")
    assert base_match.mapped == "sonnet"
    assert base_match.rule is rules[1]

    passthrough = mapper.map("claude-3-5-sonnet-20241022")
    assert passthrough.mapped == "claude-3-5-sonnet-20241022"
    assert passthrough.rule is None


def test_restore_model_aliases_updates_nested_payloads() -> None:
    from typing import Any

    metadata: dict[str, object] = {}
    add_model_alias(metadata, original="gpt-4o-mini", mapped="claude-haiku")

    payload: dict[str, Any] = {
        "model": "claude-haiku",
        "choices": [
            {
                "message": {
                    "metadata": {"model": "claude-haiku"},
                    "content": "hello",
                }
            }
        ],
    }

    restore_model_aliases(payload, metadata)

    assert payload["model"] == "gpt-4o-mini"
    nested: dict[str, Any] = payload["choices"][0]["message"]["metadata"]
    assert nested["model"] == "gpt-4o-mini"
