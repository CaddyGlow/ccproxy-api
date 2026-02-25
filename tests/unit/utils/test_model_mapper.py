from __future__ import annotations

from ccproxy.models.provider import ModelMappingRule
from ccproxy.plugins.claude_shared.model_defaults import DEFAULT_CLAUDE_MODEL_MAPPINGS
from ccproxy.plugins.codex.model_defaults import DEFAULT_CODEX_MODEL_MAPPINGS
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


def test_default_claude_mapping_prefers_latest_sonnet_and_opus() -> None:
    mapper = ModelMapper(DEFAULT_CLAUDE_MODEL_MAPPINGS)

    assert mapper.map("gpt-4o").mapped == "claude-sonnet-4-6"
    assert mapper.map("gpt-5").mapped == "claude-sonnet-4-6"
    assert mapper.map("o1-preview").mapped == "claude-opus-4-6"
    assert mapper.map("o3-mini").mapped == "claude-opus-4-6"
    assert mapper.map("sonnet").mapped == "claude-sonnet-4-6"
    assert mapper.map("opus").mapped == "claude-opus-4-6"


def test_default_codex_mapping_keeps_latest_codex_model() -> None:
    mapper = ModelMapper(DEFAULT_CODEX_MODEL_MAPPINGS)

    assert mapper.map("gpt-5-codex").mapped == "gpt-5.3-codex"
