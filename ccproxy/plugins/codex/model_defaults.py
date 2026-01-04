"""Default model metadata and mapping rules for the Codex provider."""

from __future__ import annotations

from ccproxy.models.provider import ModelCard, ModelMappingRule


DEFAULT_CODEX_MODEL_CARDS: list[ModelCard] = [
    ModelCard(
        id="gpt-5",
        created=1723075200,
        owned_by="openai",
        permission=[],
        root="gpt-5",
        parent=None,
    ),
    ModelCard(
        id="gpt-5-codex",
        created=1726444800,
        owned_by="openai",
        permission=[],
        root="gpt-5-codex",
        parent=None,
    ),
]


DEFAULT_CODEX_MODEL_MAPPINGS: list[ModelMappingRule] = [
    ModelMappingRule(match="gpt-", target="gpt-5", kind="prefix"),
    ModelMappingRule(match="o3-", target="gpt-5", kind="prefix"),
    ModelMappingRule(match="o1-", target="gpt-5", kind="prefix"),
    ModelMappingRule(match="claude-", target="gpt-5", kind="prefix"),
]


__all__ = [
    "DEFAULT_CODEX_MODEL_CARDS",
    "DEFAULT_CODEX_MODEL_MAPPINGS",
]
