"""Default model metadata and mapping rules for the MiniMax provider."""

from __future__ import annotations

from ccproxy.models.provider import ModelCard, ModelMappingRule


# Fallback metadata served from ``/models`` when the upstream listing is
# unavailable. Timestamps are rounded placeholders, matching the convention
# used by the other provider plugins in this repository.
DEFAULT_MINIMAX_MODEL_CARDS: list[ModelCard] = [
    ModelCard(
        id="MiniMax-M3",
        created=1735689600,
        owned_by="minimax",
        permission=[],
        root="MiniMax-M3",
        parent=None,
        context_window=1_000_000,
        pricing_usd_per_million_tokens={
            "input": 0.6,
            "output": 2.4,
            "cache_read": 0.12,
            "cache_write": None,
        },
        input_modalities=["text", "image", "video"],
        thinking=["adaptive", "disabled"],
    ),
    ModelCard(
        id="MiniMax-M2.7",
        created=1735689600,
        owned_by="minimax",
        permission=[],
        root="MiniMax-M2.7",
        parent=None,
        context_window=204_800,
        pricing_usd_per_million_tokens={
            "input": 0.3,
            "output": 1.2,
            "cache_read": 0.06,
            "cache_write": 0.375,
        },
        input_modalities=["text"],
        thinking=["always_on"],
    ),
]


# Convenience aliases so short, case-insensitive client identifiers resolve to
# the canonical upstream model names. Unmatched identifiers pass through
# unchanged.
DEFAULT_MINIMAX_MODEL_MAPPINGS: list[ModelMappingRule] = [
    ModelMappingRule(
        match=r"^minimax-m3$",
        target="MiniMax-M3",
        kind="regex",
        flags=["IGNORECASE"],
    ),
    ModelMappingRule(
        match=r"^minimax-m2\.7$",
        target="MiniMax-M2.7",
        kind="regex",
        flags=["IGNORECASE"],
    ),
    ModelMappingRule(
        match=r"^minimax$",
        target="MiniMax-M3",
        kind="regex",
        flags=["IGNORECASE"],
        notes="Default MiniMax alias resolves to the flagship model.",
    ),
]


__all__ = [
    "DEFAULT_MINIMAX_MODEL_CARDS",
    "DEFAULT_MINIMAX_MODEL_MAPPINGS",
]
