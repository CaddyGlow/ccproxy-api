"""Default model metadata and mapping rules for the MiniMax provider."""

from __future__ import annotations

from ccproxy.models.provider import ModelCard, ModelMappingRule


# Timestamps are rounded placeholders, matching the other provider plugins.
# Direct API metadata: https://platform.minimax.io/docs/pricing/overview
DEFAULT_MINIMAX_MODEL_CARDS: list[ModelCard] = [
    ModelCard.model_validate(
        {
            "id": "MiniMax-M3.1-Flash",
            "created": 1735689600,
            "owned_by": "minimax",
            "permission": [],
            "root": "MiniMax-M3.1-Flash",
            "parent": None,
            "context_window": 1_000_000,
            "max_output_tokens": 524_288,
            "pricing_usd_per_million_tokens": {
                "input": 0.15,
                "output": 0.60,
                "cache_read": 0.015,
                "cache_write": None,
            },
            "input_modalities": ["text", "image", "video"],
            "thinking": ["adaptive"],
            "reasoning_efforts": ["low", "medium", "high", "xhigh", "max"],
            "default_reasoning_effort": "max",
        }
    ),
    ModelCard.model_validate(
        {
            "id": "MiniMax-M3",
            "created": 1735689600,
            "owned_by": "minimax",
            "permission": [],
            "root": "MiniMax-M3",
            "parent": None,
            "context_window": 1_000_000,
            "pricing_usd_per_million_tokens": {
                "input": 0.30,
                "output": 1.20,
                "cache_read": 0.06,
                "cache_write": None,
            },
            "pricing_context_tiers": [
                {
                    "context": "<=512k",
                    "input": 0.30,
                    "output": 1.20,
                    "cache_read": 0.06,
                },
                {"context": ">512k", "input": 0.60, "output": 2.40, "cache_read": 0.12},
            ],
            "priority_price_multiplier": 1.5,
            "input_modalities": ["text", "image", "video"],
            "thinking": ["adaptive", "disabled"],
        }
    ),
]


# Unmatched and explicitly requested legacy identifiers pass through unchanged.
DEFAULT_MINIMAX_MODEL_MAPPINGS: list[ModelMappingRule] = [
    ModelMappingRule(
        match=r"^minimax-m3\.1-flash$",
        target="MiniMax-M3.1-Flash",
        kind="regex",
        flags=["IGNORECASE"],
    ),
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
        target="MiniMax-M3.1-Flash",
        kind="regex",
        flags=["IGNORECASE"],
        notes="Default MiniMax alias resolves to MiniMax-M3.1-Flash.",
    ),
]


__all__ = [
    "DEFAULT_MINIMAX_MODEL_CARDS",
    "DEFAULT_MINIMAX_MODEL_MAPPINGS",
]
