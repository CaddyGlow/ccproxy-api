"""Unified model mapping utilities for OpenAI and Claude models.

This module provides a single source of truth for all model mappings,
consolidating OpenAI→Claude mappings and Claude alias resolution.
"""

from __future__ import annotations


# Combined mapping: OpenAI models → Claude models AND Claude aliases → canonical Claude models
MODEL_MAPPING: dict[str, str] = {
    # OpenAI GPT-4 models → Claude 3.5 Sonnet (most comparable)
    "gpt-4": "claude-3-5-sonnet-20241022",
    "gpt-4-turbo": "claude-3-5-sonnet-20241022",
    "gpt-4-turbo-preview": "claude-3-5-sonnet-20241022",
    "gpt-4-1106-preview": "claude-3-5-sonnet-20241022",
    "gpt-4-0125-preview": "claude-3-5-sonnet-20241022",
    "gpt-4-turbo-2024-04-09": "claude-3-5-sonnet-20241022",
    # OpenAI GPT-4o models → Claude 3.7 Sonnet
    "gpt-4o": "claude-3-7-sonnet-20250219",
    "gpt-4o-2024-05-13": "claude-3-7-sonnet-20250219",
    "gpt-4o-2024-08-06": "claude-3-7-sonnet-20250219",
    "gpt-4o-2024-11-20": "claude-3-7-sonnet-20250219",
    # OpenAI GPT-4o-mini models → Claude 3.5 Haiku
    "gpt-4o-mini": "claude-3-5-haiku-latest",
    "gpt-4o-mini-2024-07-18": "claude-3-5-haiku-latest",
    # OpenAI o1 models → Claude models that support thinking
    "o1": "claude-opus-4-20250514",
    "o1-preview": "claude-opus-4-20250514",
    "o1-mini": "claude-sonnet-4-20250514",
    # OpenAI o3 models → Claude Opus 4
    "o3-mini": "claude-opus-4-20250514",
    # OpenAI GPT-3.5 models → Claude 3.5 Haiku (faster, cheaper)
    "gpt-3.5-turbo": "claude-3-5-haiku-20241022",
    "gpt-3.5-turbo-16k": "claude-3-5-haiku-20241022",
    "gpt-3.5-turbo-1106": "claude-3-5-haiku-20241022",
    "gpt-3.5-turbo-0125": "claude-3-5-haiku-20241022",
    # OpenAI text models → Claude 3.5 Sonnet
    "text-davinci-003": "claude-3-5-sonnet-20241022",
    "text-davinci-002": "claude-3-5-sonnet-20241022",
    # Claude model aliases → canonical Claude models
    "claude-3-5-sonnet-latest": "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-20240620": "claude-3-5-sonnet-20240620",
    "claude-3-5-sonnet-20241022": "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-latest": "claude-3-5-haiku-20241022",
    "claude-3-5-haiku-20241022": "claude-3-5-haiku-20241022",
    "claude-3-opus": "claude-3-opus-20240229",
    "claude-3-opus-20240229": "claude-3-opus-20240229",
    "claude-3-sonnet": "claude-3-sonnet-20240229",
    "claude-3-sonnet-20240229": "claude-3-sonnet-20240229",
    "claude-3-haiku": "claude-3-haiku-20240307",
    "claude-3-haiku-20240307": "claude-3-haiku-20240307",
}


def map_model_to_claude(model_name: str) -> str:
    """Map any model name to its canonical Claude model name.

    This function handles:
    - OpenAI model names → Claude equivalents
    - Claude aliases → canonical Claude names
    - Pattern matching for versioned models
    - Pass-through for unknown models

    Args:
        model_name: Model identifier (OpenAI, Claude, or alias)

    Returns:
        Canonical Claude model identifier
    """
    # Direct mapping first (handles both OpenAI and Claude aliases)
    claude_model = MODEL_MAPPING.get(model_name)
    if claude_model:
        return claude_model

    # Pattern matching for versioned OpenAI models
    if model_name.startswith("gpt-4o-mini"):
        return "claude-3-5-haiku-latest"
    elif model_name.startswith("gpt-4o") or model_name.startswith("gpt-4"):
        return "claude-3-7-sonnet-20250219"
    elif model_name.startswith("gpt-3.5"):
        return "claude-3-5-haiku-latest"
    elif model_name.startswith("o1"):
        return "claude-sonnet-4-20250514"
    elif model_name.startswith("o3"):
        return "claude-opus-4-20250514"
    elif model_name.startswith("gpt"):
        return "claude-sonnet-4-20250514"

    # If it's already a Claude model, pass through unchanged
    if model_name.startswith("claude-"):
        return model_name

    # For unknown models, pass through unchanged
    return model_name


def get_openai_to_claude_mapping() -> dict[str, str]:
    """Get mapping of OpenAI models to Claude models.

    Returns:
        Dictionary mapping OpenAI model names to Claude model names
    """
    return {k: v for k, v in MODEL_MAPPING.items() if not k.startswith("claude-")}


def get_claude_aliases_mapping() -> dict[str, str]:
    """Get mapping of Claude aliases to canonical Claude names.

    Returns:
        Dictionary mapping Claude aliases to canonical model names
    """
    return {k: v for k, v in MODEL_MAPPING.items() if k.startswith("claude-")}


def get_supported_claude_models() -> list[str]:
    """Get list of supported canonical Claude models.

    Returns:
        Sorted list of unique canonical Claude model names
    """
    return sorted(set(MODEL_MAPPING.values()))


def is_openai_model(model_name: str) -> bool:
    """Check if a model name is an OpenAI model.

    Args:
        model_name: Model identifier to check

    Returns:
        True if the model is an OpenAI model, False otherwise
    """
    return (
        model_name.startswith(("gpt-", "o1", "o3", "text-davinci"))
        or model_name in get_openai_to_claude_mapping()
    )


def is_claude_model(model_name: str) -> bool:
    """Check if a model name is a Claude model (canonical or alias).

    Args:
        model_name: Model identifier to check

    Returns:
        True if the model is a Claude model, False otherwise
    """
    return (
        model_name.startswith("claude-") or model_name in get_claude_aliases_mapping()
    )


# Backward compatibility exports
OPENAI_TO_CLAUDE_MODEL_MAPPING = get_openai_to_claude_mapping()
CLAUDE_MODEL_MAPPINGS = get_claude_aliases_mapping()


# Legacy function aliases
def map_openai_model_to_claude(openai_model: str) -> str:
    """Legacy alias for map_model_to_claude().

    Args:
        openai_model: OpenAI model identifier

    Returns:
        Claude model identifier
    """
    return map_model_to_claude(openai_model)


def get_canonical_model_name(model_name: str) -> str:
    """Legacy alias for map_model_to_claude().

    Args:
        model_name: Model name (possibly an alias)

    Returns:
        Canonical model name
    """
    return map_model_to_claude(model_name)


__all__ = [
    "MODEL_MAPPING",
    "map_model_to_claude",
    "get_openai_to_claude_mapping",
    "get_claude_aliases_mapping",
    "get_supported_claude_models",
    "is_openai_model",
    "is_claude_model",
    # Backward compatibility
    "OPENAI_TO_CLAUDE_MODEL_MAPPING",
    "CLAUDE_MODEL_MAPPINGS",
    "map_openai_model_to_claude",
    "get_canonical_model_name",
]
