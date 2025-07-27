"""Shared models provider for CCProxy API Server.

This module provides a centralized source for all available models,
combining Claude and OpenAI models in a consistent format.
"""

from __future__ import annotations

from typing import Any

from ccproxy.utils.model_mapping import get_supported_claude_models


def get_anthropic_models() -> list[dict[str, Any]]:
    """Get list of Anthropic models with metadata.

    Returns:
        List of Anthropic model entries with type, id, display_name, and created_at fields
    """
    # Model display names mapping
    display_names = {
        "claude-opus-4-20250514": "Claude Opus 4",
        "claude-sonnet-4-20250514": "Claude Sonnet 4",
        "claude-3-7-sonnet-20250219": "Claude Sonnet 3.7",
        "claude-3-5-sonnet-20241022": "Claude Sonnet 3.5 (New)",
        "claude-3-5-haiku-20241022": "Claude Haiku 3.5",
        "claude-3-5-haiku-latest": "Claude Haiku 3.5",
        "claude-3-5-sonnet-20240620": "Claude Sonnet 3.5 (Old)",
        "claude-3-haiku-20240307": "Claude Haiku 3",
        "claude-3-opus-20240229": "Claude Opus 3",
    }

    # Model creation timestamps
    timestamps = {
        "claude-opus-4-20250514": 1747526400,  # 2025-05-22
        "claude-sonnet-4-20250514": 1747526400,  # 2025-05-22
        "claude-3-7-sonnet-20250219": 1740268800,  # 2025-02-24
        "claude-3-5-sonnet-20241022": 1729555200,  # 2024-10-22
        "claude-3-5-haiku-20241022": 1729555200,  # 2024-10-22
        "claude-3-5-haiku-latest": 1729555200,  # 2024-10-22
        "claude-3-5-sonnet-20240620": 1718841600,  # 2024-06-20
        "claude-3-haiku-20240307": 1709769600,  # 2024-03-07
        "claude-3-opus-20240229": 1709164800,  # 2024-02-29
    }

    # Get supported Claude models from existing utility
    supported_models = get_supported_claude_models()

    # Create Anthropic-style model entries
    models = []
    for model_id in supported_models:
        models.append(
            {
                "type": "model",
                "id": model_id,
                "display_name": display_names.get(model_id, model_id),
                "created_at": timestamps.get(model_id, 1677610602),  # Default timestamp
            }
        )

    return models


def get_openai_models() -> list[dict[str, Any]]:
    """Get list of recent OpenAI models with metadata.

    Returns:
        List of OpenAI model entries with id, object, created, and owned_by fields
    """
    return [
        {
            "id": "gpt-4o",
            "object": "model",
            "created": 1715367049,
            "owned_by": "openai",
        },
        {
            "id": "gpt-4o-mini",
            "object": "model",
            "created": 1721172741,
            "owned_by": "openai",
        },
        {
            "id": "gpt-4-turbo",
            "object": "model",
            "created": 1712361441,
            "owned_by": "openai",
        },
        {
            "id": "gpt-4-turbo-preview",
            "object": "model",
            "created": 1706037777,
            "owned_by": "openai",
        },
        {
            "id": "o1",
            "object": "model",
            "created": 1734375816,
            "owned_by": "openai",
        },
        {
            "id": "o1-mini",
            "object": "model",
            "created": 1725649008,
            "owned_by": "openai",
        },
        {
            "id": "o1-preview",
            "object": "model",
            "created": 1725648897,
            "owned_by": "openai",
        },
        {
            "id": "o3",
            "object": "model",
            "created": 1744225308,
            "owned_by": "openai",
        },
        {
            "id": "o3-mini",
            "object": "model",
            "created": 1737146383,
            "owned_by": "openai",
        },
    ]


def get_models_list() -> dict[str, Any]:
    """Get combined list of available Claude and OpenAI models.

    Returns:
        Dictionary with combined list of models in mixed format compatible with both
        Anthropic and OpenAI API specifications
    """
    anthropic_models = get_anthropic_models()
    openai_models = get_openai_models()

    # Return combined response in mixed format
    return {
        "data": anthropic_models + openai_models,
        "has_more": False,
        "object": "list",
    }


__all__ = [
    "get_anthropic_models",
    "get_openai_models",
    "get_models_list",
]
