from __future__ import annotations

import json
from typing import Any, Literal, cast

from ccproxy.llms.models import anthropic as anthropic_models


def openai_usage_to_anthropic_usage(openai_usage: Any | None) -> anthropic_models.Usage:
    """Map OpenAI usage structures to Anthropic Usage with best-effort coverage.

    Supports both Chat Completions and Responses usage models/dicts.
    - input_tokens  <- prompt_tokens or input_tokens
    - output_tokens <- completion_tokens or output_tokens
    - cache_read_input_tokens from prompt/input tokens details.cached_tokens if present
    - cache_creation_input_tokens left 0 unless explicitly provided
    """
    if openai_usage is None:
        return anthropic_models.Usage(input_tokens=0, output_tokens=0)

    # Handle dict or pydantic model
    as_dict: dict[str, Any]
    if hasattr(openai_usage, "model_dump"):
        as_dict = openai_usage.model_dump()
    elif isinstance(openai_usage, dict):
        as_dict = openai_usage
    else:
        # Fallback to attribute access
        as_dict = {
            "input_tokens": getattr(openai_usage, "input_tokens", None),
            "output_tokens": getattr(openai_usage, "output_tokens", None),
            "prompt_tokens": getattr(openai_usage, "prompt_tokens", None),
            "completion_tokens": getattr(openai_usage, "completion_tokens", None),
            "input_tokens_details": getattr(openai_usage, "input_tokens_details", None),
            "prompt_tokens_details": getattr(
                openai_usage, "prompt_tokens_details", None
            ),
        }

    input_tokens = (
        as_dict.get("input_tokens")
        if isinstance(as_dict.get("input_tokens"), int)
        else as_dict.get("prompt_tokens")
    )
    output_tokens = (
        as_dict.get("output_tokens")
        if isinstance(as_dict.get("output_tokens"), int)
        else as_dict.get("completion_tokens")
    )

    input_tokens = int(input_tokens or 0)
    output_tokens = int(output_tokens or 0)

    # cached tokens
    cached = 0
    details = as_dict.get("input_tokens_details") or as_dict.get(
        "prompt_tokens_details"
    )
    if isinstance(details, dict):
        cached = int(details.get("cached_tokens") or 0)
    elif details is not None:
        cached = int(getattr(details, "cached_tokens", 0) or 0)

    return anthropic_models.Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cached,
        cache_creation_input_tokens=0,
    )


def map_openai_finish_to_anthropic_stop(
    finish_reason: str | None,
) -> (
    Literal[
        "end_turn", "max_tokens", "stop_sequence", "tool_use", "pause_turn", "refusal"
    ]
    | None
):
    """Map OpenAI finish_reason to Anthropic stop_reason."""
    mapping = {
        "stop": "end_turn",
        "length": "max_tokens",
        "function_call": "tool_use",
        "tool_calls": "tool_use",
        "content_filter": "stop_sequence",
        None: "end_turn",
    }
    result = mapping.get(finish_reason, "end_turn")
    return cast(
        Literal[
            "end_turn",
            "max_tokens",
            "stop_sequence",
            "tool_use",
            "pause_turn",
            "refusal",
        ]
        | None,
        result,
    )


def strict_parse_tool_arguments(
    arguments: str | dict[str, Any] | None,
) -> dict[str, Any]:
    """Strictly parse tool/function arguments as JSON object.

    - If a dict is provided, return as-is.
    - If a string is provided, it must be valid JSON and deserialize to a dict.
    - Otherwise, raise ValueError.
    """
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        parsed = json.loads(arguments)
        if not isinstance(parsed, dict):
            raise ValueError("Tool/function arguments must be a JSON object")
        return parsed
    raise ValueError("Unsupported tool/function arguments type")
