from __future__ import annotations

import base64
import contextlib
import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal, TypeVar, cast

from pydantic import BaseModel

from ccproxy.llms.models import anthropic as anthropic_models
from ccproxy.llms.models import openai as openai_models


BaseModelT = TypeVar("BaseModelT", bound=BaseModel)


@dataclass(slots=True)
class UsageSnapshot:
    """Normalized token usage fields shared across providers."""

    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    reasoning_tokens: int = 0


def _coerce_pydantic_model(model: Any, target: type[BaseModelT]) -> BaseModelT:
    """Convert raw input into the requested Pydantic model type."""

    if isinstance(model, target):
        return model

    if hasattr(model, "model_dump"):
        return target.model_validate(model.model_dump())

    if isinstance(model, dict):
        return target.model_validate(model)

    raise TypeError(
        f"Expected {target.__name__} compatible value, received {type(model).__name__}"
    )


def anthropic_usage_snapshot(usage: Any) -> UsageSnapshot:
    """Return a normalized snapshot for Anthropic Usage payloads."""

    normalized = _coerce_pydantic_model(usage, anthropic_models.Usage)

    cache_read = normalized.cache_read_input_tokens or 0

    cache_creation = normalized.cache_creation_input_tokens or 0
    if cache_creation == 0 and normalized.cache_creation:
        cache_creation = (
            normalized.cache_creation.ephemeral_1h_input_tokens
            + normalized.cache_creation.ephemeral_5m_input_tokens
        )

    return UsageSnapshot(
        input_tokens=normalized.input_tokens or 0,
        output_tokens=normalized.output_tokens or 0,
        cache_read_tokens=cache_read,
        cache_creation_tokens=cache_creation,
    )


def openai_response_usage_snapshot(usage: Any) -> UsageSnapshot:
    """Return a normalized snapshot for OpenAI ResponseUsage payloads."""

    try:
        normalized = _coerce_pydantic_model(usage, openai_models.ResponseUsage)

        cache_read = 0
        if normalized.input_tokens_details:
            cache_read = normalized.input_tokens_details.cached_tokens

        reasoning_tokens = 0
        if normalized.output_tokens_details:
            reasoning_tokens = normalized.output_tokens_details.reasoning_tokens

        return UsageSnapshot(
            input_tokens=normalized.input_tokens,
            output_tokens=normalized.output_tokens,
            cache_read_tokens=cache_read,
            reasoning_tokens=reasoning_tokens,
        )
    except (TypeError, ValueError):
        # Fallback to dictionary-based extraction if the model validation fails
        usage_dict = _coerce_usage_dict(usage)
        input_tokens = usage_dict.get("input_tokens", 0)
        output_tokens = usage_dict.get("output_tokens", 0)

        cache_read = 0
        input_details = usage_dict.get("input_tokens_details", {})
        if isinstance(input_details, dict):
            cache_read = input_details.get("cached_tokens", 0)

        reasoning_tokens = 0
        output_details = usage_dict.get("output_tokens_details", {})
        if isinstance(output_details, dict):
            reasoning_tokens = output_details.get("reasoning_tokens", 0)

        return UsageSnapshot(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            reasoning_tokens=reasoning_tokens,
        )


def openai_completion_usage_snapshot(usage: Any) -> UsageSnapshot:
    """Return a normalized snapshot for OpenAI CompletionUsage payloads."""

    normalized = _coerce_pydantic_model(usage, openai_models.CompletionUsage)

    cache_read = 0
    if normalized.prompt_tokens_details:
        cache_read = normalized.prompt_tokens_details.cached_tokens

    reasoning_tokens = 0
    if normalized.completion_tokens_details:
        reasoning_tokens = normalized.completion_tokens_details.reasoning_tokens

    return UsageSnapshot(
        input_tokens=normalized.prompt_tokens,
        output_tokens=normalized.completion_tokens,
        cache_read_tokens=cache_read,
        reasoning_tokens=reasoning_tokens,
    )


def _coerce_usage_dict(openai_usage: Any) -> dict[str, Any]:
    """Create a dictionary representation of an OpenAI usage payload."""

    if hasattr(openai_usage, "model_dump"):
        return cast(dict[str, Any], openai_usage.model_dump())
    if isinstance(openai_usage, dict):
        return openai_usage

    return {
        "input_tokens": getattr(openai_usage, "input_tokens", None),
        "output_tokens": getattr(openai_usage, "output_tokens", None),
        "prompt_tokens": getattr(openai_usage, "prompt_tokens", None),
        "completion_tokens": getattr(openai_usage, "completion_tokens", None),
        "input_tokens_details": getattr(openai_usage, "input_tokens_details", None),
        "prompt_tokens_details": getattr(openai_usage, "prompt_tokens_details", None),
    }


def openai_usage_to_anthropic_usage(openai_usage: Any | None) -> anthropic_models.Usage:
    """Map OpenAI usage structures to Anthropic Usage with best-effort coverage."""

    if openai_usage is None:
        return anthropic_models.Usage(input_tokens=0, output_tokens=0)

    # CompletionUsage is a special case since it uses prompt_tokens/completion_tokens
    # instead of input_tokens/output_tokens
    if hasattr(openai_usage, "prompt_tokens"):
        # This is likely a CompletionUsage object
        input_tokens = getattr(openai_usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(openai_usage, "completion_tokens", 0) or 0
        cached_tokens = 0

        # Extract cached tokens if available
        prompt_details = getattr(openai_usage, "prompt_tokens_details", None)
        if prompt_details and hasattr(prompt_details, "cached_tokens"):
            cached_tokens = prompt_details.cached_tokens or 0

        return anthropic_models.Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cached_tokens,
        )

    # Try to use the snapshot builders
    snapshot: UsageSnapshot | None = None
    for builder in (openai_response_usage_snapshot, openai_completion_usage_snapshot):
        with contextlib.suppress(TypeError):
            snapshot = builder(openai_usage)
            break

    if snapshot is None:
        usage_dict = _coerce_usage_dict(openai_usage)
        input_candidate: Iterable[str] = ("input_tokens", "prompt_tokens")
        output_candidate: Iterable[str] = ("output_tokens", "completion_tokens")

        input_tokens = next(
            (
                usage_dict[key]
                for key in input_candidate
                if isinstance(usage_dict.get(key), int)
            ),
            0,
        )
        output_tokens = next(
            (
                usage_dict[key]
                for key in output_candidate
                if isinstance(usage_dict.get(key), int)
            ),
            0,
        )

        cached = 0
        details = usage_dict.get("input_tokens_details") or usage_dict.get(
            "prompt_tokens_details"
        )
        if isinstance(details, dict):
            cached = int(details.get("cached_tokens") or 0)
        elif details is not None:
            cached = int(getattr(details, "cached_tokens", 0) or 0)

        snapshot = UsageSnapshot(
            input_tokens=int(input_tokens or 0),
            output_tokens=int(output_tokens or 0),
            cache_read_tokens=cached,
        )

    return anthropic_models.Usage(
        input_tokens=snapshot.input_tokens,
        output_tokens=snapshot.output_tokens,
        cache_read_input_tokens=snapshot.cache_read_tokens,
        cache_creation_input_tokens=snapshot.cache_creation_tokens,
    )


def build_obfuscation_token(
    *, seed: str, sequence: int, payload: str | None = None
) -> str:
    """Return a deterministic obfuscation token mirroring Responses streams."""

    material = f"{seed}:{sequence}:{payload or ''}"
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    token = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return token[:16]


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
    """Parse tool/function arguments as JSON object.

    Raises ValueError for invalid JSON when argument is a string."""
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        if not arguments.strip():
            return {}
        try:
            parsed = json.loads(arguments)
            if not isinstance(parsed, dict):
                return {"value": parsed}
            return parsed
        except json.JSONDecodeError:
            # Test cases expect this to raise an error for invalid JSON
            raise ValueError(f"Invalid JSON in tool arguments: {arguments}")
    return {"arguments": str(arguments)}


__all__ = [
    "UsageSnapshot",
    "anthropic_usage_snapshot",
    "openai_response_usage_snapshot",
    "openai_completion_usage_snapshot",
    "openai_usage_to_anthropic_usage",
    "build_obfuscation_token",
    "map_openai_finish_to_anthropic_stop",
    "strict_parse_tool_arguments",
]
