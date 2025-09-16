"""Usage conversion helpers shared across adapters."""

from __future__ import annotations

import ccproxy.llms.anthropic.models as antrhopic_models
from ccproxy.llms.openai.models import (
    CompletionTokensDetails,
    CompletionUsage,
    InputTokensDetails,
    OutputTokensDetails,
    PromptTokensDetails,
    ResponseUsage,
)


def convert_anthropic_usage_to_openai_completion_usage(
    usage: antrhopic_models.Usage,
) -> CompletionUsage:
    """Convert Anthropic usage payloads to OpenAI completion usage objects."""

    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)

    cached_tokens = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    cache_creation_tokens = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
    if cache_creation_tokens > 0 and cached_tokens == 0:
        cached_tokens = cache_creation_tokens

    prompt_tokens_details = PromptTokensDetails(cached_tokens=cached_tokens, audio_tokens=0)
    completion_tokens_details = CompletionTokensDetails(
        reasoning_tokens=0,
        audio_tokens=0,
        accepted_prediction_tokens=0,
        rejected_prediction_tokens=0,
    )

    return CompletionUsage(
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        prompt_tokens_details=prompt_tokens_details,
        completion_tokens_details=completion_tokens_details,
    )


def convert_anthropic_usage_to_openai_response_usage(
    usage: antrhopic_models.Usage,
) -> ResponseUsage:
    """Convert Anthropic usage payloads to OpenAI ResponseUsage objects."""

    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)

    cached_tokens = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    cache_creation_tokens = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
    if cache_creation_tokens > 0 and cached_tokens == 0:
        cached_tokens = cache_creation_tokens

    input_tokens_details = InputTokensDetails(cached_tokens=cached_tokens)
    output_tokens_details = OutputTokensDetails(reasoning_tokens=0)

    return ResponseUsage(
        input_tokens=input_tokens,
        input_tokens_details=input_tokens_details,
        output_tokens=output_tokens,
        output_tokens_details=output_tokens_details,
        total_tokens=input_tokens + output_tokens,
    )


def convert_openai_completion_usage_to_anthropic_usage(
    usage: CompletionUsage,
) -> antrhopic_models.Usage:
    """Convert OpenAI completion usage payloads to Anthropic usage models."""

    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

    cache_read_tokens = 0
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    if prompt_details:
        cache_read_tokens = int(getattr(prompt_details, "cached_tokens", 0) or 0)

    anthropic_usage = antrhopic_models.Usage(
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
    )

    if cache_read_tokens > 0:
        anthropic_usage.cache_read_input_tokens = cache_read_tokens

    return anthropic_usage


def convert_openai_response_usage_to_anthropic_usage(
    usage: ResponseUsage,
) -> antrhopic_models.Usage:
    """Convert OpenAI ResponseUsage payloads to Anthropic usage models."""

    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)

    cache_read_tokens = 0
    input_details = getattr(usage, "input_tokens_details", None)
    if input_details:
        cache_read_tokens = int(getattr(input_details, "cached_tokens", 0) or 0)

    anthropic_usage = antrhopic_models.Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    if cache_read_tokens > 0:
        anthropic_usage.cache_read_input_tokens = cache_read_tokens

    return anthropic_usage


def convert_openai_response_usage_to_openai_completion_usage(
    usage: ResponseUsage,
) -> CompletionUsage:
    """Convert OpenAI Response API usage to OpenAI Chat completion usage."""

    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)

    # Extract cached tokens from input details
    cached_tokens = 0
    input_details = getattr(usage, "input_tokens_details", None)
    if input_details:
        cached_tokens = int(getattr(input_details, "cached_tokens", 0) or 0)

    # Extract reasoning tokens from output details
    reasoning_tokens = 0
    output_details = getattr(usage, "output_tokens_details", None)
    if output_details:
        reasoning_tokens = int(getattr(output_details, "reasoning_tokens", 0) or 0)

    prompt_tokens_details = PromptTokensDetails(cached_tokens=cached_tokens, audio_tokens=0)
    completion_tokens_details = CompletionTokensDetails(
        reasoning_tokens=reasoning_tokens,
        audio_tokens=0,
        accepted_prediction_tokens=0,
        rejected_prediction_tokens=0,
    )

    return CompletionUsage(
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        prompt_tokens_details=prompt_tokens_details,
        completion_tokens_details=completion_tokens_details,
    )


def convert_openai_completion_usage_to_openai_response_usage(
    usage: CompletionUsage,
) -> ResponseUsage:
    """Convert OpenAI Chat completion usage to OpenAI Response API usage."""

    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

    # Extract cached tokens from prompt details
    cached_tokens = 0
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    if prompt_details:
        cached_tokens = int(getattr(prompt_details, "cached_tokens", 0) or 0)

    # Extract reasoning tokens from completion details
    reasoning_tokens = 0
    completion_details = getattr(usage, "completion_tokens_details", None)
    if completion_details:
        reasoning_tokens = int(getattr(completion_details, "reasoning_tokens", 0) or 0)

    input_tokens_details = InputTokensDetails(cached_tokens=cached_tokens)
    output_tokens_details = OutputTokensDetails(reasoning_tokens=reasoning_tokens)

    return ResponseUsage(
        input_tokens=prompt_tokens,
        input_tokens_details=input_tokens_details,
        output_tokens=completion_tokens,
        output_tokens_details=output_tokens_details,
        total_tokens=prompt_tokens + completion_tokens,
    )


__all__ = [
    "convert_anthropic_usage_to_openai_completion_usage",
    "convert_anthropic_usage_to_openai_response_usage",
    "convert_openai_completion_usage_to_anthropic_usage",
    "convert_openai_response_usage_to_anthropic_usage",
    "convert_openai_response_usage_to_openai_completion_usage",
    "convert_openai_completion_usage_to_openai_response_usage",
]
