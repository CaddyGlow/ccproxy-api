"""Shared token usage conversion helpers for formatter adapters."""

from __future__ import annotations

from typing import Any

from ccproxy.llms.formatters.utils import (
    anthropic_usage_snapshot,
    openai_completion_usage_snapshot,
    openai_response_usage_snapshot,
)
from ccproxy.llms.models import anthropic as anthropic_models
from ccproxy.llms.models import openai as openai_models


def convert_openai_responses_usage_to_completion_usage(
    usage: Any,
) -> openai_models.CompletionUsage:
    """Normalize Responses usage into the legacy CompletionUsage envelope."""

    snapshot = openai_response_usage_snapshot(usage)

    prompt_tokens_details = openai_models.PromptTokensDetails(
        cached_tokens=snapshot.cache_read_tokens,
        audio_tokens=0,
    )
    completion_tokens_details = openai_models.CompletionTokensDetails(
        reasoning_tokens=snapshot.reasoning_tokens,
        audio_tokens=0,
        accepted_prediction_tokens=0,
        rejected_prediction_tokens=0,
    )

    return openai_models.CompletionUsage(
        prompt_tokens=snapshot.input_tokens,
        completion_tokens=snapshot.output_tokens,
        total_tokens=snapshot.input_tokens + snapshot.output_tokens,
        prompt_tokens_details=prompt_tokens_details,
        completion_tokens_details=completion_tokens_details,
    )


def convert_openai_completion_usage_to_responses_usage(
    usage: Any,
) -> openai_models.ResponseUsage:
    """Map Completion usage payloads into Responses Usage structures."""

    snapshot = openai_completion_usage_snapshot(usage)

    input_tokens_details = openai_models.InputTokensDetails(
        cached_tokens=snapshot.cache_read_tokens
    )
    output_tokens_details = openai_models.OutputTokensDetails(
        reasoning_tokens=snapshot.reasoning_tokens
    )

    return openai_models.ResponseUsage(
        input_tokens=snapshot.input_tokens,
        input_tokens_details=input_tokens_details,
        output_tokens=snapshot.output_tokens,
        output_tokens_details=output_tokens_details,
        total_tokens=snapshot.input_tokens + snapshot.output_tokens,
    )


def convert_openai_responses_usage_to_anthropic_usage(
    usage: Any,
) -> anthropic_models.Usage:
    """Translate OpenAI Responses usage into Anthropic Usage models."""

    snapshot = openai_response_usage_snapshot(usage)

    return anthropic_models.Usage(
        input_tokens=snapshot.input_tokens,
        output_tokens=snapshot.output_tokens,
        cache_read_input_tokens=snapshot.cache_read_tokens,
        cache_creation_input_tokens=snapshot.cache_creation_tokens,
    )


def convert_anthropic_usage_to_openai_completion_usage(
    usage: Any,
) -> openai_models.CompletionUsage:
    """Translate Anthropic Usage values into OpenAI Completion usage."""

    snapshot = anthropic_usage_snapshot(usage)
    cached_tokens = snapshot.cache_read_tokens or snapshot.cache_creation_tokens

    prompt_tokens_details = openai_models.PromptTokensDetails(
        cached_tokens=cached_tokens,
        audio_tokens=0,
    )
    completion_tokens_details = openai_models.CompletionTokensDetails(
        reasoning_tokens=0,
        audio_tokens=0,
        accepted_prediction_tokens=0,
        rejected_prediction_tokens=0,
    )

    return openai_models.CompletionUsage(
        prompt_tokens=snapshot.input_tokens,
        completion_tokens=snapshot.output_tokens,
        total_tokens=snapshot.input_tokens + snapshot.output_tokens,
        prompt_tokens_details=prompt_tokens_details,
        completion_tokens_details=completion_tokens_details,
    )


def convert_anthropic_usage_to_openai_responses_usage(
    usage: Any,
) -> openai_models.ResponseUsage:
    """Translate Anthropic Usage values into OpenAI Responses usage."""

    snapshot = anthropic_usage_snapshot(usage)
    cached_tokens = snapshot.cache_read_tokens or snapshot.cache_creation_tokens

    input_tokens_details = openai_models.InputTokensDetails(cached_tokens=cached_tokens)
    output_tokens_details = openai_models.OutputTokensDetails(reasoning_tokens=0)

    return openai_models.ResponseUsage(
        input_tokens=snapshot.input_tokens,
        input_tokens_details=input_tokens_details,
        output_tokens=snapshot.output_tokens,
        output_tokens_details=output_tokens_details,
        total_tokens=snapshot.input_tokens + snapshot.output_tokens,
    )


__all__ = [
    "convert_anthropic_usage_to_openai_completion_usage",
    "convert_anthropic_usage_to_openai_responses_usage",
    "convert_openai_completion_usage_to_responses_usage",
    "convert_openai_responses_usage_to_anthropic_usage",
    "convert_openai_responses_usage_to_completion_usage",
]
