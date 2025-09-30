"""Tests covering usage snapshot helpers and related conversions."""

from ccproxy.llms.formatters.anthropic_to_openai import (
    convert__anthropic_usage_to_openai_completion__usage,
)
from ccproxy.llms.formatters.openai_to_anthropic import (
    convert__openai_responses_usage_to_anthropic__usage,
    convert__openai_responses_usage_to_openai_completion__usage,
)
from ccproxy.llms.formatters.utils import (
    UsageSnapshot,
    anthropic_usage_snapshot,
    openai_response_usage_snapshot,
)
from ccproxy.llms.models import anthropic as anthropic_models
from ccproxy.llms.models import openai as openai_models


def test_anthropic_usage_snapshot_prefers_cache_creation() -> None:
    usage = anthropic_models.Usage(
        input_tokens=120,
        output_tokens=45,
        cache_read_input_tokens=None,
        cache_creation_input_tokens=None,
        cache_creation=anthropic_models.CacheCreation(
            ephemeral_1h_input_tokens=10,
            ephemeral_5m_input_tokens=5,
        ),
    )

    snapshot = anthropic_usage_snapshot(usage)

    assert snapshot == UsageSnapshot(
        input_tokens=120,
        output_tokens=45,
        cache_read_tokens=0,
        cache_creation_tokens=15,
        reasoning_tokens=0,
    )


def test_openai_response_usage_snapshot_extracts_reasoning() -> None:
    response_usage = openai_models.ResponseUsage(
        input_tokens=80,
        input_tokens_details=openai_models.InputTokensDetails(cached_tokens=7),
        output_tokens=32,
        output_tokens_details=openai_models.OutputTokensDetails(reasoning_tokens=4),
        total_tokens=112,
    )

    snapshot = openai_response_usage_snapshot(response_usage)

    assert snapshot == UsageSnapshot(
        input_tokens=80,
        output_tokens=32,
        cache_read_tokens=7,
        cache_creation_tokens=0,
        reasoning_tokens=4,
    )


def test_convert_anthropic_usage_to_openai_completion_uses_cache_creation() -> None:
    usage = anthropic_models.Usage(
        input_tokens=150,
        output_tokens=50,
        cache_read_input_tokens=None,
        cache_creation_input_tokens=20,
    )

    converted = convert__anthropic_usage_to_openai_completion__usage(usage)

    assert converted.prompt_tokens == 150
    assert converted.completion_tokens == 50
    assert converted.prompt_tokens_details is not None
    assert converted.prompt_tokens_details.cached_tokens == 20


def test_convert_openai_response_usage_to_completion_preserves_reasoning() -> None:
    response_usage = openai_models.ResponseUsage(
        input_tokens=42,
        input_tokens_details=openai_models.InputTokensDetails(cached_tokens=3),
        output_tokens=21,
        output_tokens_details=openai_models.OutputTokensDetails(reasoning_tokens=9),
        total_tokens=63,
    )

    converted = convert__openai_responses_usage_to_openai_completion__usage(
        response_usage
    )

    assert converted.prompt_tokens == 42
    assert converted.completion_tokens == 21
    assert converted.prompt_tokens_details is not None
    assert converted.prompt_tokens_details.cached_tokens == 3
    assert converted.completion_tokens_details is not None
    assert converted.completion_tokens_details.reasoning_tokens == 9


def test_roundtrip_openai_response_usage_to_anthropic_usage() -> None:
    response_usage = openai_models.ResponseUsage(
        input_tokens=18,
        input_tokens_details=openai_models.InputTokensDetails(cached_tokens=2),
        output_tokens=8,
        output_tokens_details=openai_models.OutputTokensDetails(reasoning_tokens=0),
        total_tokens=26,
    )

    anthropic_usage = convert__openai_responses_usage_to_anthropic__usage(
        response_usage
    )

    assert anthropic_usage.input_tokens == 18
    assert anthropic_usage.output_tokens == 8
    assert anthropic_usage.cache_read_input_tokens == 2
    assert anthropic_usage.cache_creation_input_tokens == 0
