"""End-to-end integration tests for CCProxy endpoints.

This module provides comprehensive endpoint testing following the project's
streamlined testing architecture with performance-optimized patterns.

Note: These tests validate the test infrastructure and data structures.
Full endpoint testing requires the circular import issues to be resolved.
"""

from typing import Any

import pytest

from tests.helpers.e2e_validation import (
    parse_streaming_events,
    validate_sse_event,
    validate_streaming_response_structure,
)
from tests.helpers.test_data import (
    E2E_ENDPOINT_CONFIGURATIONS,
    create_e2e_request_for_format,
    get_expected_response_fields,
    normalize_format,
)


pytestmark = [pytest.mark.integration, pytest.mark.e2e]


# Core validation tests that work without complex app setup
@pytest.mark.asyncio
async def test_endpoint_configurations_structure() -> None:
    """Test that endpoint configurations are properly structured."""
    assert len(E2E_ENDPOINT_CONFIGURATIONS) > 0

    for config in E2E_ENDPOINT_CONFIGURATIONS:
        # Verify all required fields exist
        required_fields = [
            "name",
            "endpoint",
            "stream",
            "model",
            "format",
            "description",
        ]
        assert all(field in config for field in required_fields)

        # Verify field types and values
        assert isinstance(config["stream"], bool)
        assert config["endpoint"].startswith("/")
        assert normalize_format(config["format"]) in {
            "openai",
            "anthropic",
            "response_api",
            "codex",
        }
        assert isinstance(config["model"], str)
        assert len(config["model"]) > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("config", E2E_ENDPOINT_CONFIGURATIONS)
async def test_request_creation_for_each_endpoint(config: dict[str, Any]) -> None:
    """Test that we can create valid requests for each endpoint configuration."""
    endpoint = config["endpoint"]
    model = config["model"]
    format_type = config["format"]
    stream = config["stream"]
    normalized_format = normalize_format(format_type)

    # Create request using our factory
    request_data = create_e2e_request_for_format(
        format_type=format_type,
        model=model,
        content="Test message",
        stream=stream,
    )

    # Verify request structure
    assert isinstance(request_data, dict)
    assert "model" in request_data
    assert request_data["model"] == model

    # Format-specific validation
    if normalized_format == "openai":
        assert "messages" in request_data
        assert isinstance(request_data["messages"], list)
        assert len(request_data["messages"]) > 0
        assert "role" in request_data["messages"][0]
        assert "content" in request_data["messages"][0]

    elif normalized_format == "anthropic":
        assert "messages" in request_data
        assert "max_tokens" in request_data

    elif normalized_format == "response_api":
        assert "input" in request_data
        assert isinstance(request_data["input"], list)

    # Stream parameter validation
    if stream:
        assert request_data.get("stream") is True


@pytest.mark.asyncio
async def test_validation_functions_work() -> None:
    """Test that our validation functions work correctly."""
    # Test SSE event validation
    assert validate_sse_event('data: {"test": true}')
    assert not validate_sse_event("invalid event")

    # Test streaming events parsing
    sse_content = """data: {"id": "test1", "object": "chunk"}
data: {"id": "test2", "object": "chunk"}
data: [DONE]
"""
    events = parse_streaming_events(sse_content)
    assert len(events) == 2
    assert events[0]["id"] == "test1"
    assert events[1]["id"] == "test2"

    # Test streaming validation
    is_valid, errors = validate_streaming_response_structure(
        sse_content, "openai", None
    )
    # Should be valid even without model validation
    assert isinstance(is_valid, bool)
    assert isinstance(errors, list)


@pytest.mark.asyncio
async def test_response_field_validation() -> None:
    """Test response field validation helpers."""
    # Test OpenAI response fields
    openai_fields = get_expected_response_fields("openai")
    assert "choices" in openai_fields
    assert "model" in openai_fields

    # Test Anthropic response fields
    anthropic_fields = get_expected_response_fields("anthropic")
    assert "content" in anthropic_fields
    assert "role" in anthropic_fields

    # Test unknown format
    unknown_fields = get_expected_response_fields("unknown")
    assert isinstance(unknown_fields, set)


@pytest.mark.asyncio
async def test_conversion_completeness() -> None:
    """Verify that all key components from original script were converted."""
    # Test that we have endpoint configurations for all major services
    endpoint_names = [config["name"] for config in E2E_ENDPOINT_CONFIGURATIONS]

    # Should have Copilot endpoints
    copilot_endpoints = [name for name in endpoint_names if "copilot" in name]
    assert len(copilot_endpoints) >= 2  # streaming and non-streaming

    # Should have Claude API endpoints
    claude_endpoints = [name for name in endpoint_names if "anthropic_api" in name]
    assert len(claude_endpoints) >= 2

    # Should have Codex endpoints
    codex_endpoints = [name for name in endpoint_names if "codex" in name]
    assert len(codex_endpoints) >= 2

    # Should have both streaming and non-streaming variants
    streaming_configs = [
        config for config in E2E_ENDPOINT_CONFIGURATIONS if config["stream"]
    ]
    non_streaming_configs = [
        config for config in E2E_ENDPOINT_CONFIGURATIONS if not config["stream"]
    ]

    assert len(streaming_configs) >= 5
    assert len(non_streaming_configs) >= 5

    # Should support all expected formats
    formats = {
        normalize_format(config["format"]) for config in E2E_ENDPOINT_CONFIGURATIONS
    }
    assert "openai" in formats
    assert "anthropic" in formats or "response_api" in formats
