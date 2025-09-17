"""E2E endpoint validation helpers.

Provides validation utilities for end-to-end endpoint testing,
extracted from the original test_endpoint.py script.
"""

import json
from typing import Any

from pydantic import BaseModel, ValidationError

from ccproxy.core.constants import (
    FORMAT_ANTHROPIC_MESSAGES,
    FORMAT_OPENAI_CHAT,
    FORMAT_OPENAI_RESPONSES,
)


# Lazy import functions to avoid circular import issues
def _get_model_class(model_name: str) -> type[BaseModel] | None:
    """Lazy import validation models to avoid circular imports."""
    try:
        if model_name == "MessageResponse":
            from ccproxy.llms.models.anthropic import MessageResponse

            return MessageResponse
        elif model_name == "MessageStartEvent":
            from ccproxy.llms.models.anthropic import MessageStartEvent

            return MessageStartEvent
        elif model_name == "BaseStreamEvent":
            from ccproxy.llms.models.openai import BaseStreamEvent

            return BaseStreamEvent
        elif model_name == "ChatCompletionChunk":
            from ccproxy.llms.models.openai import ChatCompletionChunk

            return ChatCompletionChunk
        elif model_name == "ChatCompletionResponse":
            from ccproxy.llms.models.openai import ChatCompletionResponse

            return ChatCompletionResponse
        elif model_name == "ResponseMessage":
            from ccproxy.llms.models.openai import ResponseMessage

            return ResponseMessage
        elif model_name == "ResponseObject":
            from ccproxy.llms.models.openai import ResponseObject

            return ResponseObject
    except ImportError:
        pass
    return None


def validate_sse_event(event: str) -> bool:
    """Validate SSE event structure (basic check)."""
    return event.startswith("data: ")


def validate_response_with_model(
    response: dict[str, Any],
    model_class: type[BaseModel] | None,
    is_streaming: bool = False,
) -> tuple[bool, str]:
    """Validate response using the provided model_class.

    Returns:
        Tuple of (is_valid, error_message)
    """
    if model_class is None:
        return True, ""

    try:
        # Special handling for ResponseMessage: extract assistant message
        if model_class.__name__ == "ResponseMessage":
            payload = _extract_openai_responses_message(response)
        else:
            payload = response

        model_class.model_validate(payload)
        return True, ""
    except ValidationError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Validation error: {e}"


def validate_stream_chunk(
    chunk: dict[str, Any], chunk_model_class: type[BaseModel] | None
) -> tuple[bool, str]:
    """Validate a streaming chunk using the provided chunk_model_class.

    Returns:
        Tuple of (is_valid, error_message)
    """
    if chunk_model_class is None:
        return True, ""

    try:
        chunk_model_class.model_validate(chunk)
        return True, ""
    except ValidationError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Chunk validation error: {e}"


def parse_streaming_events(content: str) -> list[dict[str, Any]]:
    """Parse streaming content into list of event data.

    Args:
        content: Raw SSE content

    Returns:
        List of parsed JSON objects from data events
    """
    events = []
    lines = content.split("\n")

    for line in lines:
        line = line.strip()
        if line.startswith("data: ") and not line.endswith("[DONE]"):
            try:
                data_content = line[6:]  # Remove "data: " prefix
                event_data = json.loads(data_content)
                events.append(event_data)
            except json.JSONDecodeError:
                continue

    return events


def validate_streaming_response_structure(
    content: str, format_type: str, chunk_model_class: type[BaseModel] | None = None
) -> tuple[bool, list[str]]:
    """Validate the structure of a streaming response."""
    errors = []

    # Basic SSE format check
    if "data: " not in content:
        errors.append("No SSE data events found")
        return False, errors

    # Parse events
    events = parse_streaming_events(content)
    if not events:
        errors.append("No valid JSON events found in stream")
        return False, errors

    # Validate chunk structure if model provided
    if chunk_model_class:
        for i, event in enumerate(events):
            is_valid, error = validate_stream_chunk(event, chunk_model_class)
            if not is_valid:
                errors.append(f"Event {i} validation failed: {error}")

    normalized = _normalize_format(format_type)

    # Format-specific validations
    if normalized == "openai":
        _validate_openai_streaming_events(events, errors)
    elif normalized == "anthropic":
        _validate_anthropic_streaming_events(events, errors)
    elif normalized == "response_api":
        _validate_response_api_streaming_events(events, errors)

    return len(errors) == 0, errors


def _validate_openai_streaming_events(
    events: list[dict[str, Any]], errors: list[str]
) -> None:
    """Validate OpenAI streaming events structure."""
    for event in events:
        if not isinstance(event.get("choices"), list):
            errors.append("OpenAI stream event missing choices array")
            continue

        if event["choices"] and "delta" not in event["choices"][0]:
            errors.append("OpenAI stream event missing delta in choice")


def _validate_anthropic_streaming_events(
    events: list[dict[str, Any]], errors: list[str]
) -> None:
    """Validate Anthropic streaming events structure."""
    # Look for message_start, content_block events
    event_types = [event.get("type") for event in events]
    if "message_start" not in event_types:
        errors.append("Anthropic stream missing message_start event")


def _validate_response_api_streaming_events(
    events: list[dict[str, Any]], errors: list[str]
) -> None:
    """Validate Response API streaming events structure."""
    # Response API events should have specific structure
    for event in events:
        if "event" in event or "type" in event:
            continue  # Valid event structure
        else:
            errors.append("Response API event missing event/type field")
            break


def _extract_openai_responses_message(response: dict[str, Any]) -> dict[str, Any]:
    """Coerce various response shapes into an OpenAIResponseMessage dict.

    Supports:
    - Chat Completions: { choices: [{ message: {...} }] }
    - Responses API (non-stream): { output: [ { type: 'message', content: [...] } ] }
    """
    # Case 1: Chat Completions format
    try:
        if isinstance(response, dict) and "choices" in response:
            choices = response.get("choices") or []
            if choices and isinstance(choices[0], dict):
                msg = choices[0].get("message")
                if isinstance(msg, dict):
                    return msg
    except Exception:
        pass

    # Case 2: Responses API-like format with output message
    try:
        output = response.get("output") if isinstance(response, dict) else None
        if isinstance(output, list):
            for item in output:
                if isinstance(item, dict) and item.get("type") == "message":
                    content_blocks = item.get("content") or []
                    text_parts: list[str] = []
                    for block in content_blocks:
                        if (
                            isinstance(block, dict)
                            and block.get("type") in ("text", "output_text")
                            and block.get("text")
                        ):
                            text_parts.append(block["text"])
                    content_text = "".join(text_parts) if text_parts else None
                    return {"role": "assistant", "content": content_text}
    except Exception:
        pass

    # Fallback: empty assistant message
    return {"role": "assistant", "content": None}


def get_validation_model_for_format(
    format_type: str, is_streaming: bool = False
) -> type[BaseModel] | None:
    """Get the appropriate validation model class for a format type.

    Args:
        format_type: The API format (openai, anthropic, response_api, codex)
        is_streaming: Whether this is for streaming validation

    Returns:
        Model class for validation or None if not available
    """
    normalized = _normalize_format(format_type)

    if is_streaming:
        model_name_map = {
            "openai": "ChatCompletionChunk",
            "anthropic": "MessageStartEvent",
            "response_api": "BaseStreamEvent",
            "codex": "ChatCompletionChunk",
        }
    else:
        model_name_map = {
            "openai": "ChatCompletionResponse",
            "anthropic": "MessageResponse",
            "response_api": "ResponseObject",
            "codex": "ChatCompletionResponse",
        }

    model_name = model_name_map.get(normalized)
    if model_name:
        return _get_model_class(model_name)
    return None


# Format normalization helper
def _normalize_format(format_type: str) -> str:
    alias_map = {
        FORMAT_OPENAI_CHAT: "openai",
        FORMAT_OPENAI_RESPONSES: "response_api",
        FORMAT_ANTHROPIC_MESSAGES: "anthropic",
        "openai": "openai",
        "response_api": "response_api",
        "anthropic": "anthropic",
        "codex": "codex",
    }
    return alias_map.get(format_type, format_type)
