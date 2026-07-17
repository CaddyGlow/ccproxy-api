"""Compatibility gates between the installed Claude CLI and Agent SDK."""

import pytest
from claude_agent_sdk import AssistantMessage, RateLimitEvent, TextBlock, ThinkingBlock
from claude_agent_sdk._internal.message_parser import parse_message

from ccproxy.plugins.claude_sdk import models as sdk_models
from ccproxy.plugins.claude_sdk.client import ClaudeSDKClient
from ccproxy.plugins.claude_sdk.config import ClaudeSDKSettings, SDKMessageMode
from ccproxy.plugins.claude_sdk.converter import MessageConverter
from ccproxy.plugins.claude_sdk.handler import _select_terminal_assistant_message


@pytest.mark.unit
def test_current_cli_rate_limit_event_is_non_terminal_wire_data() -> None:
    """Usage telemetry must not abort an otherwise valid SDK response stream."""

    message = parse_message(
        {
            "type": "rate_limit_event",
            "uuid": "event-1",
            "session_id": "session-1",
            "rate_limit_info": {
                "status": "allowed",
                "rateLimitType": "five_hour",
                "utilization": 0.25,
            },
        }
    )

    assert isinstance(message, RateLimitEvent)
    assert message.rate_limit_info.status == "allowed"


@pytest.mark.unit
def test_current_sdk_thinking_block_converts_without_losing_final_text() -> None:
    """Extended thinking is valid wire data, not a failed assistant message."""

    message = AssistantMessage(
        content=[
            ThinkingBlock(thinking="private reasoning", signature="signed"),
            TextBlock(text='{"ok":true}'),
        ],
        model="claude-sonnet-4-5",
        parent_tool_use_id=None,
        error=None,
    )

    converted = ClaudeSDKClient(ClaudeSDKSettings())._convert_message(
        message, sdk_models.AssistantMessage
    )

    assert [block.type for block in converted.content] == ["thinking", "text"]
    assert converted.content[1] == sdk_models.TextBlock(text='{"ok":true}')


@pytest.mark.unit
def test_anthropic_compatible_default_does_not_mix_in_sdk_metadata() -> None:
    """SDK lifecycle data is opt-in and cannot inflate normal model output."""

    settings = ClaudeSDKSettings()
    assistant = sdk_models.AssistantMessage(
        content=[
            sdk_models.ThinkingBlock(thinking="private reasoning", signature="signed"),
            sdk_models.TextBlock(text='{"ok":true}'),
        ]
    )
    result = sdk_models.ResultMessage(session_id="session-1")

    response = MessageConverter.convert_to_anthropic_response(
        assistant,
        result,
        "claude-sonnet-4-5",
        mode=settings.sdk_message_mode,
        pretty_format=settings.pretty_format,
    )

    assert settings.include_system_messages_in_stream is False
    assert settings.sdk_message_mode is SDKMessageMode.IGNORE
    assert [
        (block.type, getattr(block, "text", None)) for block in response.content
    ] == [("text", '{"ok":true}')]


@pytest.mark.unit
def test_terminal_assistant_selection_skips_thinking_only_preamble() -> None:
    """The API response comes from the final answer, not a thinking preamble."""

    thinking = sdk_models.AssistantMessage(
        content=[sdk_models.ThinkingBlock(thinking="private", signature="signed")]
    )
    final = sdk_models.AssistantMessage(
        content=[sdk_models.TextBlock(text='{"ok":true}')]
    )

    assert _select_terminal_assistant_message([thinking, final]) is final
