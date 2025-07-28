"""Tests for Claude SDK XML parser module."""

import json
from typing import Any

from ccproxy.claude_sdk.parser import (
    parse_formatted_sdk_content,
    parse_result_message_tags,
    parse_system_message_tags,
    parse_text_tags,
    parse_tool_result_sdk_tags,
    parse_tool_use_sdk_tags,
)


class TestParseSystemMessageTags:
    """Test system_message XML tag parsing."""

    def test_parse_valid_system_message(self) -> None:
        """Test parsing valid system message XML."""
        system_data = {"source": "claude_code_sdk", "text": "System message content"}
        xml_content = f"<system_message>{json.dumps(system_data)}</system_message>"

        result = parse_system_message_tags(xml_content)

        assert result == "[claude_code_sdk]: System message content"

    def test_parse_system_message_default_source(self) -> None:
        """Test system message parsing with default source."""
        system_data = {"text": "System message content"}
        xml_content = f"<system_message>{json.dumps(system_data)}</system_message>"

        result = parse_system_message_tags(xml_content)

        assert result == "[claude_code_sdk]: System message content"

    def test_parse_system_message_with_surrounding_text(self) -> None:
        """Test system message parsing with surrounding text."""
        system_data = {"text": "System message"}
        xml_content = (
            f"Before <system_message>{json.dumps(system_data)}</system_message> After"
        )

        result = parse_system_message_tags(xml_content)

        assert result == "Before [claude_code_sdk]: System message After"

    def test_parse_multiple_system_messages(self) -> None:
        """Test parsing multiple system messages."""
        system_data1 = {"text": "First message"}
        system_data2 = {"text": "Second message"}
        xml_content = (
            f"<system_message>{json.dumps(system_data1)}</system_message>"
            f" and <system_message>{json.dumps(system_data2)}</system_message>"
        )

        result = parse_system_message_tags(xml_content)

        assert (
            result
            == "[claude_code_sdk]: First message and [claude_code_sdk]: Second message"
        )

    def test_parse_invalid_json_system_message(self) -> None:
        """Test system message parsing with invalid JSON."""
        xml_content = "<system_message>invalid json</system_message>"

        result = parse_system_message_tags(xml_content)

        # Should keep original when JSON parsing fails
        assert result == "<system_message>invalid json</system_message>"

    def test_parse_no_system_messages(self) -> None:
        """Test parsing text without system messages."""
        text = "Regular text without any XML tags"

        result = parse_system_message_tags(text)

        assert result == text


class TestParseToolUseSdkTags:
    """Test tool_use_sdk XML tag parsing."""

    def test_parse_tool_use_for_streaming(self) -> None:
        """Test tool_use parsing for streaming (no tool call collection)."""
        tool_data = {"id": "tool_123", "name": "search", "input": {"query": "test"}}
        xml_content = f"<tool_use_sdk>{json.dumps(tool_data)}</tool_use_sdk>"

        result_text, tool_calls = parse_tool_use_sdk_tags(
            xml_content, collect_tool_calls=False
        )

        expected_text = '[claude_code_sdk tool_use tool_123]: search({"query": "test"})'
        assert result_text == expected_text
        assert tool_calls == []

    def test_parse_tool_use_for_openai_adapter(self) -> None:
        """Test tool_use parsing for OpenAI adapter (collect tool calls)."""
        tool_data = {"id": "tool_123", "name": "search", "input": {"query": "test"}}
        xml_content = f"<tool_use_sdk>{json.dumps(tool_data)}</tool_use_sdk>"

        result_text, tool_calls = parse_tool_use_sdk_tags(
            xml_content, collect_tool_calls=True
        )

        # Text should be empty when collecting tool calls
        assert result_text == ""
        assert len(tool_calls) == 1

        # Tool call should be in OpenAI format
        tool_call = tool_calls[0]
        assert tool_call.type == "function"
        assert tool_call.id == "tool_123"
        assert tool_call.function.name == "search"
        assert tool_call.function.arguments == '{"query": "test"}'

    def test_parse_multiple_tool_uses(self) -> None:
        """Test parsing multiple tool_use tags."""
        tool_data1 = {"id": "tool_1", "name": "search", "input": {"q": "test1"}}
        tool_data2 = {"id": "tool_2", "name": "calculate", "input": {"expr": "2+2"}}
        xml_content = (
            f"<tool_use_sdk>{json.dumps(tool_data1)}</tool_use_sdk>"
            f" and <tool_use_sdk>{json.dumps(tool_data2)}</tool_use_sdk>"
        )

        result_text, tool_calls = parse_tool_use_sdk_tags(
            xml_content, collect_tool_calls=True
        )

        assert result_text == " and "  # Text between tool uses should remain
        assert len(tool_calls) == 2
        assert tool_calls[0].id == "tool_1"
        assert tool_calls[1].id == "tool_2"

    def test_parse_tool_use_invalid_json(self) -> None:
        """Test tool_use parsing with invalid JSON."""
        xml_content = "<tool_use_sdk>invalid json</tool_use_sdk>"

        result_text, tool_calls = parse_tool_use_sdk_tags(
            xml_content, collect_tool_calls=False
        )

        # Should keep original when JSON parsing fails
        assert result_text == "<tool_use_sdk>invalid json</tool_use_sdk>"
        assert tool_calls == []

    def test_parse_tool_use_empty_input(self) -> None:
        """Test tool_use parsing with empty input."""
        tool_data = {"id": "tool_123", "name": "ping", "input": {}}
        xml_content = f"<tool_use_sdk>{json.dumps(tool_data)}</tool_use_sdk>"

        result_text, tool_calls = parse_tool_use_sdk_tags(
            xml_content, collect_tool_calls=False
        )

        assert "[claude_code_sdk tool_use tool_123]: ping({})" in result_text


class TestParseToolResultSdkTags:
    """Test tool_result_sdk XML tag parsing."""

    def test_parse_tool_result_success(self) -> None:
        """Test parsing successful tool result."""
        result_data = {
            "tool_use_id": "tool_123",
            "content": "Search completed successfully",
            "is_error": False,
        }
        xml_content = f"<tool_result_sdk>{json.dumps(result_data)}</tool_result_sdk>"

        result = parse_tool_result_sdk_tags(xml_content)

        expected = (
            "[claude_code_sdk tool_result tool_123]: Search completed successfully"
        )
        assert result == expected

    def test_parse_tool_result_error(self) -> None:
        """Test parsing error tool result."""
        result_data = {
            "tool_use_id": "tool_123",
            "content": "Search failed: invalid query",
            "is_error": True,
        }
        xml_content = f"<tool_result_sdk>{json.dumps(result_data)}</tool_result_sdk>"

        result = parse_tool_result_sdk_tags(xml_content)

        expected = "[claude_code_sdk tool_result tool_123 (ERROR)]: Search failed: invalid query"
        assert result == expected

    def test_parse_tool_result_default_error_status(self) -> None:
        """Test tool result parsing with default error status."""
        result_data = {"tool_use_id": "tool_123", "content": "Result content"}
        xml_content = f"<tool_result_sdk>{json.dumps(result_data)}</tool_result_sdk>"

        result = parse_tool_result_sdk_tags(xml_content)

        expected = "[claude_code_sdk tool_result tool_123]: Result content"
        assert result == expected

    def test_parse_tool_result_invalid_json(self) -> None:
        """Test tool result parsing with invalid JSON."""
        xml_content = "<tool_result_sdk>invalid json</tool_result_sdk>"

        result = parse_tool_result_sdk_tags(xml_content)

        # Should keep original when JSON parsing fails
        assert result == "<tool_result_sdk>invalid json</tool_result_sdk>"


class TestParseResultMessageTags:
    """Test result_message XML tag parsing."""

    def test_parse_result_message_complete(self) -> None:
        """Test parsing complete result message."""
        result_data = {
            "source": "claude_code_sdk",
            "session_id": "session_123",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 20},
            "total_cost_usd": 0.001,
        }
        xml_content = f"<result_message>{json.dumps(result_data)}</result_message>"

        result = parse_result_message_tags(xml_content)

        expected = (
            "[claude_code_sdk result session_123]: "
            "stop_reason=end_turn, usage={'input_tokens': 10, 'output_tokens': 20}, "
            "cost_usd=0.001"
        )
        assert result == expected

    def test_parse_result_message_without_cost(self) -> None:
        """Test parsing result message without cost information."""
        result_data = {
            "source": "claude_code_sdk",
            "session_id": "session_123",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }
        xml_content = f"<result_message>{json.dumps(result_data)}</result_message>"

        result = parse_result_message_tags(xml_content)

        expected = (
            "[claude_code_sdk result session_123]: "
            "stop_reason=end_turn, usage={'input_tokens': 10, 'output_tokens': 20}"
        )
        assert result == expected

    def test_parse_result_message_defaults(self) -> None:
        """Test result message parsing with default values."""
        result_data: dict[str, Any] = {}
        xml_content = f"<result_message>{json.dumps(result_data)}</result_message>"

        result = parse_result_message_tags(xml_content)

        expected = "[claude_code_sdk result ]: stop_reason=, usage={}"
        assert result == expected


class TestParseTextTags:
    """Test text XML tag parsing."""

    def test_parse_text_tags_basic(self) -> None:
        """Test basic text tag parsing."""
        xml_content = "<text>Hello, world!</text>"

        result = parse_text_tags(xml_content)

        assert result == "Hello, world!"

    def test_parse_text_tags_with_newlines(self) -> None:
        """Test text tag parsing with newlines."""
        xml_content = "<text>\nHello, world!\n</text>"

        result = parse_text_tags(xml_content)

        assert result == "Hello, world!"

    def test_parse_text_tags_multiline(self) -> None:
        """Test text tag parsing with multiline content."""
        xml_content = "<text>\nLine 1\nLine 2\nLine 3\n</text>"

        result = parse_text_tags(xml_content)

        assert result == "Line 1\nLine 2\nLine 3"

    def test_parse_multiple_text_tags(self) -> None:
        """Test parsing multiple text tags."""
        xml_content = "Before <text>First</text> Middle <text>Second</text> After"

        result = parse_text_tags(xml_content)

        assert result == "Before First Middle Second After"

    def test_parse_nested_text_content(self) -> None:
        """Test text tag parsing with nested XML-like content."""
        xml_content = "<text>Content with <inner>nested</inner> tags</text>"

        result = parse_text_tags(xml_content)

        assert result == "Content with <inner>nested</inner> tags"


class TestParseFormattedSdkContent:
    """Test the main parsing function."""

    def test_parse_empty_content(self) -> None:
        """Test parsing empty content."""
        result_text, tool_calls = parse_formatted_sdk_content(
            "", collect_tool_calls=False
        )

        assert result_text == ""
        assert tool_calls == []

    def test_parse_mixed_content_streaming(self) -> None:
        """Test parsing mixed SDK content for streaming."""
        system_data = {"text": "System message"}
        tool_data = {"id": "tool_1", "name": "search", "input": {"q": "test"}}
        result_data = {"session_id": "sess_1", "stop_reason": "end_turn", "usage": {}}

        xml_content = (
            f"<system_message>{json.dumps(system_data)}</system_message>\n"
            f"<text>Some text content</text>\n"
            f"<tool_use_sdk>{json.dumps(tool_data)}</tool_use_sdk>\n"
            f"<tool_result_sdk>{json.dumps({'tool_use_id': 'tool_1', 'content': 'result'})}</tool_result_sdk>\n"
            f"<result_message>{json.dumps(result_data)}</result_message>"
        )

        result_text, tool_calls = parse_formatted_sdk_content(
            xml_content, collect_tool_calls=False
        )

        # Should process all types in sequence
        assert "[claude_code_sdk]: System message" in result_text
        assert "Some text content" in result_text
        assert "[claude_code_sdk tool_use tool_1]: search" in result_text
        assert "[claude_code_sdk tool_result tool_1]: result" in result_text
        assert "[claude_code_sdk result sess_1]: stop_reason=end_turn" in result_text
        assert tool_calls == []

    def test_parse_mixed_content_openai_adapter(self) -> None:
        """Test parsing mixed SDK content for OpenAI adapter."""
        system_data = {"text": "System message"}
        tool_data = {"id": "tool_1", "name": "search", "input": {"q": "test"}}

        xml_content = (
            f"<system_message>{json.dumps(system_data)}</system_message>\n"
            f"<text>Some text content</text>\n"
            f"<tool_use_sdk>{json.dumps(tool_data)}</tool_use_sdk>"
        )

        result_text, tool_calls = parse_formatted_sdk_content(
            xml_content, collect_tool_calls=True
        )

        # Tool use should be removed from text but collected as tool call
        assert "[claude_code_sdk]: System message" in result_text
        assert "Some text content" in result_text
        assert "tool_use" not in result_text  # Tool use XML should be removed

        assert len(tool_calls) == 1
        assert tool_calls[0].id == "tool_1"

    def test_parse_processing_order(self) -> None:
        """Test that parsing functions are applied in correct order."""
        # Text tags should be processed last to unwrap content properly
        xml_content = (
            "<text>"
            '<system_message>{"text": "System message"}</system_message>'
            "Content here"
            "</text>"
        )

        result_text, tool_calls = parse_formatted_sdk_content(
            xml_content, collect_tool_calls=False
        )

        # System message should be processed first, then text tags unwrapped
        assert "[claude_code_sdk]: System message" in result_text
        assert "Content here" in result_text
        assert "<text>" not in result_text
        assert "</text>" not in result_text

    def test_parse_real_world_example(self) -> None:
        """Test parsing a real-world example with multiple elements."""
        xml_content = (
            "<text>\n"
            "I'll help you search for that information.\n"
            "</text>\n"
            "<tool_use_sdk>"
            '{"id": "search_123", "name": "web_search", "input": {"query": "Python testing"}}'
            "</tool_use_sdk>\n"
            "<tool_result_sdk>"
            '{"tool_use_id": "search_123", "content": "Found 10 results", "is_error": false}'
            "</tool_result_sdk>\n"
            "<text>\n"
            "Based on the search results, here's what I found...\n"
            "</text>"
        )

        result_text, tool_calls = parse_formatted_sdk_content(
            xml_content, collect_tool_calls=False
        )

        assert "I'll help you search for that information." in result_text
        assert "[claude_code_sdk tool_use search_123]: web_search" in result_text
        assert (
            "[claude_code_sdk tool_result search_123]: Found 10 results" in result_text
        )
        assert "Based on the search results, here's what I found..." in result_text
        assert tool_calls == []
