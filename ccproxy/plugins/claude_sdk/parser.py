"""Centralized XML parser for Claude SDK formatted content.

This module provides parsing functions for XML-formatted SDK content that appears
in Claude Code SDK responses. It consolidates the parsing logic that was previously
duplicated across OpenAI adapter and streaming components.

Currently not usedd but could be useful to rebuild message
for turn to turn conversation.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ccproxy.llms.models import openai as openai_models


def format_openai_tool_call(tool_use: dict[str, Any]) -> openai_models.ToolCall:
    """Convert Anthropic tool use to OpenAI tool call format."""
    tool_input = tool_use.get("input", {})
    if isinstance(tool_input, dict):
        arguments_str = json.dumps(tool_input)
    else:
        arguments_str = str(tool_input)

    return openai_models.ToolCall(
        id=tool_use.get("id", ""),
        type="function",
        function=openai_models.FunctionCall(
            name=tool_use.get("name", ""),
            arguments=arguments_str,
        ),
    )


def parse_system_message_tags(text: str) -> str:
    """Parse and format system_message XML tags.

    Args:
        text: Text content that may contain system_message XML tags

    Returns:
        Text with system_message tags converted to readable format
    """
    system_pattern = r"<system_message>(.*?)</system_message>"

    def replace_system_message(match: re.Match[str]) -> str:
        try:
            system_data = json.loads(match.group(1))
            source = system_data.get("source", "claude_code_sdk")
            system_text = system_data.get("text", "")
            return f"[{source}]: {system_text}"
        except json.JSONDecodeError:
            # Keep original if parsing fails
            return match.group(0)

    return re.sub(system_pattern, replace_system_message, text, flags=re.DOTALL)


def parse_tool_use_sdk_tags(
    text: str, collect_tool_calls: bool = False
) -> tuple[str, list[Any]]:
    """Parse and format tool_use_sdk XML tags.

    Args:
        text: Text content that may contain tool_use_sdk XML tags
        collect_tool_calls: Whether to collect tool calls for OpenAI format conversion

    Returns:
        Tuple of (processed_text, tool_calls_list)
    """
    tool_use_pattern = r"<tool_use_sdk>(.*?)</tool_use_sdk>"
    tool_calls = []

    def replace_tool_use(match: re.Match[str]) -> str:
        try:
            tool_data = json.loads(match.group(1))

            if collect_tool_calls:
                # For OpenAI adapter: collect tool calls and remove from text
                tool_call_block = {
                    "type": "tool_use",
                    "id": tool_data.get("id", ""),
                    "name": tool_data.get("name", ""),
                    "input": tool_data.get("input", {}),
                }
                tool_calls.append(format_openai_tool_call(tool_call_block))
                return ""  # Remove the XML tag from text
            else:
                # For streaming: format as readable text
                tool_id = tool_data.get("id", "")
                tool_name = tool_data.get("name", "")
                tool_input = tool_data.get("input", {})
                return f"[claude_code_sdk tool_use {tool_id}]: {tool_name}({json.dumps(tool_input)})"
        except json.JSONDecodeError:
            # Keep original if parsing fails
            return match.group(0)

    processed_text = re.sub(tool_use_pattern, replace_tool_use, text, flags=re.DOTALL)
    return processed_text, tool_calls


def parse_tool_result_sdk_tags(text: str) -> str:
    """Parse and format tool_result_sdk XML tags.

    Args:
        text: Text content that may contain tool_result_sdk XML tags

    Returns:
        Text with tool_result_sdk tags converted to readable format
    """
    tool_result_pattern = r"<tool_result_sdk>(.*?)</tool_result_sdk>"

    def replace_tool_result(match: re.Match[str]) -> str:
        try:
            result_data = json.loads(match.group(1))
            tool_use_id = result_data.get("tool_use_id", "")
            result_content = result_data.get("content", "")
            is_error = result_data.get("is_error", False)
            error_indicator = " (ERROR)" if is_error else ""
            return f"[claude_code_sdk tool_result {tool_use_id}{error_indicator}]: {result_content}"
        except json.JSONDecodeError:
            # Keep original if parsing fails
            return match.group(0)

    return re.sub(tool_result_pattern, replace_tool_result, text, flags=re.DOTALL)


def parse_result_message_tags(text: str) -> str:
    """Parse and format result_message XML tags.

    Args:
        text: Text content that may contain result_message XML tags

    Returns:
        Text with result_message tags converted to readable format
    """
    result_message_pattern = r"<result_message>(.*?)</result_message>"

    def replace_result_message(match: re.Match[str]) -> str:
        try:
            result_data = json.loads(match.group(1))
            source = result_data.get("source", "claude_code_sdk")
            session_id = result_data.get("session_id", "")
            stop_reason = result_data.get("stop_reason", "")
            usage = result_data.get("usage", {})
            cost_usd = result_data.get("total_cost_usd")

            formatted_content = f"[{source} result {session_id}]: stop_reason={stop_reason}, usage={usage}"
            if cost_usd is not None:
                formatted_content += f", cost_usd={cost_usd}"
            return formatted_content
        except json.JSONDecodeError:
            # Keep original if parsing fails
            return match.group(0)

    return re.sub(result_message_pattern, replace_result_message, text, flags=re.DOTALL)


def parse_text_tags(text: str) -> str:
    """Parse and extract content from text XML tags.

    Args:
        text: Text content that may contain text XML tags

    Returns:
        Text with text tags unwrapped (inner content extracted)
    """
    text_pattern = r"<text>\n?(.*?)\n?</text>"

    def replace_text(match: re.Match[str]) -> str:
        return match.group(1).strip()

    return re.sub(text_pattern, replace_text, text, flags=re.DOTALL)


def parse_formatted_sdk_content(
    text: str, collect_tool_calls: bool = False
) -> tuple[str, list[Any]]:
    """Parse XML-formatted SDK content from text blocks.

    This is the main parsing function that handles all types of XML-formatted
    SDK content by applying the appropriate parsing functions in sequence.

    Args:
        text: Text content that may contain XML-formatted SDK data
        collect_tool_calls: Whether to collect tool calls for OpenAI format conversion
                           (used by OpenAI adapter, not by streaming processor)

    Returns:
        Tuple of (cleaned_text, tool_calls_list)
        - cleaned_text: Text with XML-formatted SDK content converted to readable format
        - tool_calls_list: List of tool calls (empty if collect_tool_calls=False)
    """
    if not text:
        return text, []

    # Apply all parsing functions in sequence
    cleaned_text = text

    # Parse system messages
    cleaned_text = parse_system_message_tags(cleaned_text)

    # Parse tool use blocks (may collect tool calls)
    cleaned_text, tool_calls = parse_tool_use_sdk_tags(cleaned_text, collect_tool_calls)

    # Parse tool result blocks
    cleaned_text = parse_tool_result_sdk_tags(cleaned_text)

    # Parse result message blocks
    cleaned_text = parse_result_message_tags(cleaned_text)

    # Parse text tags (unwrap content) - do this last
    cleaned_text = parse_text_tags(cleaned_text)

    return cleaned_text.strip(), tool_calls
