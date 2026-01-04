"""Claude SDK integration module."""

from .client import ClaudeSDKClient
from .exceptions import ClaudeSDKError, StreamTimeoutError
from .models import (
    AssistantMessage,
    ContentBlock,
    ExtendedContentBlock,
    ResultMessage,
    ResultMessageBlock,
    SDKContentBlock,
    SDKMessage,
    SDKMessageContent,
    SDKMessageMode,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolResultSDKBlock,
    ToolUseBlock,
    ToolUseSDKBlock,
    UserMessage,
    convert_sdk_result_message,
    convert_sdk_system_message,
    convert_sdk_text_block,
    convert_sdk_tool_result_block,
    convert_sdk_tool_use_block,
    create_sdk_message,
    to_sdk_variant,
)
from .options import OptionsHandler


# Lazy import to avoid circular dependency
def __getattr__(name: str) -> object:
    if name == "MessageConverter":
        from .converter import MessageConverter

        return MessageConverter
    if name == "parse_formatted_sdk_content":
        from .parser import parse_formatted_sdk_content

        return parse_formatted_sdk_content
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Session Context will be imported here once created
    "ClaudeSDKClient",
    "ClaudeSDKError",
    "StreamTimeoutError",
    "MessageConverter",  # Lazy loaded
    "OptionsHandler",
    "parse_formatted_sdk_content",  # Lazy loaded
    # Re-export SDK models from core adapter
    "AssistantMessage",
    "ContentBlock",
    "ExtendedContentBlock",
    "ResultMessage",
    "ResultMessageBlock",
    "SDKContentBlock",
    "SDKMessage",
    "SDKMessageContent",
    "SDKMessageMode",
    "SystemMessage",
    "TextBlock",
    "ThinkingBlock",
    "ToolResultBlock",
    "ToolResultSDKBlock",
    "ToolUseBlock",
    "ToolUseSDKBlock",
    "UserMessage",
    "convert_sdk_result_message",
    "convert_sdk_system_message",
    "convert_sdk_text_block",
    "convert_sdk_tool_result_block",
    "convert_sdk_tool_use_block",
    "create_sdk_message",
    "to_sdk_variant",
]
