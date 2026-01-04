"""Shared helper utilities for OpenAI↔OpenAI formatters."""

from __future__ import annotations

import contextlib
import json
from typing import Any


TOOL_FUNCTION_KEYS = {"name", "description", "parameters"}
_RESPONSES_TEXTUAL_PART_TYPES = {"input_text", "text", "output_text"}


def _get_attr(obj: Any, name: str) -> Any:
    """Safely fetch an attribute from dicts or objects."""

    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _as_serializable_dict(part: Any) -> dict[str, Any] | None:
    if isinstance(part, dict):
        return part
    if hasattr(part, "model_dump"):
        with contextlib.suppress(Exception):
            data = part.model_dump(mode="json", exclude_none=True)
            if isinstance(data, dict):
                return data
    return None


def _normalize_responses_input_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list | tuple):
        text_parts: list[str] = []
        fallback_parts: list[Any] = []

        for part in content:
            serializable = _as_serializable_dict(part)
            if serializable is not None:
                part_type = serializable.get("type")
                if part_type in _RESPONSES_TEXTUAL_PART_TYPES:
                    text_value = serializable.get("text")
                    if isinstance(text_value, str) and text_value.strip():
                        text_parts.append(text_value.strip())
                        continue
                fallback_parts.append(serializable)
                continue

            if isinstance(part, str) and part.strip():
                text_parts.append(part.strip())
            else:
                fallback_parts.append(part)

        if text_parts:
            return "\n\n".join(text_parts)

        if fallback_parts:
            with contextlib.suppress(TypeError, ValueError):
                return json.dumps(fallback_parts, ensure_ascii=False)
        return ""

    if isinstance(content, dict):
        with contextlib.suppress(TypeError, ValueError):
            return json.dumps(content, ensure_ascii=False)
        return ""

    if content is None:
        return ""

    if hasattr(content, "model_dump"):
        with contextlib.suppress(Exception):
            data = content.model_dump(mode="json", exclude_none=True)
            if isinstance(data, dict | list):
                with contextlib.suppress(TypeError, ValueError):
                    return json.dumps(data, ensure_ascii=False)

    if hasattr(content, "dict"):
        with contextlib.suppress(Exception):
            data = content.dict()
            if isinstance(data, dict | list):
                with contextlib.suppress(TypeError, ValueError):
                    return json.dumps(data, ensure_ascii=False)

    return str(content)


def _extract_responses_role_and_content(item: Any) -> tuple[str, Any]:
    if isinstance(item, dict):
        role = item.get("role")
        content = item.get("content")
    else:
        role = getattr(item, "role", None)
        content = getattr(item, "content", None)

    if isinstance(role, str) and role:
        return role, content

    return "user", content


def _flatten_chat_message_content(content: Any) -> str:
    """Extract plain text from a ChatCompletion message content payload."""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        segments: list[str] = []
        for part in content:
            text_value = None
            if isinstance(part, dict):
                text_value = part.get("text")
            else:
                text_value = getattr(part, "text", None)

            if isinstance(text_value, str) and text_value.strip():
                segments.append(text_value.strip())

        if segments:
            return " ".join(segments).strip()

    return ""


def _collect_chat_instruction_segments(messages: list[Any] | None) -> list[str]:
    """Return normalized instruction strings from chat messages."""

    if not messages:
        return []

    segments: list[str] = []
    for message in messages:
        role = getattr(message, "role", None)
        if role not in {"system", "developer"}:
            continue

        content = getattr(message, "content", None)
        text_value = _flatten_chat_message_content(content)
        if text_value:
            segments.append(text_value)

    return segments


def _convert_tool_choice_responses_to_chat(tool_choice: Any) -> Any:
    """Responses tool choice (flat) → Chat tool choice (nested)."""

    if isinstance(tool_choice, dict):
        choice_type = tool_choice.get("type")
        if choice_type != "function":
            return tool_choice

        function_block = tool_choice.get("function")
        if isinstance(function_block, dict) and function_block.get("name"):
            return tool_choice

        name = None
        if isinstance(function_block, dict):
            name = function_block.get("name")
        if not name:
            name = tool_choice.get("name")

        if not name:
            return tool_choice

        new_choice = {
            key: value for key, value in tool_choice.items() if key not in {"name"}
        }
        new_choice["function"] = {"name": name}
        return new_choice

    return tool_choice


def _convert_tool_choice_chat_to_responses(tool_choice: Any) -> Any:
    """Chat tool choice (nested) → Responses tool choice (flat)."""

    if isinstance(tool_choice, dict):
        choice_type = tool_choice.get("type")
        if choice_type != "function":
            return tool_choice

        function_block = tool_choice.get("function")
        if not isinstance(function_block, dict):
            return tool_choice

        name = function_block.get("name")
        if not name:
            return tool_choice

        new_choice = {
            key: value for key, value in tool_choice.items() if key not in {"function"}
        }
        new_choice["name"] = name
        return new_choice

    return tool_choice


def _coerce_tool_dict(tool: Any) -> dict[str, Any] | None:
    """Return a shallow dict representation for a tool model/dict."""

    if hasattr(tool, "model_dump"):
        try:
            result = tool.model_dump(mode="json", exclude_none=True)
            if isinstance(result, dict):
                return result
            return None
        except TypeError:
            result = tool.model_dump()
            if isinstance(result, dict):
                return result
            return None
    if isinstance(tool, dict):
        return dict(tool)
    return None


def _convert_tools_responses_to_chat(
    tools: list[Any] | None,
) -> list[dict[str, Any]] | None:
    """Ensure Responses-style tools conform to ChatCompletion schema."""

    if not tools:
        return None

    converted: list[dict[str, Any]] = []
    for tool in tools:
        tool_dict = _coerce_tool_dict(tool)
        if not tool_dict:
            continue

        tool_type = tool_dict.get("type")
        if tool_type != "function":
            converted.append(tool_dict)
            continue

        function_block = tool_dict.get("function")
        if not isinstance(function_block, dict):
            fn_payload = {
                key: value
                for key, value in tool_dict.items()
                if key in TOOL_FUNCTION_KEYS and value is not None
            }
        else:
            fn_payload = {
                key: value
                for key, value in function_block.items()
                if key in TOOL_FUNCTION_KEYS and value is not None
            }
            for key in TOOL_FUNCTION_KEYS:
                if key not in fn_payload and tool_dict.get(key) is not None:
                    fn_payload[key] = tool_dict[key]

        if "parameters" not in fn_payload or fn_payload.get("parameters") is None:
            fn_payload["parameters"] = {}

        new_tool = {
            key: value
            for key, value in tool_dict.items()
            if key not in (*TOOL_FUNCTION_KEYS, "function")
        }
        new_tool["function"] = fn_payload
        converted.append(new_tool)

    return converted or None


def _convert_tools_chat_to_responses(
    tools: list[Any] | None,
) -> list[dict[str, Any]] | None:
    """Normalize ChatCompletion tool payloads into Responses format."""

    if not tools:
        return None

    converted: list[dict[str, Any]] = []
    for tool in tools:
        tool_dict = _coerce_tool_dict(tool)
        if not tool_dict:
            continue

        tool_type = tool_dict.get("type")
        if tool_type != "function":
            converted.append(tool_dict)
            continue

        function_block = tool_dict.get("function")
        if not isinstance(function_block, dict):
            converted.append(tool_dict)
            continue

        base_tool = {
            key: value for key, value in tool_dict.items() if key not in {"function"}
        }

        for key in TOOL_FUNCTION_KEYS:
            value = function_block.get(key)
            if value is not None:
                base_tool[key] = value

        if "parameters" not in base_tool:
            base_tool["parameters"] = {}

        converted.append(base_tool)

    return converted or None


__all__ = [
    "TOOL_FUNCTION_KEYS",
    "_RESPONSES_TEXTUAL_PART_TYPES",
    "_get_attr",
    "_normalize_responses_input_content",
    "_extract_responses_role_and_content",
    "_collect_chat_instruction_segments",
    "_convert_tool_choice_responses_to_chat",
    "_convert_tool_choice_chat_to_responses",
    "_convert_tools_responses_to_chat",
    "_convert_tools_chat_to_responses",
]
