"""Shared helpers for Anthropic to OpenAI formatting."""

from __future__ import annotations

import json
from typing import Any

from ccproxy.llms.models import openai as openai_models


def serialize_tool_arguments(tool_input: Any) -> str:
    if isinstance(tool_input, str):
        return tool_input
    try:
        return json.dumps(tool_input, ensure_ascii=False)
    except Exception:
        return json.dumps({"arguments": str(tool_input)})


def build_openai_tool_call(
    *,
    tool_id: str | None,
    tool_name: str | None,
    tool_input: Any,
    arguments: Any = None,
    fallback_index: int = 0,
) -> openai_models.ToolCall:
    args_str = (
        arguments
        if isinstance(arguments, str) and arguments
        else serialize_tool_arguments(tool_input)
    )
    call_id = (
        tool_id if isinstance(tool_id, str) and tool_id else f"call_{fallback_index}"
    )
    name = tool_name if isinstance(tool_name, str) and tool_name else "function"

    return openai_models.ToolCall(
        id=str(call_id),
        function=openai_models.FunctionCall(
            name=str(name),
            arguments=str(args_str),
        ),
    )
