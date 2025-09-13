from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from ccproxy.llms.adapters.base import BaseAPIAdapter
from ccproxy.llms.anthropic.models import CreateMessageRequest


class OpenAIResponsesRequestToAnthropicMessagesAdapter(BaseAPIAdapter):
    """OpenAI Responses request → Anthropic CreateMessageRequest adapter.

    Implemented
    - model, stream: passthrough
    - input: string or first input message (with input_text) → Anthropic `messages`
    - max_output_tokens → `max_tokens`
    - tools: function tools → Anthropic custom tools
    - tool_choice: maps strings and function selection; sets `disable_parallel_tool_use`

    TODO
    - Map multiple input messages to a sequence of Anthropic `messages`
    - Support images and other content types in Responses input
    - Wire additional OpenAI Responses request fields (reasoning, truncation) if needed
    """

    def __init__(self) -> None:
        super().__init__(name="openai_responses_request_to_anthropic_messages")

    async def adapt_request(self, request: dict[str, Any]) -> dict[str, Any]:
        model = request.get("model")
        stream = bool(request.get("stream") or False)
        max_out = request.get("max_output_tokens")

        messages: list[dict[str, Any]] = []
        input_val = request.get("input")
        if isinstance(input_val, str):
            messages.append({"role": "user", "content": input_val})
        elif isinstance(input_val, list) and input_val:
            # Expect [{type:"message", role:"user", content:[{type:"input_text", text:"..."}, ...]}]
            first = input_val[0]
            if isinstance(first, dict) and first.get("type") == "message":
                role = first.get("role", "user")
                content_list = first.get("content") or []
                text_parts: list[str] = []
                for part in content_list:
                    if isinstance(part, dict) and part.get("type") in (
                        "input_text",
                        "text",
                    ):
                        t = part.get("text")
                        if isinstance(t, str):
                            text_parts.append(t)
                content_text = " ".join(text_parts)
                messages.append({"role": role, "content": content_text})

        payload: dict[str, Any] = {"model": model, "messages": messages}
        if max_out is not None:
            payload["max_tokens"] = int(max_out)
        if stream:
            payload["stream"] = True

        # Tools mapping (function tools)
        tools_in = request.get("tools") or []
        if isinstance(tools_in, list) and tools_in:
            anth_tools: list[dict[str, Any]] = []
            for t in tools_in:
                if (
                    isinstance(t, dict)
                    and t.get("type") == "function"
                    and isinstance(t.get("function"), dict)
                ):
                    fn = t["function"]
                    anth_tools.append(
                        {
                            "type": "custom",
                            "name": fn.get("name"),
                            "description": fn.get("description"),
                            "input_schema": fn.get("parameters") or {},
                        }
                    )
            if anth_tools:
                payload["tools"] = anth_tools

        # tool_choice mapping (+ parallel control)
        tool_choice = request.get("tool_choice")
        parallel_tool_calls = request.get("parallel_tool_calls")
        disable_parallel = None
        if isinstance(parallel_tool_calls, bool):
            disable_parallel = not parallel_tool_calls

        if tool_choice is not None:
            anth_choice: dict[str, Any] | None = None
            if isinstance(tool_choice, str):
                if tool_choice == "none":
                    anth_choice = {"type": "none"}
                elif tool_choice == "auto":
                    anth_choice = {"type": "auto"}
                elif tool_choice == "required":
                    anth_choice = {"type": "any"}
            elif isinstance(tool_choice, dict):
                if tool_choice.get("type") == "function" and isinstance(
                    tool_choice.get("function"), dict
                ):
                    anth_choice = {
                        "type": "tool",
                        "name": tool_choice["function"].get("name"),
                    }
            if anth_choice is not None:
                if disable_parallel is not None and anth_choice["type"] in {
                    "auto",
                    "any",
                    "tool",
                }:
                    anth_choice["disable_parallel_tool_use"] = disable_parallel
                payload["tool_choice"] = anth_choice

        return CreateMessageRequest.model_validate(payload).model_dump()

    async def adapt_response(self, response: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def adapt_stream(
        self, stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncGenerator[dict[str, Any], None]:
        raise NotImplementedError

    async def adapt_error(self, error: dict[str, Any]) -> dict[str, Any]:
        return error
