from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from ccproxy.llms.adapters.base import BaseAPIAdapter
from ccproxy.llms.anthropic.models import CreateMessageRequest
from ccproxy.llms.adapters.mapping import DEFAULT_MAX_TOKENS


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
        if max_out is None:
            max_out = DEFAULT_MAX_TOKENS
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

        # Structured outputs (Responses text.format) -> inject system guidance
        text_cfg = request.get("text") or {}
        if isinstance(text_cfg, dict) and isinstance(text_cfg.get("format"), dict):
            fmt = text_cfg["format"]
            ftype = fmt.get("type")
            inject: str | None = None
            if ftype == "json_object":
                inject = (
                    "Respond ONLY with a valid JSON object. No prose. Do not wrap in markdown."
                )
            elif ftype == "json_schema":
                schema = fmt.get("schema") or {}
                try:
                    import json

                    schema_str = json.dumps(schema, separators=(",", ":"))
                except Exception:
                    schema_str = str(schema)
                inject = (
                    "Respond ONLY with JSON strictly conforming to this JSON Schema. "
                    "No prose. No markdown. If unsure, use nulls for unknown fields.\n\n"
                    "Schema:\n"
                    f"{schema_str}"
                )
            if inject:
                if payload.get("system"):
                    payload["system"] = f"{payload['system']}\n\n{inject}"
                else:
                    payload["system"] = inject

        # Instructions passthrough (map to Anthropic system)
        instr = request.get("instructions")
        if isinstance(instr, str) and instr:
            if payload.get("system"):
                payload["system"] = f"{payload['system']}\n\n{instr}"
            else:
                payload["system"] = instr

        # Reasoning -> Anthropic thinking mapping
        reasoning = request.get("reasoning") or {}
        thinking_cfg = self._derive_thinking_config((model or ""), reasoning)
        if thinking_cfg is not None:
            payload["thinking"] = thinking_cfg
            budget = thinking_cfg.get("budget_tokens", 0)
            if isinstance(budget, int) and payload.get("max_tokens", 0) <= budget:
                payload["max_tokens"] = budget + 64
            payload["temperature"] = 1.0

        return CreateMessageRequest.model_validate(payload).model_dump()

    async def adapt_response(self, response: dict[str, Any]) -> dict[str, Any]:
        # Delegate Anthropic -> OpenAI Responses conversion
        from ccproxy.llms.adapters.anthropic_messages_to_openai_responses import (
            AnthropicMessagesToOpenAIResponsesAdapter,
        )

        return await AnthropicMessagesToOpenAIResponsesAdapter().adapt_response(
            response
        )

    def adapt_stream(
        self, stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncGenerator[dict[str, Any], None]:
        async def generator() -> AsyncGenerator[dict[str, Any], None]:
            # Map Anthropic SSE → OpenAI Responses stream events (text/refusal/tool args)
            item_id = "msg_stream"
            output_index = 0
            content_index = 0
            async for evt in stream:
                if not isinstance(evt, dict):
                    continue
                etype = evt.get("type")
                if etype == "message_start":
                    msg = evt.get("message") or {}
                    yield {"type": "response.created", "response": {"model": msg.get("model", "")}}
                elif etype == "content_block_start":
                    # If a tool_use block starts, emit final function_call arguments (no delta in Anthropic)
                    cb = evt.get("content_block") or {}
                    if isinstance(cb, dict) and cb.get("type") == "tool_use":
                        tool_input = cb.get("input") or {}
                        try:
                            import json

                            args_str = json.dumps(tool_input, separators=(",", ":"))
                        except Exception:
                            args_str = str(tool_input)
                        yield {
                            "type": "response.function_call_arguments.done",
                            "item_id": item_id,
                            "output_index": output_index,
                            "arguments": args_str,
                        }
                elif etype == "content_block_delta":
                    delta = evt.get("delta") or {}
                    text = delta.get("text") if isinstance(delta, dict) else None
                    if isinstance(text, str) and text:
                        yield {
                            "type": "response.output_text.delta",
                            "item_id": item_id,
                            "output_index": output_index,
                            "content_index": content_index,
                            "delta": text,
                        }
                elif etype == "message_delta":
                    # Emit in_progress with usage, and map refusal if present
                    yield {
                        "type": "response.in_progress",
                        "response": {"usage": evt.get("usage") or {}},
                    }
                    delta = evt.get("delta") or {}
                    stop_reason = delta.get("stop_reason") if isinstance(delta, dict) else None
                    if stop_reason == "refusal":
                        yield {
                            "type": "response.refusal.done",
                            "item_id": item_id,
                            "output_index": output_index,
                            "content_index": content_index,
                            "refusal": "refused",
                        }
                elif etype == "message_stop":
                    yield {"type": "response.completed", "response": {"usage": {}}}
                    break

        return generator()

    async def adapt_error(self, error: dict[str, Any]) -> dict[str, Any]:
        return error

    def _derive_thinking_config(
        self, model: str, reasoning: dict[str, Any]
    ) -> dict[str, Any] | None:
        effort = (reasoning.get("effort") or "").strip().lower()
        effort_budgets = {"low": 1000, "medium": 5000, "high": 10000}
        budget: int | None = effort_budgets.get(effort)
        m = (model or "").lower()
        if budget is None:
            if m.startswith("o3"):
                budget = 10000
            elif m.startswith("o1-mini"):
                budget = 3000
            elif m.startswith("o1"):
                budget = 5000
        if budget is None:
            return None
        return {"type": "enabled", "budget_tokens": budget}
