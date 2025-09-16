from __future__ import annotations

import json
import re
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from pydantic import BaseModel

from ccproxy.llms.adapters.base import BaseAPIAdapter
from ccproxy.llms.adapters.shared import (
    convert_openai_error_to_anthropic,
    convert_openai_response_usage_to_anthropic_usage,
)
from ccproxy.llms.adapters.shared.usage import (
    convert_anthropic_usage_to_openai_response_usage,
)
from ccproxy.llms.anthropic import models as anthropic_models
from ccproxy.llms.anthropic.models import MessageResponse as AnthropicMessageResponse
from ccproxy.llms.openai import models as openai_models
from ccproxy.llms.openai.models import ResponseRequest


THINKING_PATTERN = re.compile(
    r"<thinking(?:\s+signature=\"([^\"]*)\")?>(.*?)</thinking>",
    re.DOTALL,
)


class OpenAIResponsesToAnthropicAdapter(
    BaseAPIAdapter[
        BaseModel,
        BaseModel,
        openai_models.AnyStreamEvent,
    ]
):
    """OpenAI Responses → Anthropic Messages adapter (non-streaming + streaming subset)."""

    def __init__(self) -> None:
        super().__init__(name="openai_responses_to_anthropic")

    async def adapt_request(self, request: BaseModel) -> BaseModel:
        if not isinstance(request, ResponseRequest):
            raise ValueError(f"Expected ResponseRequest, got {type(request)}")

        return convert_openai_response_request_to_anthropic(request)

    async def adapt_response(self, response: BaseModel) -> AnthropicMessageResponse:
        if not isinstance(response, openai_models.ResponseObject):
            raise ValueError(f"Expected ResponseObject, got {type(response)}")

        return convert_openai_response_to_anthropic_message(response)

    def adapt_stream(
        self,
        stream: AsyncIterator[openai_models.AnyStreamEvent],
    ) -> AsyncGenerator[anthropic_models.MessageStreamEvent, None]:
        return self._convert_stream(stream)

    async def adapt_error(self, error: BaseModel) -> BaseModel:
        return convert_openai_error_to_anthropic(error)

    def _convert_stream(
        self,
        stream: AsyncIterator[openai_models.AnyStreamEvent],
    ) -> AsyncGenerator[anthropic_models.MessageStreamEvent, None]:
        async def generator() -> AsyncGenerator[
            anthropic_models.MessageStreamEvent, None
        ]:
            message_started = False
            index = 0
            func_args_buffer: dict[int, list[str]] = {}
            tool_info: dict[int, dict[str, str]] = {}
            reasoning_buffer: list[str] = []
            text_block_started = False

            async for evt_wrapper in stream:
                if not hasattr(evt_wrapper, "root"):
                    continue
                evt = evt_wrapper.root
                if not hasattr(evt, "type"):
                    continue

                if (
                    evt.type in ("response.created", "response.in_progress")
                    and isinstance(evt, openai_models.ResponseCreatedEvent)
                    and hasattr(evt, "response")
                    and evt.response is not None
                    and not message_started
                ):
                    yield anthropic_models.MessageStartEvent(
                        type="message_start",
                        message=anthropic_models.MessageResponse(
                            id=evt.response.id or "resp_stream",
                            type="message",
                            role="assistant",
                            content=[],
                            model=evt.response.model or "",
                            stop_reason=None,
                            stop_sequence=None,
                            usage=anthropic_models.Usage(
                                input_tokens=0, output_tokens=0
                            ),
                        ),
                    )
                    message_started = True
                elif evt.type == "response.output_text.delta":
                    text = evt.delta or ""
                    if text:
                        if not text_block_started:
                            yield anthropic_models.ContentBlockStartEvent(
                                type="content_block_start",
                                index=index,
                                content_block=anthropic_models.TextBlock(
                                    type="text", text=""
                                ),
                            )
                            text_block_started = True
                        yield anthropic_models.ContentBlockDeltaEvent(
                            type="content_block_delta",
                            index=index,
                            delta=anthropic_models.TextBlock(type="text", text=text),
                        )
                elif evt.type == "response.reasoning_summary_text.delta":
                    delta = evt.delta or ""
                    if isinstance(delta, str):
                        reasoning_buffer.append(delta)
                elif evt.type == "response.reasoning_summary_text.done":
                    if text_block_started:
                        yield anthropic_models.ContentBlockStopEvent(
                            type="content_block_stop", index=index
                        )
                        text_block_started = False
                        index += 1
                    summary = "".join(reasoning_buffer)
                    if summary:
                        yield anthropic_models.ContentBlockStartEvent(
                            type="content_block_start",
                            index=index,
                            content_block=anthropic_models.ThinkingBlock(
                                type="thinking",
                                thinking=summary,
                                signature="",
                            ),
                        )
                        yield anthropic_models.ContentBlockStopEvent(
                            type="content_block_stop", index=index
                        )
                        index += 1
                    reasoning_buffer.clear()
                elif evt.type == "response.function_call_arguments.delta":
                    output_index = evt.output_index
                    if output_index not in func_args_buffer:
                        func_args_buffer[output_index] = []
                        tool_info[output_index] = {
                            "id": evt.item_id or f"call_{output_index}",
                            "name": evt.item_id or f"call_{output_index}",
                        }
                    delta = evt.delta or ""
                    if isinstance(delta, str):
                        func_args_buffer[output_index].append(delta)
                elif evt.type == "response.function_call_arguments.done":
                    if text_block_started:
                        yield anthropic_models.ContentBlockStopEvent(
                            type="content_block_stop", index=index
                        )
                        text_block_started = False
                        index += 1
                    output_index = evt.output_index
                    args_str = evt.arguments or "".join(
                        func_args_buffer.get(output_index, [])
                    )
                    try:
                        args_obj = (
                            json.loads(args_str) if isinstance(args_str, str) else {}
                        )
                    except Exception:
                        args_obj = {}

                    info = tool_info.get(
                        output_index,
                        {"id": f"call_{output_index}", "name": f"call_{output_index}"},
                    )

                    yield anthropic_models.ContentBlockStartEvent(
                        type="content_block_start",
                        index=index,
                        content_block=anthropic_models.ToolUseBlock(
                            type="tool_use",
                            id=info["id"],
                            name=info["name"],
                            input=args_obj,
                        ),
                    )
                    yield anthropic_models.ContentBlockStopEvent(
                        type="content_block_stop", index=index
                    )
                    index += 1
                elif evt.type in {
                    "response.completed",
                    "response.failed",
                    "response.incomplete",
                } and isinstance(evt, openai_models.ResponseCompletedEvent):
                    if text_block_started:
                        yield anthropic_models.ContentBlockStopEvent(
                            type="content_block_stop", index=index
                        )
                        text_block_started = False
                    usage = getattr(evt.response, "usage", None)
                    yield anthropic_models.MessageDeltaEvent(
                        type="message_delta",
                        delta=anthropic_models.MessageDelta(stop_reason="end_turn"),
                        usage=convert_openai_response_usage_to_anthropic_usage(usage)
                        if usage
                        else anthropic_models.Usage(input_tokens=0, output_tokens=0),
                    )
                    yield anthropic_models.MessageStopEvent(type="message_stop")
                    break

        return generator()


def convert_openai_response_to_anthropic_message(
    response: openai_models.ResponseObject,
) -> AnthropicMessageResponse:
    from ccproxy.llms.anthropic.models import (
        TextBlock as AnthropicTextBlock,
    )
    from ccproxy.llms.anthropic.models import (
        ThinkingBlock as AnthropicThinkingBlock,
    )
    from ccproxy.llms.anthropic.models import (
        ToolUseBlock as AnthropicToolUseBlock,
    )

    content_blocks: list[
        AnthropicTextBlock | AnthropicThinkingBlock | AnthropicToolUseBlock
    ] = []

    for item in response.output or []:
        item_type = getattr(item, "type", None)
        if item_type == "reasoning":
            summary_parts = getattr(item, "summary", []) or []
            texts: list[str] = []
            for part in summary_parts:
                part_type = getattr(part, "type", None)
                if part_type == "summary_text":
                    text = getattr(part, "text", None)
                    if isinstance(text, str):
                        texts.append(text)
            if texts:
                content_blocks.append(
                    AnthropicThinkingBlock(
                        type="thinking",
                        thinking=" ".join(texts),
                        signature="",
                    )
                )

    for item in response.output or []:
        item_type = getattr(item, "type", None)
        if item_type == "message":
            content_list = getattr(item, "content", []) or []
            for part in content_list:
                if hasattr(part, "type") and part.type == "output_text":
                    text = getattr(part, "text", "") or ""
                    last_idx = 0
                    for match in THINKING_PATTERN.finditer(text):
                        if match.start() > last_idx:
                            prefix = text[last_idx : match.start()]
                            if prefix.strip():
                                content_blocks.append(
                                    AnthropicTextBlock(type="text", text=prefix)
                                )
                        signature = match.group(1) or ""
                        thinking_text = match.group(2) or ""
                        content_blocks.append(
                            AnthropicThinkingBlock(
                                type="thinking",
                                thinking=thinking_text,
                                signature=signature,
                            )
                        )
                        last_idx = match.end()
                    tail = text[last_idx:]
                    if tail.strip():
                        content_blocks.append(
                            AnthropicTextBlock(type="text", text=tail)
                        )
                elif isinstance(part, dict):
                    part_type = part.get("type")
                    if part_type == "output_text":
                        text = part.get("text", "") or ""
                        last_idx = 0
                        for match in THINKING_PATTERN.finditer(text):
                            if match.start() > last_idx:
                                prefix = text[last_idx : match.start()]
                                if prefix.strip():
                                    content_blocks.append(
                                        AnthropicTextBlock(type="text", text=prefix)
                                    )
                            signature = match.group(1) or ""
                            thinking_text = match.group(2) or ""
                            content_blocks.append(
                                AnthropicThinkingBlock(
                                    type="thinking",
                                    thinking=thinking_text,
                                    signature=signature,
                                )
                            )
                            last_idx = match.end()
                        tail = text[last_idx:]
                        if tail.strip():
                            content_blocks.append(
                                AnthropicTextBlock(type="text", text=tail)
                            )
                    elif part_type == "tool_use":
                        content_blocks.append(
                            AnthropicToolUseBlock(
                                type="tool_use",
                                id=part.get("id", "tool_1"),
                                name=part.get("name", "function"),
                                input=part.get("arguments", part.get("input", {}))
                                or {},
                            )
                        )
                elif (
                    hasattr(part, "type") and getattr(part, "type", None) == "tool_use"
                ):
                    content_blocks.append(
                        AnthropicToolUseBlock(
                            type="tool_use",
                            id=getattr(part, "id", "tool_1") or "tool_1",
                            name=getattr(part, "name", "function") or "function",
                            input=getattr(part, "arguments", getattr(part, "input", {}))
                            or {},
                        )
                    )

    usage = (
        convert_openai_response_usage_to_anthropic_usage(response.usage)
        if response.usage
        else anthropic_models.Usage(input_tokens=0, output_tokens=0)
    )

    return anthropic_models.MessageResponse(
        id=response.id or "msg_1",
        type="message",
        role="assistant",
        model=response.model or "",
        content=content_blocks,  # type: ignore[arg-type]
        stop_reason="end_turn",
        stop_sequence=None,
        usage=usage,
    )


def convert_openai_response_request_to_anthropic(
    request: ResponseRequest,
) -> anthropic_models.CreateMessageRequest:
    model = request.model
    stream = bool(request.stream)
    max_out = request.max_output_tokens

    messages: list[dict[str, Any]] = []
    system_parts: list[str] = []
    input_val = request.input

    if isinstance(input_val, str):
        messages.append({"role": "user", "content": input_val})
    elif isinstance(input_val, list):
        for item in input_val:
            if isinstance(item, dict) and item.get("type") == "message":
                role = item.get("role", "user")
                content_list = item.get("content", [])
                text_parts: list[str] = []
                for part in content_list:
                    if isinstance(part, dict) and part.get("type") in {
                        "input_text",
                        "text",
                    }:
                        text = part.get("text")
                        if isinstance(text, str):
                            text_parts.append(text)
                content_text = " ".join(text_parts)
                if role == "system":
                    system_parts.append(content_text)
                elif role in {"user", "assistant"}:
                    messages.append({"role": role, "content": content_text})
            elif hasattr(item, "type") and item.type == "message":
                role = getattr(item, "role", "user")
                content_list = getattr(item, "content", []) or []
                text_parts_alt: list[str] = []
                for part in content_list:
                    if hasattr(part, "type") and part.type in {"input_text", "text"}:
                        text = getattr(part, "text", None)
                        if isinstance(text, str):
                            text_parts_alt.append(text)
                content_text = " ".join(text_parts_alt)
                if role == "system":
                    system_parts.append(content_text)
                elif role in {"user", "assistant"}:
                    messages.append({"role": role, "content": content_text})

    payload_data: dict[str, Any] = {"model": model, "messages": messages}
    if max_out is None:
        max_out = DEFAULT_MAX_TOKENS
    payload_data["max_tokens"] = int(max_out)
    if stream:
        payload_data["stream"] = True

    if system_parts:
        payload_data["system"] = "\n".join(system_parts)

    tools_in = request.tools or []
    if tools_in:
        anth_tools: list[dict[str, Any]] = []
        for tool in tools_in:
            if isinstance(tool, dict):
                if tool.get("type") == "function" and isinstance(
                    tool.get("function"), dict
                ):
                    fn = tool["function"]
                    anth_tools.append(
                        {
                            "type": "custom",
                            "name": fn.get("name"),
                            "description": fn.get("description"),
                            "input_schema": fn.get("parameters") or {},
                        }
                    )
            elif (
                hasattr(tool, "type")
                and tool.type == "function"
                and hasattr(tool, "function")
                and tool.function is not None
            ):
                fn = tool.function
                anth_tools.append(
                    {
                        "type": "custom",
                        "name": fn.name,
                        "description": fn.description,
                        "input_schema": fn.parameters.model_dump()
                        if hasattr(fn.parameters, "model_dump")
                        else (fn.parameters or {}),
                    }
                )
        if anth_tools:
            payload_data["tools"] = anth_tools

    tool_choice = request.tool_choice
    parallel_tool_calls = request.parallel_tool_calls
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
        elif hasattr(tool_choice, "type") and hasattr(tool_choice, "function"):
            if tool_choice.type == "function" and tool_choice.function is not None:
                anth_choice = {"type": "tool", "name": tool_choice.function.name}
        if anth_choice is not None:
            if disable_parallel is not None and anth_choice["type"] in {
                "auto",
                "any",
                "tool",
            }:
                anth_choice["disable_parallel_tool_use"] = disable_parallel
            payload_data["tool_choice"] = anth_choice

    text_cfg = request.text
    inject: str | None = None
    if text_cfg is not None:
        fmt = None
        if isinstance(text_cfg, dict):
            fmt = text_cfg.get("format")
        elif hasattr(text_cfg, "format"):
            fmt = text_cfg.format
        if fmt is not None:
            if isinstance(fmt, dict):
                fmt_type = fmt.get("type")
                if fmt_type == "json_schema":
                    schema = fmt.get("json_schema") or fmt.get("schema") or {}
                    try:
                        inject_schema = json.dumps(schema, separators=(",", ":"))
                    except Exception:
                        inject_schema = str(schema)
                    inject = (
                        "Respond ONLY with JSON strictly conforming to this JSON Schema:\n"
                        f"{inject_schema}"
                    )
                elif fmt_type == "json_object":
                    inject = (
                        "Respond ONLY with a valid JSON object. "
                        "No prose. Do not wrap in markdown."
                    )
            elif hasattr(fmt, "type"):
                if fmt.type == "json_object":
                    inject = (
                        "Respond ONLY with a valid JSON object. "
                        "No prose. Do not wrap in markdown."
                    )
                elif fmt.type == "json_schema" and (
                    hasattr(fmt, "json_schema") or hasattr(fmt, "schema")
                ):
                    schema_obj = getattr(fmt, "json_schema", None) or getattr(
                        fmt, "schema", None
                    )
                    try:
                        schema_data = (
                            schema_obj.model_dump()
                            if hasattr(schema_obj, "model_dump")
                            else schema_obj
                        )
                        inject_schema = json.dumps(schema_data, separators=(",", ":"))
                    except Exception:
                        inject_schema = str(schema_obj)
                    inject = (
                        "Respond ONLY with JSON strictly conforming to this JSON Schema:\n"
                        f"{inject_schema}"
                    )

    if inject:
        existing_system = payload_data.get("system")
        payload_data["system"] = (
            f"{existing_system}\n\n{inject}" if existing_system else inject
        )

    text_instructions: str | None = None
    if isinstance(text_cfg, dict):
        text_instructions = text_cfg.get("instructions")
    elif hasattr(text_cfg, "instructions"):
        text_instructions = text_cfg.instructions

    if isinstance(text_instructions, str) and text_instructions:
        existing_system = payload_data.get("system")
        payload_data["system"] = (
            f"{existing_system}\n\n{text_instructions}"
            if existing_system
            else text_instructions
        )

    if isinstance(request.instructions, str) and request.instructions:
        existing_system = payload_data.get("system")
        payload_data["system"] = (
            f"{existing_system}\n\n{request.instructions}"
            if existing_system
            else request.instructions
        )

    reasoning = request.reasoning
    thinking_cfg = derive_thinking_config(model or "", reasoning)
    if thinking_cfg is not None:
        payload_data["thinking"] = thinking_cfg
        budget = thinking_cfg.get("budget_tokens", 0)
        if isinstance(budget, int) and payload_data.get("max_tokens", 0) <= budget:
            payload_data["max_tokens"] = budget + 64
        payload_data["temperature"] = 1.0

    return anthropic_models.CreateMessageRequest.model_validate(payload_data)


def derive_thinking_config(model: str, reasoning: Any) -> dict[str, Any] | None:
    if reasoning is None:
        reasoning = {}
    if isinstance(reasoning, dict):
        effort = reasoning.get("effort", "")
    else:
        effort = getattr(reasoning, "effort", None) if reasoning else None

    effort = effort.strip().lower() if isinstance(effort, str) else ""
    effort_budgets = {"low": 1024, "medium": 5000, "high": 10000}
    budget: int | None = effort_budgets.get(effort)
    model_lower = (model or "").lower()
    if budget is None:
        if model_lower.startswith("o3"):
            budget = 10000
        elif model_lower.startswith("o1-mini"):
            budget = 3000
        elif model_lower.startswith("o1"):
            budget = 5000
    if budget is None:
        return None
    return {"type": "enabled", "budget_tokens": budget}


def convert_anthropic_message_to_response_object(
    response: anthropic_models.MessageResponse,
) -> openai_models.ResponseObject:
    text_parts: list[str] = []
    tool_contents: list[dict[str, Any]] = []
    for block in response.content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text_parts.append(getattr(block, "text", ""))
        elif block_type == "thinking":
            thinking = getattr(block, "thinking", None) or ""
            signature = getattr(block, "signature", None)
            sig_attr = (
                f' signature="{signature}"'
                if isinstance(signature, str) and signature
                else ""
            )
            text_parts.append(f"<thinking{sig_attr}>{thinking}</thinking>")
        elif block_type == "tool_use":
            tool_contents.append(
                {
                    "type": "tool_use",
                    "id": getattr(block, "id", "tool_1"),
                    "name": getattr(block, "name", "function"),
                    "arguments": getattr(block, "input", {}) or {},
                }
            )

    message_content: list[dict[str, Any]] = []
    if text_parts:
        message_content.append(
            openai_models.OutputTextContent(
                type="output_text",
                text="".join(text_parts),
            ).model_dump()
        )
    message_content.extend(tool_contents)

    usage_model = None
    if response.usage is not None:
        usage_model = convert_anthropic_usage_to_openai_response_usage(response.usage)

    return openai_models.ResponseObject(
        id=response.id,
        object="response",
        created_at=0,
        status="completed",
        model=response.model,
        output=[
            openai_models.MessageOutput(
                type="message",
                id=f"{response.id}_msg_0",
                status="completed",
                role="assistant",
                content=message_content,  # type: ignore[arg-type]
            )
        ],
        parallel_tool_calls=False,
        usage=usage_model,
    )


__all__ = [
    "OpenAIResponsesToAnthropicAdapter",
    "convert_openai_response_to_anthropic_message",
]
