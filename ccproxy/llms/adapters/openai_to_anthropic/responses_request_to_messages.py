from __future__ import annotations

import json
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any, cast

from pydantic import BaseModel

from ccproxy.llms.adapters.base import BaseAPIAdapter
from ccproxy.llms.adapters.shared import (
    DEFAULT_MAX_TOKENS,
    convert_anthropic_usage_to_openai_response_usage,
    convert_openai_error_to_anthropic,
)
from ccproxy.llms.anthropic import models as anthropic_models
from ccproxy.llms.anthropic.models import (
    CreateMessageRequest,
    MessageResponse,
    MessageStreamEvent,
)
from ccproxy.llms.openai import models as openai_models
from ccproxy.llms.openai.models import ResponseRequest


ResponseStreamEvent = (
    openai_models.ResponseCreatedEvent
    | openai_models.ResponseInProgressEvent
    | openai_models.ResponseCompletedEvent
    | openai_models.ResponseOutputTextDeltaEvent
    | openai_models.ResponseFunctionCallArgumentsDoneEvent
    | openai_models.ResponseRefusalDoneEvent
)


class OpenAIResponsesRequestToAnthropicMessagesAdapter(
    BaseAPIAdapter[
        openai_models.ResponseRequest,
        anthropic_models.MessageResponse,
        anthropic_models.MessageStreamEvent,
    ]
):
    """OpenAI Responses request → Anthropic CreateMessageRequest adapter."""

    def __init__(self) -> None:
        super().__init__(name="openai_responses_request_to_anthropic_messages")

    async def adapt_request(self, request: BaseModel) -> BaseModel:
        if not isinstance(request, ResponseRequest):
            raise ValueError(f"Expected ResponseRequest, got {type(request)}")

        return convert_openai_response_request_to_anthropic(request)

    async def adapt_response(self, response: BaseModel) -> BaseModel:
        if not isinstance(response, MessageResponse):
            raise ValueError(f"Expected MessageResponse, got {type(response)}")

        return convert_anthropic_message_to_response_object(response)

    def adapt_stream(
        self, stream: AsyncIterator[MessageStreamEvent]
    ) -> AsyncGenerator[ResponseStreamEvent, None]:
        return self._convert_stream(stream)

    def _convert_stream(
        self, stream: AsyncIterator[MessageStreamEvent]
    ) -> AsyncGenerator[ResponseStreamEvent, None]:
        async def generator() -> AsyncGenerator[ResponseStreamEvent, None]:
            item_id = "msg_stream"
            output_index = 0
            content_index = 0
            model_id = ""
            response_id = ""
            sequence_counter = 0

            async for evt in stream:
                if not hasattr(evt, "type"):
                    continue

                sequence_counter += 1

                if evt.type == "message_start":
                    model_id = evt.message.model or ""
                    response_id = evt.message.id or ""
                    yield openai_models.ResponseCreatedEvent(
                        type="response.created",
                        sequence_number=sequence_counter,
                        response=openai_models.ResponseObject(
                            id=response_id,
                            object="response",
                            created_at=0,
                            status="in_progress",
                            model=model_id,
                            output=[],
                            parallel_tool_calls=False,
                        ),
                    )

                    for block in evt.message.content:
                        if block.type == "thinking":
                            sequence_counter += 1
                            thinking = block.thinking or ""
                            signature = block.signature
                            sig_attr = f' signature="{signature}"' if signature else ""
                            thinking_xml = f"<thinking{sig_attr}>{thinking}</thinking>"
                            yield openai_models.ResponseOutputTextDeltaEvent(
                                type="response.output_text.delta",
                                sequence_number=sequence_counter,
                                item_id=item_id,
                                output_index=output_index,
                                content_index=content_index,
                                delta=thinking_xml,
                            )

                elif evt.type == "content_block_start":
                    if evt.content_block.type == "tool_use":
                        tool_input = evt.content_block.input or {}
                        try:
                            args_str = json.dumps(tool_input, separators=(",", ":"))
                        except Exception:
                            args_str = str(tool_input)

                        yield openai_models.ResponseFunctionCallArgumentsDoneEvent(
                            type="response.function_call_arguments.done",
                            sequence_number=sequence_counter,
                            item_id=item_id,
                            output_index=output_index,
                            arguments=args_str,
                        )

                elif evt.type == "content_block_delta":
                    text = evt.delta.text
                    if text:
                        yield openai_models.ResponseOutputTextDeltaEvent(
                            type="response.output_text.delta",
                            sequence_number=sequence_counter,
                            item_id=item_id,
                            output_index=output_index,
                            content_index=content_index,
                            delta=text,
                        )

                elif evt.type == "message_delta":
                    yield openai_models.ResponseInProgressEvent(
                        type="response.in_progress",
                        sequence_number=sequence_counter,
                        response=openai_models.ResponseObject(
                            id=response_id,
                            object="response",
                            created_at=0,
                            status="in_progress",
                            model=model_id,
                            output=[],
                            parallel_tool_calls=False,
                            usage=cast(
                                openai_models.ResponseUsage,
                                convert_anthropic_usage_to_openai_response_usage(evt.usage),
                            ),
                        ),
                    )
                    if evt.delta.stop_reason == "refusal":
                        sequence_counter += 1
                        yield openai_models.ResponseRefusalDoneEvent(
                            type="response.refusal.done",
                            sequence_number=sequence_counter,
                            item_id=item_id,
                            output_index=output_index,
                            content_index=content_index,
                            refusal="refused",
                        )

                elif evt.type == "message_stop":
                    yield openai_models.ResponseCompletedEvent(
                        type="response.completed",
                        sequence_number=sequence_counter,
                        response=openai_models.ResponseObject(
                            id=response_id,
                            object="response",
                            created_at=0,
                            status="completed",
                            model=model_id,
                            output=[],
                            parallel_tool_calls=False,
                        ),
                    )
                    break

        return generator()

    async def adapt_error(self, error: BaseModel) -> BaseModel:
        return convert_openai_error_to_anthropic(error)


def convert_openai_response_request_to_anthropic(
    request: ResponseRequest,
) -> CreateMessageRequest:
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
                    if isinstance(part, dict) and part.get("type") in {"input_text", "text"}:
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
                if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
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
            if disable_parallel is not None and anth_choice["type"] in {"auto", "any", "tool"}:
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
                    schema = (
                        fmt.get("json_schema")
                        or fmt.get("schema")
                        or {}
                    )
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
                        schema_data = schema_obj.model_dump() if hasattr(schema_obj, "model_dump") else schema_obj
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

    return CreateMessageRequest.model_validate(payload_data)


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
            sig_attr = f' signature="{signature}"' if isinstance(signature, str) and signature else ""
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
    "OpenAIResponsesRequestToAnthropicMessagesAdapter",
    "convert_openai_response_request_to_anthropic",
    "derive_thinking_config",
    "convert_anthropic_message_to_response_object",
]
