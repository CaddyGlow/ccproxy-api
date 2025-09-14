from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from pydantic import BaseModel, TypeAdapter

from ccproxy.llms.adapters.base import BaseAPIAdapter
from ccproxy.llms.anthropic import models as anthropic_models
from ccproxy.llms.openai import models as openai_models


class AnthropicMessagesToOpenAIResponsesAdapter(BaseAPIAdapter):
    """Anthropic Messages ↔ OpenAI Responses adapter (request + response subset).

    Implemented
    - adapt_request: Anthropic → OpenAI Responses request
      - Maps `model`, `stream`, `max_tokens` → `max_output_tokens`
      - Maps last user message text into a single Responses `input` message
      - Maps custom tools → function tools
      - Maps tool_choice (auto/any/tool/none) and `parallel_tool_calls`
      - Maps `system` into `instructions` when present
    - adapt_response: Anthropic → OpenAI Responses response
      - Serializes `thinking` blocks into a single OutputTextContent using
        <thinking signature="…">…</thinking> XML followed by visible text
      - Maps ToolUseBlock into content dicts with type "tool_use"
      - Maps usage fields into OpenAI ResponseUsage

    TODO
    - Expand request mapping to include multiple turns and multimodal content
    - Include richer content types beyond text and tool use
    - Consider mapping stop reasons to status/incomplete details
    """

    def __init__(self) -> None:
        super().__init__(name="anthropic_messages_to_openai_responses")

    # Model conversion helpers
    def _dict_to_request_model(self, request: dict[str, Any]) -> BaseModel:
        """Convert dict to CreateMessageRequest."""
        # Preprocess tools to satisfy union discriminator if missing
        req_dict = dict(request)
        tools_in = req_dict.get("tools")
        if isinstance(tools_in, list):
            new_tools = []
            for t in tools_in:
                if isinstance(t, dict) and "type" not in t:
                    t = {"type": "custom", **t}
                new_tools.append(t)
            req_dict["tools"] = new_tools
        return anthropic_models.CreateMessageRequest.model_validate(req_dict)

    def _dict_to_response_model(self, response: dict[str, Any]) -> BaseModel:
        """Convert dict to MessageResponse."""
        return anthropic_models.MessageResponse.model_validate(response)

    def _dict_to_error_model(self, error: dict[str, Any]) -> BaseModel:
        """Convert dict to ErrorResponse."""
        return anthropic_models.ErrorResponse.model_validate(error)

    def _dict_stream_to_typed_stream(
        self, stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncIterator[BaseModel]:
        """Convert dict stream to MessageStreamEvent stream."""

        event_adapter: TypeAdapter[anthropic_models.MessageStreamEvent] = TypeAdapter(
            anthropic_models.MessageStreamEvent
        )

        async def typed_generator() -> AsyncIterator[BaseModel]:
            async for chunk_dict in stream:
                try:
                    yield event_adapter.validate_python(chunk_dict)
                except Exception:
                    # Skip invalid chunks in stream
                    continue

        return typed_generator()

    # New strongly-typed methods
    async def adapt_request_typed(self, request: BaseModel) -> BaseModel:
        """Convert Anthropic CreateMessageRequest to OpenAI ResponseRequest."""
        if not isinstance(request, anthropic_models.CreateMessageRequest):
            raise ValueError(f"Expected CreateMessageRequest, got {type(request)}")

        return await self._convert_request_typed(request)

    async def adapt_response_typed(self, response: BaseModel) -> BaseModel:
        """Convert Anthropic MessageResponse to OpenAI ResponseObject."""
        if not isinstance(response, anthropic_models.MessageResponse):
            raise ValueError(f"Expected MessageResponse, got {type(response)}")

        return await self._convert_response_typed(response)

    def adapt_stream_typed(
        self, stream: AsyncIterator[BaseModel]
    ) -> AsyncGenerator[BaseModel, None]:
        """Convert streams - not implemented yet."""
        raise NotImplementedError("Stream adaptation not implemented for this adapter")

    async def adapt_error_typed(self, error: BaseModel) -> BaseModel:
        """Convert error response - pass through for now."""
        return error

    # Implementation methods
    async def _convert_request_typed(
        self, request: anthropic_models.CreateMessageRequest
    ) -> openai_models.ResponseRequest:
        """Convert Anthropic CreateMessageRequest to OpenAI ResponseRequest using typed models."""
        # For now, delegate to the existing dict-based implementation
        # TODO: Rewrite this to work directly with typed models for better performance
        request_dict = request.model_dump()
        result_dict = await self._adapt_request_dict_impl(request_dict)
        return openai_models.ResponseRequest.model_validate(result_dict)

    async def _convert_response_typed(
        self, response: anthropic_models.MessageResponse
    ) -> openai_models.ResponseObject:
        """Convert Anthropic MessageResponse to OpenAI ResponseObject using typed models."""
        # For now, delegate to the existing dict-based implementation
        # TODO: Rewrite this to work directly with typed models for better performance
        response_dict = response.model_dump()
        result_dict = await self._adapt_response_dict_impl(response_dict)
        return openai_models.ResponseObject.model_validate(result_dict)

    async def _adapt_request_dict_impl(self, request: dict[str, Any]) -> dict[str, Any]:
        """Implementation moved from adapt_request - works with dicts."""
        # Preprocess tools to satisfy union discriminator if missing
        req_dict = dict(request)
        tools_in = req_dict.get("tools")
        if isinstance(tools_in, list):
            new_tools = []
            for t in tools_in:
                if isinstance(t, dict) and "type" not in t:
                    t = {"type": "custom", **t}
                new_tools.append(t)
            req_dict["tools"] = new_tools

        # Validate as Anthropic CreateMessageRequest
        anthropic_request = anthropic_models.CreateMessageRequest.model_validate(
            req_dict
        )

        # Build OpenAI Responses request payload
        payload: dict[str, Any] = {
            "model": anthropic_request.model,
        }

        if anthropic_request.max_tokens is not None:
            payload["max_output_tokens"] = int(anthropic_request.max_tokens)
        if anthropic_request.stream:
            payload["stream"] = True

        # Map system to instructions if present
        if anthropic_request.system:
            if isinstance(anthropic_request.system, str):
                payload["instructions"] = anthropic_request.system
            else:
                payload["instructions"] = "".join(
                    block.text for block in anthropic_request.system
                )

        # Map last user message text to Responses input
        last_user_text: str | None = None
        for msg in reversed(anthropic_request.messages):
            if msg.role == "user":
                if isinstance(msg.content, str):
                    last_user_text = msg.content
                elif isinstance(msg.content, list):
                    texts: list[str] = []
                    for block in msg.content:
                        # Support raw dicts and models
                        if isinstance(block, dict):
                            if block.get("type") == "text" and isinstance(
                                block.get("text"), str
                            ):
                                texts.append(block.get("text") or "")
                        else:
                            # Type guard for TextBlock
                            if (
                                getattr(block, "type", None) == "text"
                                and hasattr(block, "text")
                                and isinstance(getattr(block, "text", None), str)
                            ):
                                texts.append(block.text or "")
                    if texts:
                        last_user_text = " ".join(texts)
                break

        if last_user_text:
            payload["input"] = [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": last_user_text},
                    ],
                }
            ]

        # Tools mapping (custom tools -> function tools)
        if anthropic_request.tools:
            tools: list[dict[str, Any]] = []
            for tool in anthropic_request.tools:
                if isinstance(tool, anthropic_models.Tool):
                    tools.append(
                        {
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": tool.input_schema,
                            },
                        }
                    )
            if tools:
                payload["tools"] = tools

        # tool_choice mapping (+ parallel control)
        tc = anthropic_request.tool_choice
        if tc is not None:
            tc_type = getattr(tc, "type", None)
            if tc_type == "none":
                payload["tool_choice"] = "none"
            elif tc_type == "auto":
                payload["tool_choice"] = "auto"
            elif tc_type == "any":
                payload["tool_choice"] = "required"
            elif tc_type == "tool":
                name = getattr(tc, "name", None)
                if name:
                    payload["tool_choice"] = {
                        "type": "function",
                        "function": {"name": name},
                    }
            disable_parallel = getattr(tc, "disable_parallel_tool_use", None)
            if isinstance(disable_parallel, bool):
                payload["parallel_tool_calls"] = not disable_parallel

        # Validate
        return openai_models.ResponseRequest.model_validate(payload).model_dump()

    # Override to delegate to typed implementation
    async def adapt_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Legacy dict interface - delegates to typed implementation."""
        typed_request = self._dict_to_request_model(request)
        typed_result = await self.adapt_request_typed(typed_request)
        return typed_result.model_dump()

    async def _adapt_response_dict_impl(
        self, response: dict[str, Any]
    ) -> dict[str, Any]:
        """Implementation moved from adapt_response - works with dicts."""
        # Validate incoming as Anthropic MessageResponse (or minimally shaped dict)
        anth = anthropic_models.MessageResponse.model_validate(response)

        # Aggregate thinking blocks (serialized) and text blocks into a single
        # OutputTextContent item, and include tool_use blocks as-is
        text_parts: list[str] = []
        other_contents: list[dict[str, Any]] = []
        for block in anth.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(getattr(block, "text", ""))
            elif btype == "thinking":
                thinking = getattr(block, "thinking", None) or ""
                signature = getattr(block, "signature", None)
                sig_attr = (
                    f' signature="{signature}"'
                    if isinstance(signature, str) and signature
                    else ""
                )
                text_parts.append(f"<thinking{sig_attr}>{thinking}</thinking>")
            elif btype == "tool_use":
                other_contents.append(
                    {
                        "type": "tool_use",
                        "id": getattr(block, "id", "tool_1"),
                        "name": getattr(block, "name", "function"),
                        # Prefer `input` -> `arguments` for Responses format
                        "arguments": getattr(block, "input", {}) or {},
                    }
                )

        msg_contents: list[dict[str, Any]] = []
        if text_parts:
            msg_contents.append(
                openai_models.OutputTextContent(
                    type="output_text", text="".join(text_parts)
                ).model_dump()
            )
        msg_contents.extend(other_contents)

        # Usage mapping
        usage = anth.usage
        input_tokens_details = openai_models.InputTokensDetails(
            cached_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        )
        output_tokens_details = openai_models.OutputTokensDetails(reasoning_tokens=0)
        resp_usage = openai_models.ResponseUsage(
            input_tokens=int(usage.input_tokens or 0),
            input_tokens_details=input_tokens_details,
            output_tokens=int(usage.output_tokens or 0),
            output_tokens_details=output_tokens_details,
            total_tokens=int((usage.input_tokens or 0) + (usage.output_tokens or 0)),
        )

        response_object = openai_models.ResponseObject(
            id=anth.id,
            object="response",
            created_at=0,
            status="completed",
            model=anth.model,
            output=[
                openai_models.MessageOutput(
                    type="message",
                    id=anth.id,
                    status="completed",
                    role="assistant",
                    content=msg_contents,  # type: ignore[arg-type]
                )
            ],
            parallel_tool_calls=False,
            usage=resp_usage,
        )
        return response_object.model_dump()

    # Override to delegate to typed implementation
    async def adapt_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Legacy dict interface - delegates to typed implementation."""
        typed_response = self._dict_to_response_model(response)
        typed_result = await self.adapt_response_typed(typed_response)
        return typed_result.model_dump()

    def adapt_stream(
        self, stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncGenerator[dict[str, Any], None]:
        raise NotImplementedError

    async def adapt_error(self, error: dict[str, Any]) -> dict[str, Any]:
        return error
