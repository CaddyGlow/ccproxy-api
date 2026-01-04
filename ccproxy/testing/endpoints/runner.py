"""Async endpoint test runner implementation."""

from __future__ import annotations

import ast
import asyncio
import copy
import json
import re
from collections.abc import Iterable, Sequence
from typing import Any, Literal, overload

import httpx
import structlog

from ccproxy.llms.models.openai import ResponseMessage, ResponseObject
from ccproxy.llms.streaming.accumulators import StreamAccumulator

from .config import (
    ENDPOINT_TESTS,
    FORMAT_TOOLS,
    PROVIDER_TOOL_ACCUMULATORS,
    REQUEST_DATA,
)
from .console import (
    colored_error,
    colored_header,
    colored_info,
    colored_progress,
    colored_success,
    colored_warning,
)
from .models import (
    EndpointRequestResult,
    EndpointTest,
    EndpointTestResult,
    EndpointTestRunSummary,
)
from .tools import handle_tool_call


logger = structlog.get_logger(__name__)


def extract_thinking_blocks(content: str) -> list[tuple[str, str]]:
    """Extract thinking blocks from content."""
    thinking_pattern = r'<thinking signature="([^"]*)">(.*?)</thinking>'
    matches = re.findall(thinking_pattern, content, re.DOTALL)
    return matches


def extract_visible_content(content: str) -> str:
    """Extract only the visible content (not thinking blocks)."""
    thinking_pattern = r'<thinking signature="[^"]*">.*?</thinking>'
    return re.sub(thinking_pattern, "", content, flags=re.DOTALL).strip()


def get_request_payload(test: EndpointTest) -> dict[str, Any]:
    """Get formatted request payload for a test, excluding validation classes."""
    template = REQUEST_DATA[test.request].copy()

    validation_keys = {
        "model_class",
        "chunk_model_class",
        "accumulator_class",
        "api_format",
    }
    template = {k: v for k, v in template.items() if k not in validation_keys}

    def format_value(value: Any) -> Any:
        if isinstance(value, str):
            return value.format(model=test.model)
        if isinstance(value, dict):
            return {k: format_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [format_value(item) for item in value]
        return value

    formatted_template = format_value(template)
    # Type assertion for mypy - we know the format_value function preserves the dict type
    return formatted_template  # type: ignore[no-any-return]


class TestEndpoint:
    """Test endpoint utility for CCProxy API testing."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        trace: bool = False,
        *,
        cors_origin: str | None = None,
        default_headers: dict[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
    ):
        self.base_url = base_url
        self.trace = trace
        self.cors_origin = cors_origin
        self.base_headers: dict[str, str] = {"Accept-Encoding": "identity"}

        if default_headers:
            self.base_headers.update(default_headers)

        if self.cors_origin:
            self.base_headers["Origin"] = self.cors_origin

        if client is None:
            self.client = httpx.AsyncClient(
                timeout=30.0,
                headers=self.base_headers.copy(),
            )
        else:
            self.client = client
            # Ensure client carries required defaults without overwriting explicit values
            for key, value in self.base_headers.items():
                if key not in self.client.headers:
                    self.client.headers[key] = value

    async def __aenter__(self) -> TestEndpoint:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:  # noqa: D401
        await self.client.aclose()

    def _build_headers(self, extra: dict[str, Any] | None = None) -> dict[str, str]:
        """Compose request headers for requests made by the tester."""

        headers = self.base_headers.copy()
        if extra:
            headers.update(extra)
        return headers

    def extract_and_display_request_id(
        self,
        headers: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> str | None:
        """Extract request ID from response headers and display it."""
        request_id_headers = [
            "x-request-id",
            "request-id",
            "x-amzn-requestid",
            "x-correlation-id",
            "x-trace-id",
            "traceparent",
        ]

        request_id = None
        context_data = context or {}
        for header_name in request_id_headers:
            for key in [header_name, header_name.lower()]:
                if key in headers:
                    request_id = headers[key]
                    break
            if request_id:
                break

        if request_id:
            print(colored_info(f"-> Request ID: {request_id}"))
            logger.info(
                "Request ID extracted",
                request_id=request_id,
                **context_data,
            )
        else:
            logger.debug(
                "No request ID found in headers",
                available_headers=list(headers.keys()),
                **context_data,
            )

        return request_id

    @overload
    async def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        capture_result: Literal[False] = False,
    ) -> dict[str, Any]: ...

    @overload
    async def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        capture_result: Literal[True],
    ) -> tuple[dict[str, Any], EndpointRequestResult]: ...

    async def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        capture_result: bool = False,
    ) -> dict[str, Any] | tuple[dict[str, Any], EndpointRequestResult]:
        """Post JSON request and return parsed response."""
        request_headers = self._build_headers({"Content-Type": "application/json"})
        if headers:
            request_headers.update(headers)

        context_data = context or {}

        print(colored_info(f"-> Making JSON request to {url}"))
        logger.info(
            "Making JSON request",
            url=url,
            payload_model=payload.get("model"),
            payload_stream=payload.get("stream"),
            **context_data,
        )

        response = await self.client.post(url, json=payload, headers=request_headers)

        logger.info(
            "Received JSON response",
            status_code=response.status_code,
            headers=dict(response.headers),
            **context_data,
        )

        self.extract_and_display_request_id(
            dict(response.headers), context=context_data
        )

        status_code = response.status_code
        response_headers = dict(response.headers)

        parsed_body: dict[str, Any]
        if status_code != 200:
            print(colored_error(f"[ERROR] Request failed: HTTP {status_code}"))
            logger.error(
                "Request failed",
                status_code=status_code,
                response_text=response.text,
                **context_data,
            )
            parsed_body = {"error": f"HTTP {status_code}: {response.text}"}
        else:
            try:
                json_response = response.json()
            except json.JSONDecodeError as exc:  # noqa: TRY003
                logger.error(
                    "Failed to parse JSON response",
                    error=str(exc),
                    **context_data,
                )
                parsed_body = {"error": f"JSON decode error: {exc}"}
            else:
                parsed_body = json_response

        request_result_details: dict[str, Any] = {
            "headers": response_headers,
        }
        if isinstance(parsed_body, dict):
            request_result_details["response"] = parsed_body
            error_detail = parsed_body.get("error")
            if error_detail:
                request_result_details["error_detail"] = error_detail
        else:
            request_result_details["response"] = parsed_body

        request_result = EndpointRequestResult(
            phase=context_data.get("phase", "initial"),
            method="POST",
            status_code=status_code,
            stream=False,
            details=request_result_details,
        )

        if capture_result:
            return parsed_body, request_result

        return parsed_body

    async def post_stream(
        self,
        url: str,
        payload: dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[list[str], list[EndpointRequestResult]]:
        """Post streaming request and return list of SSE events."""
        request_headers = self._build_headers(
            {"Accept": "text/event-stream", "Content-Type": "application/json"}
        )
        if headers:
            request_headers.update(headers)

        context_data = context or {}

        print(colored_info(f"-> Making streaming request to {url}"))
        logger.info(
            "Making streaming request",
            url=url,
            payload_model=payload.get("model"),
            payload_stream=payload.get("stream"),
            **context_data,
        )

        events: list[str] = []
        raw_chunks: list[str] = []
        request_results: list[EndpointRequestResult] = []
        fallback_request_result: EndpointRequestResult | None = None
        fallback_used = False
        stream_status_code: int | None = None
        primary_event_count = 0

        try:
            async with self.client.stream(
                "POST", url, json=payload, headers=request_headers
            ) as resp:
                logger.info(
                    "Streaming response received",
                    status_code=resp.status_code,
                    headers=dict(resp.headers),
                    **context_data,
                )

                self.extract_and_display_request_id(
                    dict(resp.headers), context=context_data
                )

                stream_status_code = resp.status_code

                if resp.status_code != 200:
                    error_text = await resp.aread()
                    error_message = error_text.decode()
                    print(
                        colored_error(
                            f"[ERROR] Streaming request failed: HTTP {resp.status_code}"
                        )
                    )
                    logger.error(
                        "Streaming request failed",
                        status_code=resp.status_code,
                        response_text=error_message,
                        **context_data,
                    )
                    error_payload = json.dumps(
                        {
                            "error": {
                                "status": resp.status_code,
                                "message": error_message,
                            }
                        },
                        ensure_ascii=False,
                    )
                    events = [f"data: {error_payload}", "data: [DONE]"]
                    request_results.append(
                        EndpointRequestResult(
                            phase=context_data.get("phase", "initial"),
                            method="POST",
                            status_code=stream_status_code,
                            stream=True,
                            details={
                                "event_count": len(events),
                                "error_detail": error_message,
                            },
                        )
                    )
                    return events, request_results

                buffer = ""
                async for chunk in resp.aiter_text():
                    if not chunk:
                        continue

                    # normalized_segments = self._normalize_stream_chunk(chunk)

                    for segment in chunk:  # normalized_segments:
                        if not segment:
                            continue

                        raw_chunks.append(segment)
                        buffer += segment

                        while "\n\n" in buffer:
                            raw_event, buffer = buffer.split("\n\n", 1)
                            if raw_event.strip():
                                events.append(raw_event.strip())

                if buffer.strip():
                    events.append(buffer.strip())

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Streaming request exception",
                error=str(exc),
                **context_data,
            )
            error_payload = json.dumps(
                {"error": {"message": str(exc)}}, ensure_ascii=False
            )
            events.append(f"data: {error_payload}")
            events.append("data: [DONE]")
            request_results.append(
                EndpointRequestResult(
                    phase=context_data.get("phase", "initial"),
                    method="POST",
                    status_code=stream_status_code,
                    stream=True,
                    details={
                        "event_count": len(events),
                        "error_detail": str(exc),
                    },
                )
            )
            return events, request_results

        raw_text = "".join(raw_chunks).strip()
        primary_event_count = len(events)
        only_done = events and all(
            evt.strip().lower() == "data: [done]" for evt in events
        )

        if not events or only_done:
            logger.debug(
                "stream_response_empty",
                event_count=len(events),
                raw_length=len(raw_text),
                **context_data,
            )

            fallback_events: list[str] | None = None

            if raw_text and raw_text.lower() != "data: [done]":
                if raw_text.startswith("data:"):
                    fallback_events = [raw_text, "data: [DONE]"]
                else:
                    fallback_events = [f"data: {raw_text}", "data: [DONE]"]
            else:
                (
                    fallback_events,
                    fallback_request_result,
                ) = await self._fallback_stream_to_json(
                    url=url,
                    payload=payload,
                    context=context_data,
                )

            if fallback_events:
                logger.info(
                    "stream_fallback_applied",
                    fallback_event_count=len(fallback_events),
                    **context_data,
                )
                events = fallback_events
                fallback_used = True

        events = [evt.rstrip("'\"") if isinstance(evt, str) else evt for evt in events]

        request_details: dict[str, Any] = {
            "event_count": len(events),
        }
        if fallback_used:
            request_details["fallback_applied"] = True
        if primary_event_count and primary_event_count != len(events):
            request_details["primary_event_count"] = primary_event_count
        if raw_text:
            request_details["raw_preview"] = raw_text[:120]

        request_results.append(
            EndpointRequestResult(
                phase=context_data.get("phase", "initial"),
                method="POST",
                status_code=stream_status_code,
                stream=True,
                details=request_details,
            )
        )

        if fallback_request_result is not None:
            request_results.append(fallback_request_result)

        logger.info(
            "Streaming completed",
            event_count=len(events),
            **context_data,
        )
        return events, request_results

    async def options_preflight(
        self,
        url: str,
        *,
        request_method: str = "POST",
        request_headers: Sequence[str] | None = None,
        headers: dict[str, str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """Send a CORS preflight OPTIONS request and return status and headers."""

        preflight_headers = self._build_headers({})
        preflight_headers["Access-Control-Request-Method"] = request_method
        if request_headers:
            preflight_headers["Access-Control-Request-Headers"] = ", ".join(
                request_headers
            )
        if headers:
            preflight_headers.update(headers)

        context_data = context or {}

        print(colored_info(f"-> Making CORS preflight request to {url}"))
        logger.info(
            "Making CORS preflight request",
            url=url,
            request_method=request_method,
            request_headers=request_headers,
            **context_data,
        )

        response = await self.client.options(url, headers=preflight_headers)
        status_code = response.status_code
        response_headers = dict(response.headers)

        logger.info(
            "Preflight response received",
            status_code=status_code,
            headers=response_headers,
            **context_data,
        )

        self.extract_and_display_request_id(response_headers, context=context_data)
        print(colored_info(f"-> Preflight response status: HTTP {status_code}"))
        return status_code, response_headers

    def _normalize_stream_chunk(self, chunk: str) -> list[str]:
        """Decode chunks that arrive as Python bytes reprs (e.g. b'...')."""

        if not chunk:
            return []

        pattern = re.compile(r"b(['\"])(.*?)(?<!\\)\1", re.DOTALL)
        matches = list(pattern.finditer(chunk))

        if not matches:
            return [chunk]

        segments: list[str] = []
        last_end = 0
        for match in matches:
            literal = match.group(0)
            try:
                value = ast.literal_eval(literal)
                if isinstance(value, bytes):
                    segments.append(value.decode("utf-8", "replace"))
                else:
                    segments.append(str(value))
            except Exception:
                segments.append(match.group(2).replace("\\n", "\n"))
            last_end = match.end()

        remainder = chunk[last_end:]
        if remainder.strip():
            segments.append(remainder)

        return segments or [chunk]

    async def _fallback_stream_to_json(
        self,
        *,
        url: str,
        payload: dict[str, Any],
        context: dict[str, Any],
    ) -> tuple[list[str], EndpointRequestResult | None]:
        """Retry streaming request as JSON when no SSE events are emitted."""

        if not isinstance(payload, dict):
            return [], None

        fallback_payload = copy.deepcopy(payload)
        fallback_payload["stream"] = False

        fallback_context = {**context, "phase": context.get("phase", "fallback")}
        fallback_context["fallback"] = "stream_to_json"

        try:
            response, request_result = await self.post_json(
                url,
                fallback_payload,
                context=fallback_context,
                capture_result=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "stream_fallback_request_failed",
                error=str(exc),
                **fallback_context,
            )
            return [], None

        if isinstance(response, dict | list):
            body = json.dumps(response, ensure_ascii=False)
        else:
            body = str(response)

        return [f"data: {body}", "data: [DONE]"], request_result

    def validate_response(
        self, response: dict[str, Any], model_class: Any, *, is_streaming: bool = False
    ) -> bool:
        """Validate response using the provided model_class."""
        try:
            payload = response
            if model_class is ResponseMessage:
                payload = self._extract_openai_responses_message(response)
            elif model_class is ResponseObject and isinstance(payload.get("text"), str):
                try:
                    payload = payload.copy()
                    payload["text"] = json.loads(payload["text"])
                except json.JSONDecodeError:
                    logger.debug(
                        "Failed to decode response.text as JSON",
                        text_value=payload.get("text"),
                    )
            model_class.model_validate(payload)
            print(colored_success(f"[OK] {model_class.__name__} validation passed"))
            logger.info(f"{model_class.__name__} validation passed")
            return True
        except Exception as exc:  # noqa: BLE001
            print(
                colored_error(
                    f"[ERROR] {model_class.__name__} validation failed: {exc}"
                )
            )
            logger.error(f"{model_class.__name__} validation failed", error=str(exc))
            return False

    def _extract_openai_responses_message(
        self, response: dict[str, Any]
    ) -> dict[str, Any]:
        """Coerce various response shapes into an OpenAIResponseMessage dict."""

        try:
            if isinstance(response, dict) and "choices" in response:
                choices = response.get("choices") or []
                if choices and isinstance(choices[0], dict):
                    message = choices[0].get("message")
                    if isinstance(message, dict):
                        return message
        except Exception:  # pragma: no cover - defensive fallback
            pass

        try:
            output = response.get("output") if isinstance(response, dict) else None
            if isinstance(output, list):
                for item in output:
                    if isinstance(item, dict) and item.get("type") == "message":
                        content_blocks = item.get("content") or []
                        text_parts: list[str] = []
                        for block in content_blocks:
                            if (
                                isinstance(block, dict)
                                and block.get("type") in ("text", "output_text")
                                and block.get("text")
                            ):
                                text_parts.append(block["text"])
                        content_text = "".join(text_parts) if text_parts else None
                        return {"role": "assistant", "content": content_text}
        except Exception:  # pragma: no cover - defensive fallback
            pass

        return {"role": "assistant", "content": None}

    def validate_sse_event(self, event: str) -> bool:
        """Validate SSE event structure (basic check)."""
        return event.startswith("data: ")

    def _is_partial_tool_call_chunk(self, chunk: dict[str, Any]) -> bool:
        """Check if chunk contains partial tool call data that shouldn't be validated."""
        if not isinstance(chunk, dict) or "choices" not in chunk:
            return False

        for choice in chunk.get("choices", []):
            if not isinstance(choice, dict):
                continue

            delta = choice.get("delta", {})
            if not isinstance(delta, dict):
                continue

            tool_calls = delta.get("tool_calls", [])
            if not tool_calls:
                continue

            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue

                function = tool_call.get("function", {})
                if isinstance(function, dict):
                    if "arguments" in function and (
                        "name" not in function or not tool_call.get("id")
                    ):
                        return True

        return False

    def _has_tool_calls_in_chunk(self, chunk: dict[str, Any]) -> bool:
        """Check if chunk contains any tool call data."""
        if not isinstance(chunk, dict) or "choices" not in chunk:
            return False

        for choice in chunk.get("choices", []):
            if not isinstance(choice, dict):
                continue

            delta = choice.get("delta", {})
            if isinstance(delta, dict) and "tool_calls" in delta:
                return True

        return False

    def _accumulate_tool_calls(
        self, chunk: dict[str, Any], accumulator: dict[str, dict[str, Any]]
    ) -> None:
        """Accumulate tool call fragments across streaming chunks."""
        if not isinstance(chunk, dict) or "choices" not in chunk:
            return

        for choice in chunk.get("choices", []):
            if not isinstance(choice, dict):
                continue

            delta = choice.get("delta", {})
            if not isinstance(delta, dict) or "tool_calls" not in delta:
                continue

            for tool_call in delta["tool_calls"]:
                if not isinstance(tool_call, dict):
                    continue

                index = tool_call.get("index", 0)
                call_key = f"call_{index}"

                accumulator.setdefault(
                    call_key,
                    {
                        "id": None,
                        "type": None,
                        "function": {"name": None, "arguments": ""},
                    },
                )

                if "id" in tool_call:
                    accumulator[call_key]["id"] = tool_call["id"]

                if "type" in tool_call:
                    accumulator[call_key]["type"] = tool_call["type"]

                function = tool_call.get("function", {})
                if isinstance(function, dict):
                    if "name" in function:
                        accumulator[call_key]["function"]["name"] = function["name"]

                    if "arguments" in function:
                        accumulator[call_key]["function"]["arguments"] += function[
                            "arguments"
                        ]

    def _get_complete_tool_calls(
        self, accumulator: dict[str, dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Extract complete tool calls from accumulator."""
        complete_calls = []

        for call_data in accumulator.values():
            if (
                call_data.get("id")
                and call_data.get("type")
                and call_data["function"].get("name")
                and call_data["function"].get("arguments")
            ):
                complete_calls.append(
                    {
                        "id": call_data["id"],
                        "type": call_data["type"],
                        "function": {
                            "name": call_data["function"]["name"],
                            "arguments": call_data["function"]["arguments"],
                        },
                    }
                )

        return complete_calls

    def _execute_accumulated_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        tool_definitions: list[dict[str, Any]] | None = None,
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute accumulated tool calls and return results."""
        if not tool_calls:
            return []

        print(
            colored_info(
                f"-> {len(tool_calls)} tool call(s) accumulated from streaming"
            )
        )
        context_data = context or {}
        logger.info(
            "Executing accumulated tool calls",
            tool_count=len(tool_calls),
            tool_names=[
                (tool.get("function") or {}).get("name")
                if isinstance(tool, dict)
                else None
                for tool in tool_calls
            ],
            **context_data,
        )

        tool_results = []

        for tool_call in tool_calls:
            try:
                tool_name = None
                tool_arguments: Any = None

                if "function" in tool_call:
                    func = tool_call.get("function", {})
                    tool_name = func.get("name")
                    tool_arguments = func.get("arguments")
                elif "name" in tool_call:
                    tool_name = tool_call.get("name")
                    tool_arguments = tool_call.get("arguments")

                if tool_arguments and isinstance(tool_arguments, str):
                    tool_arguments = json.loads(tool_arguments)

                if tool_definitions:
                    available_names = [
                        tool.get("name")
                        if "name" in tool
                        else tool.get("function", {}).get("name")
                        for tool in tool_definitions
                    ]
                    logger.debug(
                        "Available tool definitions", tool_names=available_names
                    )

                logger.info(
                    "Executing tool call",
                    tool_name=tool_name,
                    **context_data,
                )
                # Ensure tool_name is a string before calling handle_tool_call
                safe_tool_name = str(tool_name) if tool_name is not None else ""
                tool_result = handle_tool_call(safe_tool_name, tool_arguments or {})
                tool_results.append(
                    {
                        "tool_call": tool_call,
                        "result": tool_result,
                        "tool_name": tool_name,
                        "tool_input": tool_arguments,
                    }
                )
                print(
                    colored_success(
                        f"-> Tool result: {json.dumps(tool_result, indent=2)}"
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Tool execution failed",
                    error=str(exc),
                    tool_call=tool_call,
                    **context_data,
                )
                tool_results.append(
                    {
                        "tool_call": tool_call,
                        "result": {"error": str(exc)},
                        "tool_name": tool_call.get("name"),
                        "tool_input": tool_call.get("function", {}),
                    }
                )
        if tool_results:
            logger.info(
                "Tool calls executed",
                tool_count=len(tool_results),
                **context_data,
            )

        return tool_results

    def handle_tool_calls_in_response(
        self,
        response: dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Handle tool calls in a response and return modified response and tool results."""
        tool_results: list[dict[str, Any]] = []
        context_data = context or {}

        if "choices" in response:
            for choice in response.get("choices", []):
                message = choice.get("message", {})
                if message.get("tool_calls"):
                    print(colored_info("-> Tool calls detected in response"))
                    logger.info(
                        "Tool calls detected in response",
                        tool_call_count=len(message["tool_calls"]),
                        **context_data,
                    )
                    for tool_call in message["tool_calls"]:
                        tool_name = tool_call["function"]["name"]
                        tool_input = json.loads(tool_call["function"]["arguments"])
                        print(colored_info(f"-> Calling tool: {tool_name}"))
                        print(
                            colored_info(
                                f"-> Tool input: {json.dumps(tool_input, indent=2)}"
                            )
                        )

                        logger.info(
                            "Executing tool call",
                            tool_name=tool_name,
                            **context_data,
                        )
                        # Ensure tool_name is a string before calling handle_tool_call
                        safe_tool_name = str(tool_name) if tool_name is not None else ""
                        tool_result = handle_tool_call(safe_tool_name, tool_input)
                        print(
                            colored_success(
                                f"-> Tool result: {json.dumps(tool_result, indent=2)}"
                            )
                        )
                        logger.info(
                            "Tool call completed",
                            tool_name=tool_name,
                            **context_data,
                        )

                        tool_results.append(
                            {
                                "tool_call": tool_call,
                                "result": tool_result,
                                "tool_name": tool_name,
                                "tool_input": tool_input,
                            }
                        )

        if "output" in response:
            for item in response.get("output", []):
                if (
                    isinstance(item, dict)
                    and item.get("type") == "function_call"
                    and item.get("name")
                ):
                    tool_name = item.get("name")
                    tool_arguments = item.get("arguments", "")

                    print(colored_info("-> Tool calls detected in response"))
                    print(colored_info(f"-> Calling tool: {tool_name}"))

                    try:
                        tool_input = (
                            json.loads(tool_arguments)
                            if isinstance(tool_arguments, str)
                            else tool_arguments
                        )
                        print(
                            colored_info(
                                f"-> Tool input: {json.dumps(tool_input, indent=2)}"
                            )
                        )

                        logger.info(
                            "Executing tool call",
                            tool_name=tool_name,
                            **context_data,
                        )
                        # Ensure tool_name is a string before calling handle_tool_call
                        safe_tool_name = str(tool_name) if tool_name is not None else ""
                        tool_result = handle_tool_call(safe_tool_name, tool_input)
                        print(
                            colored_success(
                                f"-> Tool result: {json.dumps(tool_result, indent=2)}"
                            )
                        )
                        logger.info(
                            "Tool call completed",
                            tool_name=tool_name,
                            **context_data,
                        )

                        tool_results.append(
                            {
                                "tool_call": {
                                    "name": tool_name,
                                    "arguments": tool_arguments,
                                },
                                "result": tool_result,
                                "tool_name": tool_name,
                                "tool_input": tool_input,
                            }
                        )
                    except json.JSONDecodeError as exc:
                        print(
                            colored_error(f"-> Failed to parse tool arguments: {exc}")
                        )
                        print(colored_error(f"-> Raw arguments: {tool_arguments}"))
                        tool_results.append(
                            {
                                "tool_call": {
                                    "name": tool_name,
                                    "arguments": tool_arguments,
                                },
                                "result": {
                                    "error": f"Failed to parse arguments: {exc}"
                                },
                                "tool_name": tool_name,
                                "tool_input": None,
                            }
                        )

        if "content" in response:
            for content_block in response.get("content", []):
                if (
                    isinstance(content_block, dict)
                    and content_block.get("type") == "tool_use"
                ):
                    print(colored_info("-> Tool calls detected in response"))
                    tool_name = content_block.get("name")
                    tool_input = content_block.get("input", {})
                    print(colored_info(f"-> Calling tool: {tool_name}"))
                    print(
                        colored_info(
                            f"-> Tool input: {json.dumps(tool_input, indent=2)}"
                        )
                    )

                    logger.info(
                        "Executing tool call",
                        tool_name=tool_name,
                        **context_data,
                    )
                    # Ensure tool_name is a string before calling handle_tool_call
                    safe_tool_name = str(tool_name) if tool_name is not None else ""
                    tool_result = handle_tool_call(safe_tool_name, tool_input)
                    print(
                        colored_success(
                            f"-> Tool result: {json.dumps(tool_result, indent=2)}"
                        )
                    )
                    logger.info(
                        "Tool call completed",
                        tool_name=tool_name,
                        **context_data,
                    )

                    tool_results.append(
                        {
                            "tool_call": content_block,
                            "result": tool_result,
                            "tool_name": tool_name,
                            "tool_input": tool_input,
                        }
                    )

        if tool_results:
            logger.info(
                "Tool call handling completed",
                tool_count=len(tool_results),
                **context_data,
            )

        return response, tool_results

    def display_thinking_blocks(self, content: str) -> None:
        """Display thinking blocks from response content."""
        thinking_blocks = extract_thinking_blocks(content)
        if thinking_blocks:
            print(colored_info("-> Thinking blocks detected"))
            for i, (signature, thinking_content) in enumerate(thinking_blocks, 1):
                print(colored_warning(f"[THINKING BLOCK {i}]"))
                print(colored_warning(f"Signature: {signature}"))
                print(colored_warning("=" * 60))
                print(thinking_content.strip())
                print(colored_warning("=" * 60))

    def display_response_content(self, response: dict[str, Any]) -> None:
        """Display response content with thinking block handling."""
        content = ""

        if "choices" in response:
            for choice in response.get("choices", []):
                message = choice.get("message", {})
                if message.get("content"):
                    content = message["content"]
                    break
        elif "content" in response:
            text_parts = []
            for content_block in response.get("content", []):
                if (
                    isinstance(content_block, dict)
                    and content_block.get("type") == "text"
                ):
                    text_parts.append(content_block.get("text", ""))
            content = "".join(text_parts)
        elif "output" in response:
            text_parts = []
            for item in response.get("output", []):
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "message":
                    for part in item.get("content", []):
                        if isinstance(part, dict) and part.get("type") in {
                            "output_text",
                            "text",
                        }:
                            text_parts.append(part.get("text", ""))
                elif item.get("type") == "reasoning" and item.get("summary"):
                    for part in item.get("summary", []):
                        if isinstance(part, dict) and part.get("text"):
                            text_parts.append(part.get("text"))
            content = "\n".join(text_parts)
        elif isinstance(response.get("text"), str):
            content = response.get("text", "")

        if content:
            self.display_thinking_blocks(content)
            visible_content = extract_visible_content(content)
            if visible_content:
                print(colored_info("-> Response content:"))
                print(visible_content)

    def _consume_stream_events(
        self,
        stream_events: Iterable[str],
        chunk_model_class: Any | None,
        accumulator_class: type[StreamAccumulator] | None,
        *,
        context: dict[str, Any] | None = None,
    ) -> tuple[str, str | None, StreamAccumulator | None, int]:
        """Consume SSE chunks, returning accumulated text, metadata, and count."""

        last_event_name: str | None = None
        full_content = ""
        finish_reason: str | None = None
        accumulator = accumulator_class() if accumulator_class else None
        processed_events = 0
        context_data = context or {}

        for event_chunk in stream_events:
            print(event_chunk)

            for raw_event in event_chunk.strip().split("\n"):
                event = raw_event.strip()
                if not event:
                    continue

                if event.startswith("event: "):
                    last_event_name = event[len("event: ") :].strip()
                    continue

                if not self.validate_sse_event(event) or event.endswith("[DONE]"):
                    continue

                try:
                    data = json.loads(event[6:])
                except json.JSONDecodeError:
                    logger.warning(
                        "Invalid JSON in streaming event",
                        event_type=event,
                        **context_data,
                    )
                    continue

                if accumulator:
                    accumulator.accumulate(last_event_name or "", data)

                processed_events += 1

                if isinstance(data, dict):
                    if "choices" in data:
                        for choice in data.get("choices", []):
                            delta = choice.get("delta", {})
                            content = delta.get("content")
                            if content:
                                full_content += content

                            finish_reason_value = choice.get("finish_reason")
                            if finish_reason_value:
                                finish_reason = finish_reason_value

                    if chunk_model_class and not self._is_partial_tool_call_chunk(data):
                        self.validate_stream_chunk(data, chunk_model_class)

        return full_content, finish_reason, accumulator, processed_events

    def _get_format_type_for_test(self, test: EndpointTest) -> str:
        """Determine the API format type for a test based on request data configuration."""
        if test.request in REQUEST_DATA:
            request_data = REQUEST_DATA[test.request]
            if "api_format" in request_data:
                api_format = request_data["api_format"]
                return str(api_format)

        raise ValueError(
            f"Missing api_format for request type: {test.request}. Please add to REQUEST_DATA."
        )

    async def run_endpoint_test(
        self, test: EndpointTest, index: int
    ) -> EndpointTestResult:
        """Run a single endpoint test and return its result."""
        request_log: list[EndpointRequestResult] = []

        try:
            full_url = f"{self.base_url}{test.endpoint}"
            provider_key = test.name.split("_", 1)[0]
            payload = get_request_payload(test)

            log_context = {
                "test_name": test.name,
                "endpoint": test.endpoint,
                "model": test.model,
                "stream": test.stream,
            }

            template = REQUEST_DATA[test.request]
            model_class = template.get("model_class")
            chunk_model_class = template.get("chunk_model_class")
            accumulator_class = template.get(
                "accumulator_class"
            ) or PROVIDER_TOOL_ACCUMULATORS.get(provider_key)

            has_tools = "tools" in payload

            logger.info(
                "Running endpoint test",
                test_name=test.name,
                endpoint=test.endpoint,
                stream=test.stream,
                has_tools=has_tools,
                accumulator_class=getattr(accumulator_class, "__name__", None)
                if accumulator_class
                else None,
                model_class=getattr(model_class, "__name__", None)
                if model_class
                else None,
            )

            if has_tools:
                print(colored_info("-> This test includes function tools"))

            if test.stream:
                stream_events, stream_request_results = await self.post_stream(
                    full_url,
                    payload,
                    context={**log_context, "phase": "initial"},
                )
                request_log.extend(stream_request_results)

                (
                    full_content,
                    finish_reason,
                    stream_accumulator,
                    processed_events,
                ) = self._consume_stream_events(
                    stream_events,
                    chunk_model_class,
                    accumulator_class,
                    context={**log_context, "phase": "initial"},
                )

                if (
                    not full_content
                    and stream_accumulator
                    and getattr(stream_accumulator, "text_content", None)
                ):
                    full_content = stream_accumulator.text_content

                if processed_events == 0:
                    message = f"{test.name}: streaming response ended without emitting any events"
                    print(colored_warning(message))
                    logger.warning(
                        "Streaming response empty",
                        event_count=processed_events,
                        **log_context,
                    )
                    return EndpointTestResult(
                        test=test,
                        index=index,
                        success=False,
                        error=message,
                        request_results=request_log,
                    )

                logger.info(
                    "Stream events processed",
                    event_count=processed_events,
                    finish_reason=finish_reason,
                    content_preview=(full_content[:120] if full_content else None),
                    has_tools=has_tools,
                    **log_context,
                )

                if full_content:
                    self.display_thinking_blocks(full_content)
                    visible_content = extract_visible_content(full_content)
                    if visible_content:
                        print(colored_info("-> Accumulated response:"))
                        print(visible_content)

                if stream_accumulator and processed_events > 0:
                    aggregated_snapshot = stream_accumulator.rebuild_response_object(
                        {"choices": [], "content": [], "tool_calls": []}
                    )
                    if any(
                        aggregated_snapshot.get(key)
                        for key in ("choices", "content", "tool_calls", "output")
                    ):
                        print(colored_info("-> Aggregated response object (partial):"))
                        print(json.dumps(aggregated_snapshot, indent=2))
                        self.display_response_content(aggregated_snapshot)
                        logger.debug(
                            "Stream accumulator snapshot",
                            snapshot_keys=[
                                key
                                for key, value in aggregated_snapshot.items()
                                if value
                            ],
                            **log_context,
                        )

                tool_results: list[dict[str, Any]] = []
                if has_tools and stream_accumulator:
                    complete_tool_calls = stream_accumulator.get_complete_tool_calls()
                    if (
                        finish_reason in ["tool_calls", "tool_use"]
                        or complete_tool_calls
                    ):
                        tool_defs = (
                            payload.get("tools") if isinstance(payload, dict) else None
                        )
                        tool_results = self._execute_accumulated_tool_calls(
                            complete_tool_calls,
                            tool_defs,
                            context={**log_context, "phase": "tool_execution"},
                        )

                        if tool_results:
                            print(
                                colored_info(
                                    "-> Sending tool results back to LLM for final response"
                                )
                            )
                            logger.info(
                                "Tool results ready for continuation",
                                tool_count=len(tool_results),
                                **log_context,
                            )

                            format_type = self._get_format_type_for_test(test)

                            response = {
                                "choices": [{"finish_reason": finish_reason}],
                                "content": full_content,
                            }
                            response["tool_calls"] = complete_tool_calls

                            format_tools = FORMAT_TOOLS[format_type]
                            continuation_payload = (
                                format_tools.build_continuation_request(
                                    payload, response, tool_results
                                )
                            )

                            (
                                continuation_events,
                                continuation_request_results,
                            ) = await self.post_stream(
                                full_url,
                                continuation_payload,
                                context={**log_context, "phase": "continuation"},
                            )
                            request_log.extend(continuation_request_results)
                            print(colored_info("Final response (with tool results):"))
                            (
                                continuation_content,
                                _,
                                continuation_accumulator,
                                continuation_events_processed,
                            ) = self._consume_stream_events(
                                continuation_events,
                                chunk_model_class,
                                accumulator_class,
                                context={**log_context, "phase": "continuation"},
                            )

                            if continuation_events_processed == 0:
                                message = f"{test.name}: continuation streaming response contained no events"
                                print(colored_warning(message))
                                logger.warning(
                                    "Continuation response empty",
                                    event_count=continuation_events_processed,
                                    **log_context,
                                )
                                return EndpointTestResult(
                                    test=test,
                                    index=index,
                                    success=False,
                                    error=message,
                                    request_results=request_log,
                                )

                            logger.info(
                                "Continuation stream processed",
                                event_count=continuation_events_processed,
                                content_preview=(
                                    continuation_content[:120]
                                    if continuation_content
                                    else None
                                ),
                                **log_context,
                            )

                            if continuation_content:
                                self.display_thinking_blocks(continuation_content)
                                visible_content = extract_visible_content(
                                    continuation_content
                                )
                                if visible_content:
                                    print(colored_info("-> Accumulated response:"))
                                    print(visible_content)

                            if (
                                continuation_accumulator
                                and continuation_events_processed > 0
                            ):
                                aggregated_snapshot = (
                                    continuation_accumulator.rebuild_response_object(
                                        {"choices": [], "content": [], "tool_calls": []}
                                    )
                                )
                                if any(
                                    aggregated_snapshot.get(key)
                                    for key in ("choices", "content", "tool_calls")
                                ):
                                    print(
                                        colored_info(
                                            "-> Aggregated response object (partial):"
                                        )
                                    )
                                    print(json.dumps(aggregated_snapshot, indent=2))
                                    self.display_response_content(aggregated_snapshot)
                                    logger.debug(
                                        "Continuation accumulator snapshot",
                                        snapshot_keys=[
                                            key
                                            for key, value in aggregated_snapshot.items()
                                            if value
                                        ],
                                        **log_context,
                                    )

            else:
                response, initial_request_result = await self.post_json(
                    full_url,
                    payload,
                    context={**log_context, "phase": "initial"},
                    capture_result=True,
                )
                request_log.append(initial_request_result)

                print(json.dumps(response, indent=2))

                json_tool_results: list[dict[str, Any]] = []
                if has_tools:
                    response, json_tool_results = self.handle_tool_calls_in_response(
                        response, context={**log_context, "phase": "tool_detection"}
                    )

                    if json_tool_results:
                        print(
                            colored_info(
                                "-> Sending tool results back to LLM for final response"
                            )
                        )
                        logger.info(
                            "Tool results ready for continuation",
                            tool_count=len(json_tool_results),
                            **log_context,
                        )

                        format_type = self._get_format_type_for_test(test)
                        format_tools = FORMAT_TOOLS[format_type]
                        continuation_payload = format_tools.build_continuation_request(
                            payload, response, json_tool_results
                        )

                        (
                            continuation_response,
                            continuation_request_result,
                        ) = await self.post_json(
                            full_url,
                            continuation_payload,
                            context={**log_context, "phase": "continuation"},
                            capture_result=True,
                        )
                        request_log.append(continuation_request_result)
                        print(colored_info("Final response (with tool results):"))
                        print(json.dumps(continuation_response, indent=2))
                        self.display_response_content(continuation_response)
                        preview_data = json.dumps(
                            continuation_response, ensure_ascii=False
                        )
                        logger.info(
                            "Continuation response received",
                            tool_count=len(json_tool_results),
                            content_preview=preview_data[:120],
                            **log_context,
                        )

                self.display_response_content(response)

                if "error" not in response and model_class:
                    self.validate_response(response, model_class, is_streaming=False)

            print(colored_success(f"[OK] Test {test.name} completed successfully"))
            logger.info("Test completed successfully", **log_context)
            return EndpointTestResult(
                test=test,
                index=index,
                success=True,
                request_results=request_log,
            )

        except Exception as exc:  # noqa: BLE001
            print(colored_error(f"[FAIL] Test {test.name} failed: {exc}"))
            logger.error(
                "Test execution failed",
                **log_context,
                error=str(exc),
                exc_info=exc,
            )
            return EndpointTestResult(
                test=test,
                index=index,
                success=False,
                error=str(exc),
                exception=exc,
                request_results=request_log,
            )

    def validate_stream_chunk(
        self, chunk: dict[str, Any], chunk_model_class: Any
    ) -> bool:
        """Validate a streaming chunk against the provided model class."""

        # Some providers emit housekeeping chunks (e.g. pure filter results) that
        # do not include the standard fields expected by the OpenAI schema. Skip
        # validation for those so we only flag real contract violations.
        if not chunk.get("choices") and "model" not in chunk:
            logger.debug(
                "Skipping validation for non-standard chunk",
                chunk_keys=list(chunk.keys()),
            )
            return True

        if chunk.get("type") == "message" and "choices" not in chunk:
            logger.debug(
                "Skipping validation for provider message chunk",
                chunk_type=chunk.get("type"),
                chunk_keys=list(chunk.keys()),
            )
            return True

        try:
            chunk_model_class.model_validate(chunk)
            return True
        except Exception as exc:  # noqa: BLE001
            if self._has_tool_calls_in_chunk(chunk):
                logger.debug(
                    "Validation failed for tool call chunk (expected)", error=str(exc)
                )
                return True

            print(
                colored_error(
                    f"[ERROR] {chunk_model_class.__name__} chunk validation failed: {exc}"
                )
            )
            return False

    async def run_all_tests(
        self, selected_indices: list[int] | None = None
    ) -> EndpointTestRunSummary:
        """Run endpoint tests, optionally filtered by selected indices."""
        print(colored_header("CCProxy Endpoint Tests"))
        print(colored_info(f"Test endpoints at {self.base_url}"))
        logger.info("Starting endpoint tests", base_url=self.base_url)

        total_available = len(ENDPOINT_TESTS)

        if selected_indices is not None:
            indices_to_run = [i for i in selected_indices if 0 <= i < total_available]
            logger.info(
                "Running selected tests",
                selected_count=len(indices_to_run),
                total_count=total_available,
                selected_indices=selected_indices,
            )
        else:
            indices_to_run = list(range(total_available))
            logger.info("Running all tests", test_count=total_available)

        total_to_run = len(indices_to_run)
        print(
            colored_info(
                f"Selected tests: {total_to_run} of {total_available} available"
            )
        )

        if total_to_run == 0:
            print(colored_warning("No tests selected; nothing to execute."))
            logger.warning("No tests selected for execution")
            return EndpointTestRunSummary(
                base_url=self.base_url,
                results=[],
                successful_count=0,
                failure_count=0,
            )

        results: list[EndpointTestResult] = []
        successful_tests = 0
        failed_tests = 0

        for position, index in enumerate(indices_to_run, 1):
            test = ENDPOINT_TESTS[index]

            progress_message = (
                f"[{position}/{total_to_run}] Running test #{index + 1}: {test.name}"
            )
            if test.description and test.description != test.name:
                progress_message += f" - {test.description}"

            print(colored_progress(progress_message))
            logger.info(
                "Dispatching endpoint test",
                test_name=test.name,
                endpoint=test.endpoint,
                ordinal=position,
                total=total_to_run,
                stream=test.stream,
                model=test.model,
            )

            result = await self.run_endpoint_test(test, index)
            results.append(result)

            if result.success:
                successful_tests += 1
            else:
                failed_tests += 1

        error_messages = [result.error for result in results if result.error]

        summary = EndpointTestRunSummary(
            base_url=self.base_url,
            results=results,
            successful_count=successful_tests,
            failure_count=failed_tests,
            errors=error_messages,
        )

        if summary.failure_count == 0:
            print(
                colored_success(
                    f"\nAll {summary.total} endpoint tests completed successfully."
                )
            )
            logger.info(
                "All endpoint tests completed successfully",
                total_tests=summary.total,
                successful=summary.successful_count,
                failed=summary.failure_count,
                error_count=len(summary.errors),
            )
        else:
            print(
                colored_warning(
                    f"\nTest run completed: {summary.successful_count} passed, "
                    f"{summary.failure_count} failed (out of {summary.total})."
                )
            )
            logger.warning(
                "Endpoint tests completed with failures",
                total_tests=summary.total,
                successful=summary.successful_count,
                failed=summary.failure_count,
                errors=summary.errors,
                error_count=len(summary.errors),
            )

            if summary.failed_results:
                print(colored_error("Failed tests:"))
                for failed in summary.failed_results:
                    error_detail = failed.error or "no error message provided"
                    print(
                        colored_error(
                            f"  - {failed.test.name} (#{failed.index + 1}): {error_detail}"
                        )
                    )

            additional_errors = [err for err in summary.errors if err]
            if additional_errors and len(additional_errors) > summary.failure_count:
                print(colored_error("Additional errors:"))
                for err in additional_errors:
                    print(colored_error(f"  - {err}"))

        return summary


def resolve_selected_indices(
    selection: str | Sequence[int] | None,
) -> list[int] | None:
    """Normalize test selection input into 0-based indices."""

    if selection is None:
        return None

    total_tests = len(ENDPOINT_TESTS)

    if isinstance(selection, str):
        indices = parse_test_selection(selection, total_tests)
    else:
        try:
            seen: set[int] = set()
            indices = []
            for raw in selection:
                index = int(raw)
                if index in seen:
                    continue
                seen.add(index)
                indices.append(index)
        except TypeError as exc:
            raise TypeError(
                "tests must be a selection string or a sequence of integers"
            ) from exc

        indices.sort()

    for index in indices:
        if index < 0 or index >= total_tests:
            raise ValueError(
                f"Test index {index} is out of range (0-{total_tests - 1})"
            )

    return indices


def find_tests_by_pattern(pattern: str) -> list[int]:
    """Find test indices by pattern (regex, exact match, or partial match)."""
    pattern_lower = pattern.lower()
    matches: list[int] = []

    for i, test in enumerate(ENDPOINT_TESTS):
        if test.name.lower() == pattern_lower:
            return [i]

    try:
        regex = re.compile(pattern_lower, re.IGNORECASE)
        for i, test in enumerate(ENDPOINT_TESTS):
            if regex.search(test.name.lower()):
                matches.append(i)
        if matches:
            return matches
    except re.error:
        pass

    for i, test in enumerate(ENDPOINT_TESTS):
        if pattern_lower in test.name.lower():
            matches.append(i)

    return matches


def parse_test_selection(selection: str, total_tests: int) -> list[int]:
    """Parse test selection string into list of test indices (0-based)."""
    indices: set[int] = set()

    for part in selection.split(","):
        part = part.strip()

        if ".." in part:
            if part.startswith(".."):
                try:
                    end = int(part[2:])
                    indices.update(range(0, end))
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid range format: '{part}' - ranges must use numbers"
                    ) from exc
            elif part.endswith(".."):
                try:
                    start = int(part[:-2]) - 1
                    indices.update(range(start, total_tests))
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid range format: '{part}' - ranges must use numbers"
                    ) from exc
            else:
                try:
                    start_str, end_str = part.split("..", 1)
                    start = int(start_str) - 1
                    end = int(end_str)
                    indices.update(range(start, end))
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid range format: '{part}' - ranges must use numbers"
                    ) from exc
        else:
            try:
                index = int(part) - 1
                if 0 <= index < total_tests:
                    indices.add(index)
                else:
                    raise ValueError(
                        f"Test index {part} is out of range (1-{total_tests})"
                    )
            except ValueError:
                matched_indices = find_tests_by_pattern(part)
                if matched_indices:
                    indices.update(matched_indices)
                else:
                    suggestions = []
                    part_lower = part.lower()
                    for test in ENDPOINT_TESTS:
                        if any(
                            word in test.name.lower() for word in part_lower.split("_")
                        ):
                            suggestions.append(test.name)

                    error_msg = f"No tests match pattern '{part}'"
                    if suggestions:
                        error_msg += (
                            f". Did you mean one of: {', '.join(suggestions[:3])}"
                        )
                    raise ValueError(error_msg)

    return sorted(indices)


async def run_endpoint_tests_async(
    base_url: str = "http://127.0.0.1:8000",
    tests: str | Sequence[int] | None = None,
) -> EndpointTestRunSummary:
    """Execute endpoint tests asynchronously and return the summary."""

    selected_indices = resolve_selected_indices(tests)
    if selected_indices is not None and not selected_indices:
        raise ValueError("No valid tests selected")

    async with TestEndpoint(base_url=base_url) as tester:
        return await tester.run_all_tests(selected_indices)


def run_endpoint_tests(
    base_url: str = "http://127.0.0.1:8000",
    tests: str | Sequence[int] | None = None,
) -> EndpointTestRunSummary:
    """Convenience wrapper to run endpoint tests from synchronous code."""

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        raise RuntimeError(
            "run_endpoint_tests() cannot be called while an event loop is running; "
            "use await run_endpoint_tests_async(...) instead"
        )

    return asyncio.run(run_endpoint_tests_async(base_url=base_url, tests=tests))


__all__ = [
    "TestEndpoint",
    "run_endpoint_tests",
    "run_endpoint_tests_async",
    "resolve_selected_indices",
    "parse_test_selection",
    "find_tests_by_pattern",
]
