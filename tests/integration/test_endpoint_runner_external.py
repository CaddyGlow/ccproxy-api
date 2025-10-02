"""Legacy integration tests that exercise the endpoint runner against a live server."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from pathlib import Path
from typing import Any

import httpx
import pytest

from ccproxy.testing.endpoints import (
    ENDPOINT_TESTS,
    EndpointRequestResult,
    EndpointTest,
    EndpointTestResult,
    TestEndpoint,
)
from ccproxy.testing.endpoints.config import (
    PROVIDER_TOOL_ACCUMULATORS,
    REQUEST_DATA,
)
from ccproxy.testing.endpoints.runner import get_request_payload
from tests.conftest import get_selected_endpoint_indices


pytestmark = [
    pytest.mark.integration,
    pytest.mark.e2e,
    pytest.mark.slow,
    pytest.mark.network,
    pytest.mark.real_api,
]


TestEndpoint.__test__ = False

CAPTURE_DIR_ENV = "CCPROXY_ENDPOINT_CAPTURE_DIR"
DEFAULT_ENDPOINT_BASE_URL = "http://127.0.0.1:8000"

CAPTURE_TARGETS: dict[str, set[str]] = {
    "codex": {"/codex/v1/responses"},
    "copilot": {"/copilot/v1/chat/completions"},
    "claude": {"/claude/v1/messages"},
}


def _should_capture(case: EndpointTest) -> bool:
    provider = case.name.split("_", 1)[0]
    target_paths = CAPTURE_TARGETS.get(provider)
    if not target_paths:
        return False
    return case.endpoint in target_paths


@pytest.fixture(autouse=True)
async def task_manager_fixture():
    """Disable the global task manager for endpoint integration tests."""
    yield


def _selected_indices() -> list[int]:
    return get_selected_endpoint_indices()


def _endpoint_case_params() -> list[pytest.ParameterSet]:
    indices = _selected_indices()
    if not indices:
        return [
            pytest.param(
                None,
                None,
                marks=pytest.mark.skip("No endpoint tests are configured"),
            )
        ]

    return [
        pytest.param(index, ENDPOINT_TESTS[index], id=ENDPOINT_TESTS[index].name)
        for index in indices
    ]


DEFAULT_VALIDATION_FIELDS = {
    "openai": "choices",
    "responses": "output",
    "anthropic": "content",
}


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    fixture_names = set(metafunc.fixturenames)
    if {"endpoint_case_index", "endpoint_case"} <= fixture_names:
        metafunc.parametrize(
            ("endpoint_case_index", "endpoint_case"),
            _endpoint_case_params(),
        )


def _run_endpoint_case(base_url: str, index: int) -> EndpointTestResult:
    async def _run() -> EndpointTestResult:
        async with TestEndpoint(base_url=base_url) as tester:
            return await tester.run_endpoint_test(ENDPOINT_TESTS[index], index)

    return asyncio.run(_run())


def _assert_initial_request(
    result: EndpointTestResult, expected_stream: bool, request_key: str
) -> EndpointRequestResult:
    assert result.request_results, "Test run should record at least one request"
    initial_request = result.request_results[0]

    assert initial_request.status_code == 200, (
        f"Expected HTTP 200 for initial request, got {initial_request.status_code}"
    )
    assert initial_request.stream == expected_stream, (
        "Recorded request type does not match test configuration"
    )

    if expected_stream:
        if initial_request.details.get("fallback_applied", False):
            pytest.fail(
                "Streaming response fell back to JSON; expected native streamed output"
            )
        assert initial_request.details.get("event_count", 0) > 0, (
            "Streaming test completed without emitting events"
        )
    else:
        payload = initial_request.details.get("response")
        assert isinstance(payload, dict), "Expected JSON response payload"
        assert not payload.get("error"), payload.get("error")

        api_format = REQUEST_DATA[request_key]["api_format"]
        required_field = REQUEST_DATA[request_key].get(
            "validation_field", DEFAULT_VALIDATION_FIELDS.get(api_format)
        )
        if required_field:
            assert payload.get(required_field), (
                f"{api_format} response missing '{required_field}' field"
            )

    return initial_request


def _assert_follow_up_requests(result: EndpointTestResult) -> None:
    for extra in result.request_results[1:]:
        if extra.status_code is None:
            continue
        error_detail = extra.details.get("error_detail")
        suffix = f": {error_detail}" if error_detail else ""
        assert extra.status_code == 200, (
            f"Expected HTTP 200 for follow-up request, got {extra.status_code}{suffix}"
        )


def _capture_case_output(
    base_url: str,
    case: EndpointTest,
    result: EndpointTestResult,
) -> None:
    capture_dir = os.getenv(CAPTURE_DIR_ENV, "").strip()
    if not capture_dir or not _should_capture(case):
        return

    payload = get_request_payload(case)

    async def _fetch_response() -> dict[str, Any]:
        async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
            if case.stream:
                async with client.stream("POST", case.endpoint, json=payload) as resp:
                    events: list[dict[str, Any]] = []
                    buffer = ""
                    async for chunk in resp.aiter_text():
                        if not chunk:
                            continue
                        buffer += chunk
                        while "\n\n" in buffer:
                            raw_event, buffer = buffer.split("\n\n", 1)
                            entry = _parse_sse_chunk(raw_event)
                            if entry:
                                events.append(entry)
                    if buffer.strip():
                        trailing = _parse_sse_chunk(buffer)
                        if trailing:
                            events.append(trailing)

                    return {
                        "mode": "stream",
                        "status_code": resp.status_code,
                        "headers": dict(resp.headers),
                        "events": events,
                        "summary": _summarize_stream_response(case, events),
                    }

            resp = await client.post(case.endpoint, json=payload)
            try:
                body: Any = resp.json()
            except ValueError:
                body = resp.text

            return {
                "mode": "json",
                "status_code": resp.status_code,
                "headers": dict(resp.headers),
                "body": body,
            }

    response_data = asyncio.run(_fetch_response())

    output: dict[str, Any] = {
        "name": case.name,
        "endpoint": case.endpoint,
        "stream": case.stream,
        "model": case.model,
        "request": {
            "payload": payload,
            "request_key": case.request,
        },
        "response": response_data,
        "metadata": {
            "success": result.success,
            "error": result.error,
        },
    }

    target_path = Path(capture_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    output_path = target_path / f"{case.name}.json"
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _parse_sse_chunk(raw_chunk: str) -> dict[str, Any] | None:
    lines = [line.strip() for line in raw_chunk.splitlines() if line.strip()]
    if not lines:
        return None

    event_name: str | None = None
    data_lines: list[str] = []

    for line in lines:
        if line.startswith("event:"):
            event_name = line[len("event:") :].strip() or None
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())

    data_combined = "\n".join(data_lines).strip()
    entry: dict[str, Any] = {}
    if event_name:
        entry["event"] = event_name
    if data_combined:
        entry["data"] = data_combined
        if data_combined == "[DONE]":
            entry["done"] = True
        else:
            with contextlib.suppress(ValueError):
                entry["json"] = json.loads(data_combined)

    return entry or None


def _summarize_stream_response(
    case: EndpointTest, events: list[dict[str, Any]]
) -> dict[str, Any] | None:
    if not events:
        return None

    template = REQUEST_DATA.get(case.request, {})
    chunk_model_class = template.get("chunk_model_class")
    provider_key = case.name.split("_", 1)[0]
    accumulator_class = template.get(
        "accumulator_class"
    ) or PROVIDER_TOOL_ACCUMULATORS.get(provider_key)

    accumulator = accumulator_class() if accumulator_class else None
    full_content_parts: list[str] = []
    finish_reason: str | None = None
    processed_events = 0

    for entry in events:
        data = entry.get("json")
        if not isinstance(data, dict):
            continue

        event_name = entry.get("event") or ""
        if accumulator:
            accumulator.accumulate(event_name, data)

        processed_events += 1

        if "choices" in data:
            for choice in data.get("choices", []):
                delta = choice.get("delta", {}) if isinstance(choice, dict) else {}
                content_piece = delta.get("content")
                if isinstance(content_piece, str):
                    full_content_parts.append(content_piece)

                finish = (
                    choice.get("finish_reason") if isinstance(choice, dict) else None
                )
                if finish:
                    finish_reason = finish

    if processed_events == 0:
        return None

    summary: dict[str, Any] = {
        "processed_events": processed_events,
    }

    if full_content_parts:
        summary["full_content"] = "".join(full_content_parts)
    if finish_reason:
        summary["finish_reason"] = finish_reason

    if accumulator:
        aggregated_snapshot = accumulator.rebuild_response_object(
            {"choices": [], "content": [], "tool_calls": []}
        )
        if isinstance(aggregated_snapshot, dict):
            summary["aggregated_response"] = aggregated_snapshot

    if chunk_model_class:
        summary["chunk_model_class"] = getattr(
            chunk_model_class, "__name__", str(chunk_model_class)
        )

    return summary


def test_endpoint_case_passes(
    endpoint_case_index: int,
    endpoint_case: EndpointTest,
) -> None:
    base_url = DEFAULT_ENDPOINT_BASE_URL

    try:
        httpx.get(base_url, timeout=2.0)
    except (httpx.HTTPError, OSError) as exc:
        pytest.skip(f"Endpoint test server not reachable at {base_url}: {exc}")

    result = _run_endpoint_case(base_url, endpoint_case_index)

    _assert_initial_request(result, endpoint_case.stream, endpoint_case.request)
    _assert_follow_up_requests(result)

    _capture_case_output(base_url, endpoint_case, result)

    assert result.success, (
        result.error or f"Endpoint test '{endpoint_case.name}' failed"
    )
