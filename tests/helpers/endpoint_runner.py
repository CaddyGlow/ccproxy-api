"""Shared utilities for endpoint runner sample tests."""

from __future__ import annotations

import os
from collections.abc import Iterable

import pytest

from ccproxy.testing.endpoints import (
    ENDPOINT_TESTS,
    EndpointRequestResult,
    EndpointTestResult,
    TestEndpoint,
)
from ccproxy.testing.endpoints.config import REQUEST_DATA
from tests.conftest import ENDPOINT_TEST_SELECTION_ENV, get_selected_endpoint_indices
from tests.helpers.sample_loader import load_sample_registry


BASE_URL = "http://test"

SAMPLE_REGISTRY = load_sample_registry()
AVAILABLE_CASES = {case.name: case for case in ENDPOINT_TESTS}
SAMPLE_NAMES = [name for name in sorted(SAMPLE_REGISTRY) if name in AVAILABLE_CASES]
CASE_INDEX_LOOKUP = {case.name: index for index, case in enumerate(ENDPOINT_TESTS)}

# Prevent pytest from collecting TestEndpoint as a test class
TestEndpoint.__test__ = False  # type: ignore[attr-defined]


def _resolve_sample_names() -> list[str]:
    selection = os.getenv(ENDPOINT_TEST_SELECTION_ENV)
    if not selection:
        return SAMPLE_NAMES

    indices = get_selected_endpoint_indices(selection)
    return [SAMPLE_NAMES[idx] for idx in indices if 0 <= idx < len(SAMPLE_NAMES)]


SELECTED_SAMPLE_NAMES = _resolve_sample_names()


def provider_sample_names(provider: str) -> list[str]:
    """Return sample names filtered for the given provider prefix."""

    prefix = f"{provider}_"
    filtered = [name for name in SELECTED_SAMPLE_NAMES if name.startswith(prefix)]

    # If selection env filtered out all entries for this provider, fall back to full set.
    if not filtered:
        filtered = [name for name in SAMPLE_NAMES if name.startswith(prefix)]
    return filtered


DEFAULT_VALIDATION_FIELDS = {
    "openai": "choices",
    "responses": "output",
    "anthropic": "content",
}


def assert_initial_request(
    result: EndpointTestResult,
    expected_stream: bool,
    request_key: str,
) -> EndpointRequestResult:
    """Validate the first recorded request matches expectations."""

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


def assert_follow_up_requests(result: EndpointTestResult) -> None:
    """Ensure subsequent requests succeed without errors."""

    for extra in result.request_results[1:]:
        status = extra.status_code
        if status is None:
            continue
        error_detail = extra.details.get("error_detail")
        suffix = f": {error_detail}" if error_detail else ""
        assert status == 200, (
            f"Expected HTTP 200 for follow-up request, got {status}{suffix}"
        )


def iter_request_results(result: EndpointTestResult) -> Iterable[EndpointRequestResult]:
    """Return an iterator over recorded request results."""

    return iter(result.request_results)
