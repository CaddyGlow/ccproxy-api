"""Copilot endpoint runner sample coverage."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from ccproxy.testing.endpoints import TestEndpoint
from tests.helpers.endpoint_runner import (
    AVAILABLE_CASES,
    BASE_URL,
    CASE_INDEX_LOOKUP,
    assert_follow_up_requests,
    assert_initial_request,
    provider_sample_names,
)
from tests.helpers.provider_apps import PROVIDER_APP_BUILDERS, PROVIDER_FIXTURES


pytestmark = [
    pytest.mark.integration,
    pytest.mark.e2e,
    pytest.mark.asyncio(loop_scope="module"),
]


PROVIDER = "copilot"
SAMPLES = provider_sample_names(PROVIDER)


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def copilot_endpoint_tester() -> AsyncGenerator[TestEndpoint, None]:
    """Initialize a shared app/client pair for Copilot endpoint tests."""

    app_builder = PROVIDER_APP_BUILDERS[PROVIDER]
    async with app_builder() as app:
        transport = ASGITransport(app=app)
        client = AsyncClient(transport=transport, base_url=BASE_URL)
        tester = TestEndpoint(base_url=BASE_URL, client=client)
        try:
            yield tester
        finally:
            await client.aclose()


@pytest.mark.parametrize("sample_name", SAMPLES, ids=SAMPLES)
async def test_copilot_endpoint_sample(
    sample_name: str,
    request: pytest.FixtureRequest,
    httpx_mock: Any,
    copilot_endpoint_tester: TestEndpoint,
) -> None:
    fixture_name = PROVIDER_FIXTURES.get(PROVIDER)
    if fixture_name:
        request.getfixturevalue(fixture_name)

    endpoint_case = AVAILABLE_CASES[sample_name]
    result = await copilot_endpoint_tester.run_endpoint_test(
        endpoint_case,
        CASE_INDEX_LOOKUP[endpoint_case.name],
    )

    assert_initial_request(result, endpoint_case.stream, endpoint_case.request)
    assert_follow_up_requests(result)

    assert result.success, (
        result.error or f"Endpoint test '{endpoint_case.name}' failed"
    )
