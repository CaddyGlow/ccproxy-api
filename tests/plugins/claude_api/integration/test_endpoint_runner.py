"""Claude API endpoint runner sample coverage."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient
from tests.helpers.endpoint_runner import (
    AVAILABLE_CASES,
    BASE_URL,
    CASE_INDEX_LOOKUP,
    assert_follow_up_requests,
    assert_initial_request,
    provider_sample_names,
)
from tests.helpers.provider_apps import PROVIDER_APP_BUILDERS, PROVIDER_FIXTURES

from ccproxy.testing.endpoints import TestEndpoint


pytestmark = [
    pytest.mark.integration,
    pytest.mark.e2e,
    pytest.mark.asyncio(loop_scope="function"),
]


PROVIDER = "claude"
SAMPLES = provider_sample_names(PROVIDER)


@pytest.mark.parametrize("sample_name", SAMPLES, ids=SAMPLES)
async def test_claude_endpoint_sample(
    sample_name: str,
    request: pytest.FixtureRequest,
    httpx_mock,
) -> None:
    fixture_name = PROVIDER_FIXTURES.get(PROVIDER)
    if fixture_name:
        request.getfixturevalue(fixture_name)

    app_builder = PROVIDER_APP_BUILDERS[PROVIDER]

    async with app_builder() as app:
        transport = ASGITransport(app=app)
        async with (
            AsyncClient(transport=transport, base_url=BASE_URL) as client,
            TestEndpoint(base_url=BASE_URL, client=client) as tester,
        ):
            endpoint_case = AVAILABLE_CASES[sample_name]
            result = await tester.run_endpoint_test(
                endpoint_case,
                CASE_INDEX_LOOKUP[endpoint_case.name],
            )

    assert_initial_request(result, endpoint_case.stream, endpoint_case.request)
    assert_follow_up_requests(result)

    assert result.success, (
        result.error or f"Endpoint test '{endpoint_case.name}' failed"
    )
