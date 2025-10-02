import pytest

from ccproxy.core.errors import AuthenticationError
from ccproxy.plugins.copilot.adapter import CopilotAdapter


class DummyConfig:
    base_url = "https://example"
    api_headers = {}


class DummyDetection:
    pass


class DummyHTTPPool:
    pass


@pytest.mark.unit
@pytest.mark.asyncio
async def test_copilot_adapter_raises_auth_error_when_no_manager():
    # Create adapter with auth_manager=None
    adapter = CopilotAdapter(
        config=DummyConfig(),
        auth_manager=None,  # type: ignore
        detection_service=DummyDetection(),
        http_pool_manager=DummyHTTPPool(),
        oauth_provider=None,
    )

    with pytest.raises(AuthenticationError):
        await adapter.prepare_provider_request(b"{}", {}, "/responses")
