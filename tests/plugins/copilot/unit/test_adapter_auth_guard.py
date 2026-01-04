import pytest

from ccproxy.core.errors import AuthenticationError
from ccproxy.plugins.copilot.adapter import CopilotAdapter


class DummyConfig:
    base_url = "https://example"
    api_headers: dict[str, str] = {}


class DummyDetection:
    pass


class DummyHTTPPool:
    pass


@pytest.mark.unit
@pytest.mark.asyncio
async def test_copilot_adapter_raises_auth_error_when_no_manager() -> None:
    # Create adapter with auth_manager=None
    adapter = CopilotAdapter(
        config=DummyConfig(),  # type: ignore[arg-type]
        auth_manager=None,
        detection_service=DummyDetection(),  # type: ignore[arg-type]
        http_pool_manager=DummyHTTPPool(),  # type: ignore[arg-type]
        oauth_provider=None,  # type: ignore[arg-type]
    )

    with pytest.raises(AuthenticationError):
        await adapter.prepare_provider_request(b"{}", {}, "/responses")
