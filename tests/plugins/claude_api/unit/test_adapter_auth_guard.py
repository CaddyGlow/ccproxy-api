import pytest

from ccproxy.core.errors import AuthenticationError
from ccproxy.plugins.claude_api.adapter import ClaudeAPIAdapter


class DummyDetection:
    pass


class DummyConfig:
    base_url = "https://example"
    support_openai_format = False
    system_prompt_injection_mode = "none"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_claude_api_adapter_raises_auth_error_when_no_manager():
    adapter = ClaudeAPIAdapter(
        detection_service=DummyDetection(),
        config=DummyConfig(),
        auth_manager=None,  # type: ignore
        http_pool_manager=None,  # type: ignore
    )
    # Force missing token_manager
    adapter.token_manager = None  # type: ignore

    with pytest.raises(AuthenticationError):
        await adapter._resolve_access_token()
