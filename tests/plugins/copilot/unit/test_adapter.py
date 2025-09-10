"""Unit tests for CopilotAdapter."""

from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from ccproxy.plugins.copilot.adapter import CopilotAdapter
from ccproxy.plugins.copilot.config import CopilotConfig
from ccproxy.plugins.copilot.oauth.provider import CopilotOAuthProvider


class TestCopilotAdapter:
    """Test the CopilotAdapter HTTP adapter methods."""

    @pytest.fixture
    def mock_oauth_provider(self) -> CopilotOAuthProvider:
        """Create mock OAuth provider."""
        provider = Mock(spec=CopilotOAuthProvider)
        provider.ensure_copilot_token = AsyncMock(return_value="test-token")
        return provider

    @pytest.fixture
    def config(self) -> CopilotConfig:
        """Create CopilotConfig instance."""
        return CopilotConfig(
            api_headers={
                "Editor-Version": "vscode/1.71.0",
                "Editor-Plugin-Version": "copilot/1.73.8685",
            }
        )

    @pytest.fixture
    def mock_auth_manager(self):
        """Create mock auth manager."""
        return Mock()

    @pytest.fixture
    def mock_http_pool_manager(self):
        """Create mock HTTP pool manager."""
        return Mock()

    @pytest.fixture
    def adapter(
        self,
        mock_oauth_provider: CopilotOAuthProvider,
        config: CopilotConfig,
        mock_auth_manager,
        mock_http_pool_manager,
    ) -> CopilotAdapter:
        """Create CopilotAdapter instance."""
        return CopilotAdapter(
            oauth_provider=mock_oauth_provider,
            config=config,
            auth_manager=mock_auth_manager,
            http_pool_manager=mock_http_pool_manager,
        )

    @pytest.mark.asyncio
    async def test_get_target_url(self, adapter: CopilotAdapter) -> None:
        """Test target URL generation."""
        url = await adapter.get_target_url("/chat/completions")
        assert url == "https://api.githubcopilot.com/chat/completions"

    @pytest.mark.asyncio
    async def test_prepare_provider_request(self, adapter: CopilotAdapter) -> None:
        """Test provider request preparation."""
        body = b'{"messages": [{"role": "user", "content": "Hello"}]}'
        headers = {
            "content-type": "application/json",
            "authorization": "Bearer old-token",  # Should be overridden
            "x-request-id": "old-id",  # Should be overridden
        }

        result_body, result_headers = await adapter.prepare_provider_request(
            body, headers, "/chat/completions"
        )

        # Body should be unchanged
        assert result_body == body

        # Headers should be filtered and enhanced
        assert result_headers["content-type"] == "application/json"
        assert result_headers["authorization"] == "Bearer test-token"
        assert "x-request-id" in result_headers
        assert result_headers["x-request-id"] != "old-id"  # Should be new UUID
        assert result_headers["editor-version"] == "vscode/1.71.0"
        assert result_headers["editor-plugin-version"] == "copilot/1.73.8685"

    @pytest.mark.asyncio
    async def test_process_provider_response_non_streaming(
        self, adapter: CopilotAdapter
    ) -> None:
        """Test non-streaming response processing."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.content = b'{"choices": []}'
        mock_response.headers = {
            "content-type": "application/json",
            "x-response-id": "resp-123",
            "connection": "keep-alive",  # Should be filtered
            "transfer-encoding": "chunked",  # Should be filtered
        }

        result = await adapter.process_provider_response(
            mock_response, "/chat/completions"
        )

        assert result.status_code == 200
        assert result.body == b'{"choices": []}'
        assert "Content-Type" in result.headers
        assert result.headers["Content-Type"] == "application/json"
        assert "X-Response-Id" in result.headers
        assert result.headers["X-Response-Id"] == "resp-123"
        # Filtered headers should not be present
        assert "Connection" not in result.headers
        assert "Transfer-Encoding" not in result.headers

    @pytest.mark.asyncio
    async def test_process_provider_response_streaming(
        self, adapter: CopilotAdapter
    ) -> None:
        """Test streaming response processing."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {
            "content-type": "text/event-stream",
            "x-response-id": "resp-123",
        }

        # Mock the async iterator
        async def mock_aiter_bytes():
            yield b"data: chunk1\n\n"
            yield b"data: chunk2\n\n"

        mock_response.aiter_bytes = mock_aiter_bytes

        result = await adapter.process_provider_response(
            mock_response, "/chat/completions"
        )

        assert result.status_code == 200
        assert hasattr(result, "body_iterator")  # StreamingResponse
        assert "Content-Type" in result.headers
        assert result.headers["Content-Type"] == "text/event-stream"
        assert "X-Response-Id" in result.headers
        assert result.headers["X-Response-Id"] == "resp-123"

    @pytest.mark.asyncio
    async def test_oauth_provider_token_call(
        self,
        mock_oauth_provider: CopilotOAuthProvider,
        config: CopilotConfig,
        mock_auth_manager,
        mock_http_pool_manager,
    ) -> None:
        """Test that OAuth provider is called for token."""
        adapter = CopilotAdapter(
            oauth_provider=mock_oauth_provider,
            config=config,
            auth_manager=mock_auth_manager,
            http_pool_manager=mock_http_pool_manager,
        )

        await adapter.prepare_provider_request(b"{}", {}, "/chat/completions")

        mock_oauth_provider.ensure_copilot_token.assert_called_once()

    @pytest.mark.asyncio
    async def test_header_case_handling(self, adapter: CopilotAdapter) -> None:
        """Test that headers are normalized to lowercase."""
        body = b"{}"
        headers = {
            "Content-Type": "application/json",  # Mixed case
            "Authorization": "Bearer old-token",  # Mixed case
        }

        result_body, result_headers = await adapter.prepare_provider_request(
            body, headers, "/chat/completions"
        )

        # Check that all keys are lowercase
        for key in result_headers:
            assert key.islower(), f"Header key '{key}' is not lowercase"

        # Check specific headers are present with correct values
        assert result_headers["content-type"] == "application/json"
        assert result_headers["authorization"] == "Bearer test-token"
