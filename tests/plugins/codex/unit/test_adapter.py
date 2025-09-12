"""Unit tests for CodexAdapter."""

import json
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from ccproxy.plugins.codex.adapter import CodexAdapter
from ccproxy.plugins.codex.detection_service import CodexDetectionService


class TestCodexAdapter:
    """Test the CodexAdapter HTTP adapter methods."""

    @pytest.fixture
    def mock_detection_service(self) -> CodexDetectionService:
        """Create mock detection service."""
        service = Mock(spec=CodexDetectionService)
        service.get_cached_data.return_value = None
        return service

    @pytest.fixture
    def mock_auth_manager(self):
        """Create mock auth manager."""
        auth_manager = Mock()
        auth_data = Mock()
        auth_data.access_token = "test-token"
        auth_data.account_id = "account-123"
        auth_manager.load_credentials = AsyncMock(return_value=auth_data)

        profile = Mock()
        profile.chatgpt_account_id = "test-account-123"
        auth_manager.get_profile_quick = AsyncMock(return_value=profile)
        return auth_manager

    @pytest.fixture
    def mock_http_pool_manager(self):
        """Create mock HTTP pool manager."""
        return Mock()

    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        config = Mock()
        config.base_url = "https://chat.openai.com/backend-anon"
        return config

    @pytest.fixture
    def adapter(
        self,
        mock_detection_service: CodexDetectionService,
        mock_auth_manager,
        mock_http_pool_manager,
        mock_config,
    ) -> CodexAdapter:
        """Create CodexAdapter instance."""
        return CodexAdapter(
            detection_service=mock_detection_service,
            config=mock_config,
            auth_manager=mock_auth_manager,
            http_pool_manager=mock_http_pool_manager,
        )

    @pytest.mark.asyncio
    async def test_get_target_url(self, adapter: CodexAdapter) -> None:
        """Test target URL generation."""
        url = await adapter.get_target_url("/responses")
        assert url == "https://chat.openai.com/backend-anon/responses"

    @pytest.mark.asyncio
    async def test_prepare_provider_request_basic(self, adapter: CodexAdapter) -> None:
        """Test basic provider request preparation."""
        body_dict = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-4",
        }
        body = json.dumps(body_dict).encode()
        headers = {
            "content-type": "application/json",
            "authorization": "Bearer old-token",  # Should be overridden
        }

        result_body, result_headers = await adapter.prepare_provider_request(
            body, headers, "/responses"
        )

        # Body should preserve original format but add Codex-specific fields
        result_data = json.loads(result_body.decode())
        assert "messages" in result_data  # Original format preserved
        assert result_data["stream"] is True  # Always set to True for Codex
        assert "instructions" in result_data

        # Headers should be filtered and enhanced
        assert result_headers["content-type"] == "application/json"
        assert result_headers["authorization"] == "Bearer test-token"
        assert result_headers["chatgpt-account-id"] == "test-account-123"
        assert "session_id" in result_headers

    @pytest.mark.asyncio
    async def test_prepare_provider_request_with_instructions(
        self,
        mock_detection_service: CodexDetectionService,
        mock_auth_manager,
        mock_http_pool_manager,
    ) -> None:
        """Test request preparation with custom instructions."""
        # Setup detection service with custom instructions
        cached_data = Mock()
        cached_data.instructions = Mock()
        cached_data.instructions.instructions_field = "You are a Python expert."
        cached_data.headers = None
        mock_detection_service.get_cached_data.return_value = cached_data

        mock_config = Mock()
        mock_config.base_url = "https://chat.openai.com/backend-anon"

        adapter = CodexAdapter(
            detection_service=mock_detection_service,
            config=mock_config,
            auth_manager=mock_auth_manager,
            http_pool_manager=mock_http_pool_manager,
        )

        body_dict = {
            "messages": [{"role": "user", "content": "Write a function"}],
            "model": "gpt-4",
        }
        body = json.dumps(body_dict).encode()
        headers = {"content-type": "application/json"}

        result_body, result_headers = await adapter.prepare_provider_request(
            body, headers, "/responses"
        )

        # Body should have custom instructions
        result_data = json.loads(result_body.decode())
        assert result_data["instructions"] == "You are a Python expert."

    @pytest.mark.asyncio
    async def test_prepare_provider_request_preserves_existing_instructions(
        self, adapter: CodexAdapter
    ) -> None:
        """Test that existing instructions are preserved."""
        body_dict = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-4",
            "instructions": "You are a JavaScript expert.",
        }
        body = json.dumps(body_dict).encode()
        headers = {"content-type": "application/json"}

        result_body, result_headers = await adapter.prepare_provider_request(
            body, headers, "/responses"
        )

        # Should keep existing instructions
        result_data = json.loads(result_body.decode())
        assert result_data["instructions"] == "You are a JavaScript expert."

    @pytest.mark.asyncio
    async def test_prepare_provider_request_sets_stream_true(
        self, adapter: CodexAdapter
    ) -> None:
        """Test that stream is always set to True."""
        body_dict = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-4",
            "stream": False,  # Should be overridden
        }
        body = json.dumps(body_dict).encode()
        headers = {"content-type": "application/json"}

        result_body, result_headers = await adapter.prepare_provider_request(
            body, headers, "/responses"
        )

        # Stream should always be True for Codex
        result_data = json.loads(result_body.decode())
        assert result_data["stream"] is True

    @pytest.mark.asyncio
    async def test_process_provider_response(self, adapter: CodexAdapter) -> None:
        """Test response processing and format conversion."""
        # Mock Codex response format
        codex_response = {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "text", "text": "Hello! How can I help?"}],
                }
            ]
        }
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.content = json.dumps(codex_response).encode()
        mock_response.headers = {
            "content-type": "application/json",
            "x-response-id": "resp-123",
        }

        result = await adapter.process_provider_response(mock_response, "/responses")

        assert result.status_code == 200
        # Adapter now returns response as-is; format conversion handled upstream
        result_data = json.loads(result.body)
        # Should return original Codex response unchanged
        assert result_data == codex_response
        assert result.headers.get("content-type") == "application/json"

    @pytest.mark.asyncio
    async def test_cli_headers_injection(
        self,
        mock_detection_service: CodexDetectionService,
        mock_auth_manager,
        mock_http_pool_manager,
    ) -> None:
        """Test CLI headers injection."""
        # Setup detection service with CLI headers
        cached_data = Mock()
        cached_data.headers = Mock()
        cached_data.headers.to_headers_dict.return_value = {
            "X-CLI-Version": "1.0.0",
            "X-Session-ID": "cli-session-123",
        }
        cached_data.instructions = None
        mock_detection_service.get_cached_data.return_value = cached_data

        mock_config = Mock()
        mock_config.base_url = "https://chat.openai.com/backend-anon"

        adapter = CodexAdapter(
            detection_service=mock_detection_service,
            config=mock_config,
            auth_manager=mock_auth_manager,
            http_pool_manager=mock_http_pool_manager,
        )

        body_dict = {"messages": [{"role": "user", "content": "Hello"}]}
        body = json.dumps(body_dict).encode()
        headers = {"content-type": "application/json"}

        result_body, result_headers = await adapter.prepare_provider_request(
            body, headers, "/responses"
        )

        # Should include CLI headers (normalized to lowercase)
        assert result_headers["x-cli-version"] == "1.0.0"
        assert result_headers["x-session-id"] == "cli-session-123"

    def test_needs_format_conversion(self, adapter: CodexAdapter) -> None:
        """Test format conversion detection."""
        # Format conversion now handled by format chain, adapter always returns False
        assert adapter._needs_format_conversion("/responses") is False
        assert adapter._needs_format_conversion("/chat/completions") is False

    def test_get_instructions_default(self, adapter: CodexAdapter) -> None:
        """Test default instructions when no detection service data."""
        instructions = adapter._get_instructions()
        assert "coding agent" in instructions.lower()

    def test_get_instructions_from_detection_service(
        self,
        mock_detection_service: CodexDetectionService,
        mock_auth_manager,
        mock_http_pool_manager,
    ) -> None:
        """Test instructions from detection service."""
        cached_data = Mock()
        cached_data.instructions = Mock()
        cached_data.instructions.instructions_field = "Custom instructions"
        mock_detection_service.get_cached_data.return_value = cached_data

        mock_config = Mock()
        mock_config.base_url = "https://chat.openai.com/backend-anon"

        adapter = CodexAdapter(
            detection_service=mock_detection_service,
            config=mock_config,
            auth_manager=mock_auth_manager,
            http_pool_manager=mock_http_pool_manager,
        )

        instructions = adapter._get_instructions()
        assert instructions == "Custom instructions"

    @pytest.mark.asyncio
    async def test_auth_data_usage(
        self, adapter: CodexAdapter, mock_auth_manager
    ) -> None:
        """Test that auth data is properly used."""
        body = b'{"messages": []}'
        headers = {"content-type": "application/json"}

        result_body, result_headers = await adapter.prepare_provider_request(
            body, headers, "/responses"
        )

        # Verify auth manager was called
        mock_auth_manager.load_credentials.assert_called_once()

        # Verify auth headers are set
        assert result_headers["authorization"] == "Bearer test-token"
        assert result_headers["chatgpt-account-id"] == "test-account-123"
