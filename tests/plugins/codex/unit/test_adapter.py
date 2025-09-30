"""Unit tests for CodexAdapter."""

import json
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from ccproxy.models.detection import DetectedHeaders, DetectedPrompts
from ccproxy.plugins.codex.adapter import CodexAdapter
from ccproxy.plugins.codex.detection_service import CodexDetectionService
from ccproxy.plugins.oauth_codex.manager import CodexTokenManager


class TestCodexAdapter:
    """Test the CodexAdapter HTTP adapter methods."""

    @pytest.fixture
    def mock_detection_service(self) -> CodexDetectionService:
        """Create mock detection service."""
        service = Mock(spec=CodexDetectionService)
        service.get_cached_data.return_value = None
        service.instructions_value = "Mock detection instructions"
        prompts = DetectedPrompts.from_body(
            {"instructions": service.instructions_value}
        )
        service.get_detected_prompts.return_value = prompts
        service.get_system_prompt.return_value = prompts.instructions_payload()
        headers = DetectedHeaders(
            {
                "session_id": "session-123",
                "chatgpt-account-id": "test-account-123",
                "authorization": "existing-auth",  # will be filtered
            }
        )
        service.get_detected_headers.return_value = headers
        service.get_ignored_headers.return_value = list(
            CodexDetectionService.ignores_header
        )
        service.get_redacted_headers.return_value = list(
            CodexDetectionService.REDACTED_HEADERS
        )
        return service

    @pytest.fixture
    def mock_auth_manager(self):
        """Create mock auth manager."""
        auth_manager = Mock(spec=CodexTokenManager)
        auth_manager.get_access_token = AsyncMock(return_value="test-token")
        auth_manager.get_access_token_with_refresh = AsyncMock(
            return_value="test-token"
        )

        credentials = Mock()
        credentials.access_token = "test-token"
        auth_manager.load_credentials = AsyncMock(return_value=credentials)
        auth_manager.should_refresh = Mock(return_value=False)
        auth_manager.get_token_snapshot = AsyncMock(return_value=None)

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
        prompts = DetectedPrompts.from_body(
            {"instructions": "You are a Python expert."}
        )
        mock_detection_service.get_detected_prompts.return_value = prompts
        mock_detection_service.get_system_prompt.return_value = (
            prompts.instructions_payload()
        )

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
        expected_instructions = getattr(
            adapter.detection_service,
            "instructions_value",
            "Mock detection instructions",
        )
        assert (
            result_data["instructions"]
            == f"{expected_instructions}\nYou are a JavaScript expert."
        )

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
        cli_headers = DetectedHeaders(
            {
                "X-CLI-Version": "1.0.0",
                "X-Session-ID": "cli-session-123",
            }
        )
        mock_detection_service.get_detected_headers.return_value = cli_headers
        prompts = DetectedPrompts.from_body(
            {"instructions": mock_detection_service.instructions_value}
        )
        mock_detection_service.get_detected_prompts.return_value = prompts
        mock_detection_service.get_system_prompt.return_value = (
            prompts.instructions_payload()
        )

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

    def test_get_instructions_default(self, adapter: CodexAdapter) -> None:
        """Test default instructions when no detection service data."""
        instructions = adapter._get_instructions()
        expected = getattr(
            adapter.detection_service,
            "instructions_value",
            "Mock detection instructions",
        )
        assert instructions == expected

    def test_get_instructions_from_detection_service(
        self,
        mock_detection_service: CodexDetectionService,
        mock_auth_manager,
        mock_http_pool_manager,
    ) -> None:
        """Test instructions from detection service."""
        prompts = DetectedPrompts.from_body({"instructions": "Custom instructions"})
        mock_detection_service.get_detected_prompts.return_value = prompts
        mock_detection_service.get_system_prompt.return_value = (
            prompts.instructions_payload()
        )

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
        mock_auth_manager.get_access_token.assert_awaited()

        # Verify auth headers are set
        assert result_headers["authorization"] == "Bearer test-token"
        assert result_headers["chatgpt-account-id"] == "test-account-123"
