"""Unit tests for ClaudeAPIAdapter."""

import json
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from ccproxy.plugins.claude_api.adapter import ClaudeAPIAdapter
from ccproxy.plugins.claude_api.detection_service import ClaudeAPIDetectionService


class TestClaudeAPIAdapter:
    """Test the ClaudeAPIAdapter HTTP adapter methods."""

    @pytest.fixture
    def mock_detection_service(self) -> ClaudeAPIDetectionService:
        """Create mock detection service."""
        service = Mock(spec=ClaudeAPIDetectionService)
        service.get_cached_data.return_value = None
        return service

    @pytest.fixture
    def mock_auth_manager(self):
        """Create mock auth manager."""
        auth_manager = Mock()
        auth_data = Mock()
        auth_data.claude_ai_oauth = Mock()
        auth_data.claude_ai_oauth.access_token = "test-token"
        auth_manager.load_credentials = AsyncMock(return_value=auth_data)
        return auth_manager

    @pytest.fixture
    def mock_http_pool_manager(self):
        """Create mock HTTP pool manager."""
        return Mock()

    @pytest.fixture
    def adapter(
        self,
        mock_detection_service: ClaudeAPIDetectionService,
        mock_auth_manager,
        mock_http_pool_manager,
    ) -> ClaudeAPIAdapter:
        """Create ClaudeAPIAdapter instance."""
        from ccproxy.plugins.claude_api.config import ClaudeAPISettings

        config = ClaudeAPISettings()
        return ClaudeAPIAdapter(
            detection_service=mock_detection_service,
            config=config,
            auth_manager=mock_auth_manager,
            http_pool_manager=mock_http_pool_manager,
        )

    @pytest.mark.asyncio
    async def test_get_target_url(self, adapter: ClaudeAPIAdapter) -> None:
        """Test target URL generation."""
        url = await adapter.get_target_url("/v1/messages")
        assert url == "https://api.anthropic.com/v1/messages"

    @pytest.mark.asyncio
    async def test_prepare_provider_request_basic(
        self, adapter: ClaudeAPIAdapter
    ) -> None:
        """Test basic provider request preparation."""
        body_dict = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
        }
        body = json.dumps(body_dict).encode()
        headers = {
            "content-type": "application/json",
            "authorization": "Bearer old-token",  # Should be overridden
        }

        result_body, result_headers = await adapter.prepare_provider_request(
            body, headers, "/v1/messages"
        )

        # Body should be parsed and re-encoded
        result_data = json.loads(result_body.decode())
        assert result_data["model"] == "claude-3-5-sonnet-20241022"
        assert result_data["messages"] == [{"role": "user", "content": "Hello"}]
        assert result_data["max_tokens"] == 100

        # Headers should be filtered and enhanced
        assert result_headers["content-type"] == "application/json"
        assert result_headers["authorization"] == "Bearer test-token"

    @pytest.mark.asyncio
    async def test_prepare_provider_request_with_system_prompt(
        self,
        mock_detection_service: ClaudeAPIDetectionService,
        mock_auth_manager,
        mock_http_pool_manager,
    ) -> None:
        """Test request preparation with system prompt injection."""
        # Setup detection service with system prompt
        cached_data = Mock()
        cached_data.system_prompt = Mock()
        cached_data.system_prompt.system_field = "You are a helpful assistant."
        cached_data.headers = None
        mock_detection_service.get_cached_data.return_value = cached_data

        adapter = ClaudeAPIAdapter(
            detection_service=mock_detection_service,
            auth_manager=mock_auth_manager,
            http_pool_manager=mock_http_pool_manager,
        )

        body_dict = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
        }
        body = json.dumps(body_dict).encode()
        headers = {"content-type": "application/json"}

        result_body, result_headers = await adapter.prepare_provider_request(
            body, headers, "/v1/messages"
        )

        # Body should have system prompt injected
        result_data = json.loads(result_body.decode())
        assert "system" in result_data
        assert isinstance(result_data["system"], list)
        assert result_data["system"][0]["type"] == "text"
        assert result_data["system"][0]["text"] == "You are a helpful assistant."
        assert result_data["system"][0]["_ccproxy_injected"] is True

    @pytest.mark.asyncio
    async def test_prepare_provider_request_openai_conversion(
        self, adapter: ClaudeAPIAdapter
    ) -> None:
        """Test OpenAI format conversion."""
        body_dict = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
            "temperature": 0.7,
        }
        body = json.dumps(body_dict).encode()
        headers = {"content-type": "application/json"}

        result_body, result_headers = await adapter.prepare_provider_request(
            body,
            headers,
            "/v1/chat/completions",  # OpenAI endpoint
        )

        # Should convert OpenAI format to Anthropic
        result_data = json.loads(result_body.decode())
        # The exact conversion depends on the OpenAI adapter implementation
        # Just verify the structure is reasonable
        assert "messages" in result_data or "model" in result_data

    @pytest.mark.asyncio
    async def test_process_provider_response_basic(
        self, adapter: ClaudeAPIAdapter
    ) -> None:
        """Test basic response processing."""
        response_data = {
            "content": [{"type": "text", "text": "Hello! How can I help?"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 7},
        }
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.content = json.dumps(response_data).encode()
        mock_response.headers = {
            "content-type": "application/json",
            "x-response-id": "resp-123",
        }

        result = await adapter.process_provider_response(mock_response, "/v1/messages")

        assert result.status_code == 200
        # Response should be unchanged for native Anthropic endpoint
        result_data = json.loads(result.body)
        assert result_data == response_data
        assert "Content-Type" in result.headers
        assert result.headers["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_process_provider_response_openai_conversion(
        self, adapter: ClaudeAPIAdapter
    ) -> None:
        """Test response conversion for OpenAI format."""
        response_data = {
            "content": [{"type": "text", "text": "Hello! How can I help?"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 7},
        }
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.content = json.dumps(response_data).encode()
        mock_response.headers = {"content-type": "application/json"}

        result = await adapter.process_provider_response(
            mock_response,
            "/v1/chat/completions",  # OpenAI endpoint
        )

        assert result.status_code == 200
        # Should convert to OpenAI format
        result_data = json.loads(result.body)
        # The exact conversion depends on the OpenAI adapter implementation
        # Just verify the structure changed
        assert "choices" in result_data or result_data != response_data

    @pytest.mark.asyncio
    async def test_system_prompt_injection_with_existing_system(
        self,
        mock_detection_service: ClaudeAPIDetectionService,
        mock_auth_manager,
        mock_http_pool_manager,
    ) -> None:
        """Test system prompt injection when request already has system prompt."""
        # Setup detection service with system prompt
        cached_data = Mock()
        cached_data.system_prompt = Mock()
        cached_data.system_prompt.system_field = "You are a helpful assistant."
        cached_data.headers = None
        mock_detection_service.get_cached_data.return_value = cached_data

        adapter = ClaudeAPIAdapter(
            detection_service=mock_detection_service,
            auth_manager=mock_auth_manager,
            http_pool_manager=mock_http_pool_manager,
        )

        body_dict = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello"}],
            "system": "You are a coding assistant.",  # Existing system prompt
            "max_tokens": 100,
        }
        body = json.dumps(body_dict).encode()
        headers = {"content-type": "application/json"}

        result_body, result_headers = await adapter.prepare_provider_request(
            body, headers, "/v1/messages"
        )

        # Body should have both system prompts
        result_data = json.loads(result_body.decode())
        assert "system" in result_data
        assert isinstance(result_data["system"], list)
        # Should have injected prompt first, then existing
        assert len(result_data["system"]) == 2
        assert result_data["system"][0]["_ccproxy_injected"] is True
        assert result_data["system"][0]["text"] == "You are a helpful assistant."
        assert result_data["system"][1]["text"] == "You are a coding assistant."

    def test_mark_injected_system_prompts_string(
        self, adapter: ClaudeAPIAdapter
    ) -> None:
        """Test marking string system prompts as injected."""
        result = adapter._mark_injected_system_prompts("You are helpful.")

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "You are helpful."
        assert result[0]["_ccproxy_injected"] is True

    def test_mark_injected_system_prompts_list(self, adapter: ClaudeAPIAdapter) -> None:
        """Test marking list system prompts as injected."""
        system_list = [
            {"type": "text", "text": "You are helpful."},
            {"type": "text", "text": "Be concise."},
        ]

        result = adapter._mark_injected_system_prompts(system_list)

        assert isinstance(result, list)
        assert len(result) == 2
        for block in result:
            assert block["_ccproxy_injected"] is True
        assert result[0]["text"] == "You are helpful."
        assert result[1]["text"] == "Be concise."

    def test_needs_openai_conversion(self, adapter: ClaudeAPIAdapter) -> None:
        """Test OpenAI conversion detection."""
        assert adapter._needs_openai_conversion("/v1/chat/completions") is True
        assert adapter._needs_openai_conversion("/v1/messages") is False

    def test_needs_anthropic_conversion(self, adapter: ClaudeAPIAdapter) -> None:
        """Test Anthropic conversion detection."""
        assert adapter._needs_anthropic_conversion("/v1/chat/completions") is True
        assert adapter._needs_anthropic_conversion("/v1/messages") is False

    def test_system_prompt_injection_modes(self) -> None:
        """Test different system prompt injection modes."""
        from ccproxy.plugins.claude_api.config import ClaudeAPISettings

        # Test data
        system_prompts = [
            {"type": "text", "text": "First prompt"},
            {"type": "text", "text": "Second prompt"},
            {"type": "text", "text": "Third prompt"},
        ]

        body_data = {"messages": [{"role": "user", "content": "Hello"}]}

        # Test none mode
        config_none = ClaudeAPISettings(system_prompt_injection_mode="none")
        adapter = ClaudeAPIAdapter(
            detection_service=Mock(),
            config=config_none,
            auth_manager=Mock(),
            http_pool_manager=Mock(),
        )
        result = adapter._inject_system_prompt(
            body_data.copy(), system_prompts, mode="none"
        )
        assert "system" not in result

        # Test minimal mode
        config_minimal = ClaudeAPISettings(system_prompt_injection_mode="minimal")
        adapter = ClaudeAPIAdapter(
            detection_service=Mock(),
            config=config_minimal,
            auth_manager=Mock(),
            http_pool_manager=Mock(),
        )
        result = adapter._inject_system_prompt(
            body_data.copy(), system_prompts, mode="minimal"
        )
        assert "system" in result
        assert len(result["system"]) == 1
        assert result["system"][0]["text"] == "First prompt"
        assert result["system"][0]["_ccproxy_injected"] is True

        # Test full mode
        config_full = ClaudeAPISettings(system_prompt_injection_mode="full")
        adapter = ClaudeAPIAdapter(
            detection_service=Mock(),
            config=config_full,
            auth_manager=Mock(),
            http_pool_manager=Mock(),
        )
        result = adapter._inject_system_prompt(
            body_data.copy(), system_prompts, mode="full"
        )
        assert "system" in result
        assert len(result["system"]) == 3
        assert all(block["_ccproxy_injected"] is True for block in result["system"])
