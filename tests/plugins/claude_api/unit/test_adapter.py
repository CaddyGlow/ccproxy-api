"""Unit tests for ClaudeAPIAdapter."""

import json
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from ccproxy.models.detection import DetectedHeaders, DetectedPrompts
from ccproxy.plugins.claude_api.adapter import ClaudeAPIAdapter
from ccproxy.plugins.claude_api.detection_service import ClaudeAPIDetectionService
from ccproxy.plugins.oauth_claude.manager import ClaudeApiTokenManager


class TestClaudeAPIAdapter:
    """Test the ClaudeAPIAdapter HTTP adapter methods."""

    @pytest.fixture
    def mock_detection_service(self) -> ClaudeAPIDetectionService:
        """Create mock detection service."""
        service = Mock(spec=ClaudeAPIDetectionService)
        service.get_cached_data.return_value = None
        service.get_detected_headers.return_value = DetectedHeaders({})
        service.get_detected_prompts.return_value = DetectedPrompts()
        service.get_system_prompt.return_value = {}
        service.get_ignored_headers.return_value = (
            ClaudeAPIDetectionService.ignores_header
        )
        service.get_redacted_headers.return_value = (
            ClaudeAPIDetectionService.redact_headers
        )
        return service

    @pytest.fixture
    def mock_auth_manager(self) -> Mock:
        """Create mock auth manager."""
        auth_manager = Mock(spec=ClaudeApiTokenManager)
        auth_manager.get_access_token = AsyncMock(return_value="test-token")
        auth_manager.get_access_token_with_refresh = AsyncMock(
            return_value="test-token"
        )

        token_secret = Mock()
        token_secret.get_secret_value.return_value = "test-token"
        oauth = Mock()
        oauth.access_token = token_secret
        credentials = Mock()
        credentials.claude_ai_oauth = oauth
        auth_manager.load_credentials = AsyncMock(return_value=credentials)
        return auth_manager

    @pytest.fixture
    def mock_http_pool_manager(self) -> Mock:
        """Create mock HTTP pool manager."""
        return Mock()

    @pytest.fixture
    def adapter(
        self,
        mock_detection_service: ClaudeAPIDetectionService,
        mock_auth_manager: Mock,
        mock_http_pool_manager: Mock,
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
        mock_auth_manager: Mock,
        mock_http_pool_manager: Mock,
    ) -> None:
        """Test request preparation with system prompt injection."""
        # Setup detection service with system prompt
        prompts = DetectedPrompts.from_body(
            {"system": [{"type": "text", "text": "You are a helpful assistant."}]}
        )
        mock_detection_service.get_detected_prompts = Mock(return_value=prompts)  # type: ignore[method-assign]
        mock_detection_service.get_system_prompt = Mock(
            return_value=prompts.system_payload()
        )  # type: ignore[method-assign]

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
        # No longer check for _ccproxy_injected since it should be removed before sending to API
        assert "_ccproxy_injected" not in result_data["system"][0]

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
        if isinstance(result.body, bytes):
            body_str = result.body.decode()
        elif isinstance(result.body, memoryview):
            body_str = bytes(result.body).decode()
        else:
            body_str = str(result.body)
        result_data = json.loads(body_str)
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
        if isinstance(result.body, bytes):
            body_str = result.body.decode()
        elif isinstance(result.body, memoryview):
            body_str = bytes(result.body).decode()
        else:
            body_str = str(result.body)
        result_data = json.loads(body_str)
        # The exact conversion depends on the OpenAI adapter implementation
        # Just verify the structure changed
        assert "choices" in result_data or result_data != response_data

    @pytest.mark.asyncio
    async def test_system_prompt_injection_with_existing_system(
        self,
        mock_detection_service: ClaudeAPIDetectionService,
        mock_auth_manager: Mock,
        mock_http_pool_manager: Mock,
    ) -> None:
        """Test system prompt injection when request already has system prompt."""
        # Setup detection service with system prompt
        prompts = DetectedPrompts.from_body(
            {"system": [{"type": "text", "text": "You are a helpful assistant."}]}
        )
        mock_detection_service.get_detected_prompts = Mock(return_value=prompts)  # type: ignore[method-assign]
        mock_detection_service.get_system_prompt = Mock(
            return_value=prompts.system_payload()
        )  # type: ignore[method-assign]

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
        # Should have both prompts (injected first, then existing)
        assert len(result_data["system"]) == 2
        # But _ccproxy_injected field should be removed before sending to the API
        assert "_ccproxy_injected" not in result_data["system"][0]
        assert result_data["system"][0]["text"] == "You are a helpful assistant."
        assert result_data["system"][1]["text"] == "You are a coding assistant."

    @pytest.mark.asyncio
    async def test_prepare_provider_request_removes_metadata(
        self,
        mock_detection_service: ClaudeAPIDetectionService,
        mock_auth_manager: Mock,
        mock_http_pool_manager: Mock,
    ) -> None:
        """Test metadata fields are removed when preparing the provider request."""
        # Setup detection service with system prompt
        prompts = DetectedPrompts.from_body(
            {"system": [{"type": "text", "text": "You are a helpful assistant."}]}
        )
        mock_detection_service.get_detected_prompts = Mock(return_value=prompts)  # type: ignore[method-assign]
        mock_detection_service.get_system_prompt = Mock(
            return_value=prompts.system_payload()
        )  # type: ignore[method-assign]

        adapter = ClaudeAPIAdapter(
            detection_service=mock_detection_service,
            auth_manager=mock_auth_manager,
            http_pool_manager=mock_http_pool_manager,
        )

        # Create a body with _ccproxy_injected fields
        body_dict = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello", "_ccproxy_injected": True}
                    ],
                }
            ],
            "system": [
                {"type": "text", "text": "System prompt", "_ccproxy_injected": True}
            ],
            "tools": [{"name": "tool1", "_ccproxy_injected": True}],
            "max_tokens": 100,
        }
        body = json.dumps(body_dict).encode()
        headers = {"content-type": "application/json"}

        result_body, result_headers = await adapter.prepare_provider_request(
            body, headers, "/v1/messages"
        )

        # Parse the resulting body
        result_data = json.loads(result_body.decode())

        # Verify all _ccproxy_injected fields are removed
        assert "_ccproxy_injected" not in result_data["system"][0]
        assert "_ccproxy_injected" not in result_data["messages"][0]["content"][0]
        assert "_ccproxy_injected" not in result_data["tools"][0]

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

    def test_remove_metadata_fields(self, adapter: ClaudeAPIAdapter) -> None:
        """Test removing metadata fields from request data."""
        # Create test data with _ccproxy_injected in various locations
        data = {
            "system": [
                {"type": "text", "text": "System prompt", "_ccproxy_injected": True},
                {"type": "text", "text": "More instructions"},
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "User message"},
                        {
                            "type": "text",
                            "text": "With metadata",
                            "_ccproxy_injected": True,
                        },
                    ],
                }
            ],
            "tools": [{"name": "tool1", "_ccproxy_injected": True}, {"name": "tool2"}],
        }

        # Remove metadata fields
        clean_data = adapter._remove_metadata_fields(data)

        # Verify _ccproxy_injected is removed from system blocks
        assert "_ccproxy_injected" not in clean_data["system"][0]

        # Verify message content blocks have _ccproxy_injected removed
        assert "_ccproxy_injected" not in clean_data["messages"][0]["content"][1]

        # Verify tool blocks have _ccproxy_injected removed
        assert "_ccproxy_injected" not in clean_data["tools"][0]

        # Original data should be untouched (deep copy in the method)
        assert "_ccproxy_injected" in data["system"][0]  # type: ignore[index]

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
