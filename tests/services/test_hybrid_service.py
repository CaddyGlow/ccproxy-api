"""Tests for HybridService."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from ccproxy.services.hybrid_service import HybridService


class TestHybridService:
    """Test cases for HybridService."""

    @pytest.fixture
    def mock_claude_client(self):
        """Create a mock Claude client."""
        mock = AsyncMock()
        mock.validate_health = AsyncMock()
        mock.close = AsyncMock()
        return mock

    @pytest.fixture
    def mock_reverse_proxy_service(self):
        """Create a mock reverse proxy service."""
        mock = AsyncMock()
        mock.proxy_request = AsyncMock()
        return mock

    @pytest.fixture
    def hybrid_service(self, mock_claude_client, mock_reverse_proxy_service):
        """Create a HybridService instance with mocked dependencies."""
        return HybridService(
            claude_client=mock_claude_client,
            reverse_proxy_service=mock_reverse_proxy_service,
            default_proxy_mode="hybrid",
        )

    def test_hybrid_service_initialization(self, hybrid_service):
        """Test that HybridService initializes correctly."""
        assert hybrid_service.default_proxy_mode == "hybrid"
        assert hybrid_service._use_sdk_for_tools is True
        assert hybrid_service._use_proxy_for_simple_requests is True
        assert hybrid_service._sdk_timeout_threshold == 30.0

    def test_parse_request_body_valid_json(self, hybrid_service):
        """Test parsing valid JSON request body."""
        body = b'{"messages": ["hello"], "stream": false}'
        result = hybrid_service._parse_request_body(body)
        
        expected = {"messages": ["hello"], "stream": False}
        assert result == expected

    def test_parse_request_body_invalid_json(self, hybrid_service):
        """Test parsing invalid JSON request body."""
        body = b'{"invalid": json}'
        result = hybrid_service._parse_request_body(body)
        
        assert result == {}

    def test_parse_request_body_empty(self, hybrid_service):
        """Test parsing empty request body."""
        result = hybrid_service._parse_request_body(None)
        assert result == {}

    def test_has_tools_with_tools(self, hybrid_service):
        """Test _has_tools with tools present."""
        request_body = {"tools": [{"name": "test_tool"}]}
        assert hybrid_service._has_tools(request_body) is True

    def test_has_tools_without_tools(self, hybrid_service):
        """Test _has_tools without tools."""
        request_body = {"messages": ["hello"]}
        assert hybrid_service._has_tools(request_body) is False

    def test_has_streaming_true(self, hybrid_service):
        """Test _has_streaming with streaming enabled."""
        request_body = {"stream": True}
        assert hybrid_service._has_streaming(request_body) is True

    def test_has_streaming_false(self, hybrid_service):
        """Test _has_streaming with streaming disabled."""
        request_body = {"stream": False}
        assert hybrid_service._has_streaming(request_body) is False

    def test_is_simple_request_true(self, hybrid_service):
        """Test _is_simple_request with a simple request."""
        request_body = {"messages": ["hello"]}
        assert hybrid_service._is_simple_request(request_body) is True

    def test_is_simple_request_false_with_tools(self, hybrid_service):
        """Test _is_simple_request with tools present."""
        request_body = {"messages": ["hello"], "tools": [{"name": "test"}]}
        assert hybrid_service._is_simple_request(request_body) is False

    @pytest.mark.asyncio
    async def test_should_use_sdk_non_chat_endpoint(self, hybrid_service):
        """Test _should_use_sdk with non-chat endpoint."""
        request_body = {"messages": ["hello"]}
        result = await hybrid_service._should_use_sdk(request_body, "/v1/models")
        assert result is False

    @pytest.mark.asyncio
    async def test_should_use_sdk_with_tools(self, hybrid_service):
        """Test _should_use_sdk with tools present."""
        request_body = {"tools": [{"name": "test"}]}
        result = await hybrid_service._should_use_sdk(request_body, "/v1/messages")
        assert result is True

    @pytest.mark.asyncio
    async def test_health_check(self, hybrid_service, mock_claude_client):
        """Test health check functionality."""
        # Configure mock to succeed
        mock_claude_client.validate_health.return_value = None
        
        result = await hybrid_service.health_check()
        
        assert "hybrid_service" in result
        assert result["hybrid_service"]["status"] == "healthy"
        assert result["hybrid_service"]["sdk"]["status"] == "healthy"
        assert result["hybrid_service"]["proxy"]["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_check_sdk_failure(self, hybrid_service, mock_claude_client):
        """Test health check with SDK failure."""
        # Configure mock to fail
        mock_claude_client.validate_health.side_effect = Exception("SDK error")
        
        result = await hybrid_service.health_check()
        
        assert result["hybrid_service"]["sdk"]["status"] == "unhealthy"
        assert result["hybrid_service"]["sdk"]["error"] == "SDK error"

    @pytest.mark.asyncio
    async def test_close(self, hybrid_service, mock_claude_client):
        """Test closing the service."""
        await hybrid_service.close()
        mock_claude_client.close.assert_called_once()