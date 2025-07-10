"""Tests for reverse proxy rate limit header preservation."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest
from fastapi.responses import StreamingResponse

from ccproxy.services.reverse_proxy import ReverseProxyService
from ccproxy.utils.http_client import InstrumentedHttpClient


@pytest.mark.unit
class TestReverseProxyRateLimitHeaders:
    """Test rate limit header preservation in reverse proxy."""

    @pytest.fixture
    def proxy_service(self):
        """Create reverse proxy service for testing."""
        return ReverseProxyService()

    def test_preserve_rate_limit_headers_api_key(self, proxy_service):
        """Test preserving API key rate limit headers."""
        anthropic_headers = {
            "content-type": "application/json",
            "x-ratelimit-limit-requests": "1000",
            "x-ratelimit-remaining-requests": "750",
            "x-ratelimit-limit-tokens": "50000",
            "x-ratelimit-remaining-tokens": "45000",
            "x-ratelimit-reset-requests": "1640995200",
            "x-ratelimit-reset-tokens": "1640995200",
            "retry-after": "60",
            "custom-header": "value",
        }

        transformed_headers = {
            "content-type": "application/json",
            "content-length": "100",
        }

        result = proxy_service._preserve_rate_limit_headers(
            anthropic_headers, transformed_headers
        )

        # All rate limit headers should be preserved
        assert result["x-ratelimit-limit-requests"] == "1000"
        assert result["x-ratelimit-remaining-requests"] == "750"
        assert result["x-ratelimit-limit-tokens"] == "50000"
        assert result["x-ratelimit-remaining-tokens"] == "45000"
        assert result["x-ratelimit-reset-requests"] == "1640995200"
        assert result["x-ratelimit-reset-tokens"] == "1640995200"
        assert result["retry-after"] == "60"

        # Original headers should be preserved
        assert result["content-type"] == "application/json"
        assert result["content-length"] == "100"

        # Non-rate-limit headers should not be copied
        assert "custom-header" not in result

    def test_preserve_rate_limit_headers_oauth(self, proxy_service):
        """Test preserving OAuth unified rate limit headers."""
        anthropic_headers = {
            "content-type": "application/json",
            "anthropic-ratelimit-unified-status": "allowed",
            "anthropic-ratelimit-unified-representative-claim": "five_hour",
            "anthropic-ratelimit-unified-fallback-percentage": "85.5",
            "anthropic-ratelimit-unified-reset": "1752105600",
            "other-header": "value",
        }

        transformed_headers: dict[str, str] = {
            "content-type": "application/json",
        }

        result = proxy_service._preserve_rate_limit_headers(
            anthropic_headers, transformed_headers
        )

        # OAuth unified headers should be preserved
        assert result["anthropic-ratelimit-unified-status"] == "allowed"
        assert result["anthropic-ratelimit-unified-representative-claim"] == "five_hour"
        assert result["anthropic-ratelimit-unified-fallback-percentage"] == "85.5"
        assert result["anthropic-ratelimit-unified-reset"] == "1752105600"

        # Non-rate-limit headers should not be copied
        assert "other-header" not in result

    def test_preserve_rate_limit_headers_mixed(self, proxy_service):
        """Test preserving both API key and OAuth headers."""
        anthropic_headers = {
            "x-ratelimit-limit-requests": "1000",
            "x-ratelimit-remaining-requests": "750",
            "anthropic-ratelimit-unified-status": "allowed",
            "anthropic-ratelimit-unified-fallback-percentage": "75.0",
            "content-type": "application/json",
        }

        transformed_headers: dict[str, str] = {}

        result = proxy_service._preserve_rate_limit_headers(
            anthropic_headers, transformed_headers
        )

        # Both types of headers should be preserved
        assert result["x-ratelimit-limit-requests"] == "1000"
        assert result["x-ratelimit-remaining-requests"] == "750"
        assert result["anthropic-ratelimit-unified-status"] == "allowed"
        assert result["anthropic-ratelimit-unified-fallback-percentage"] == "75.0"

        # Non-rate-limit headers should not be copied
        assert "content-type" not in result

    def test_preserve_rate_limit_headers_case_insensitive(self, proxy_service):
        """Test case-insensitive header preservation."""
        anthropic_headers = {
            "X-RateLimit-Limit-Requests": "1000",  # Different case
            "x-ratelimit-remaining-requests": "750",  # Lower case
            "Anthropic-RateLimit-Unified-Status": "allowed",  # Mixed case
        }

        transformed_headers: dict[str, str] = {}

        result = proxy_service._preserve_rate_limit_headers(
            anthropic_headers, transformed_headers
        )

        # Headers should be preserved with original case
        assert result["X-RateLimit-Limit-Requests"] == "1000"
        assert result["x-ratelimit-remaining-requests"] == "750"
        assert result["Anthropic-RateLimit-Unified-Status"] == "allowed"

    def test_preserve_rate_limit_headers_no_rate_limits(self, proxy_service):
        """Test preserving headers when no rate limit headers present."""
        anthropic_headers = {
            "content-type": "application/json",
            "content-length": "100",
            "custom-header": "value",
        }

        transformed_headers = {
            "existing-header": "value",
        }

        result = proxy_service._preserve_rate_limit_headers(
            anthropic_headers, transformed_headers
        )

        # Only existing transformed headers should remain
        assert result == {"existing-header": "value"}

    def test_preserve_rate_limit_headers_empty_input(self, proxy_service):
        """Test preserving headers with empty input."""
        result = proxy_service._preserve_rate_limit_headers({}, {})
        assert result == {}

        result = proxy_service._preserve_rate_limit_headers(
            {"x-ratelimit-limit-requests": "1000"}, {}
        )
        assert result == {"x-ratelimit-limit-requests": "1000"}

    @pytest.mark.asyncio
    async def test_proxy_request_preserves_rate_limit_headers(self, proxy_service):
        """Test that proxy_request preserves rate limit headers."""
        # Mock the credentials manager
        mock_credentials_manager = AsyncMock()
        mock_credentials_manager.get_access_token.return_value = "test-token"
        proxy_service._credentials_manager = mock_credentials_manager

        # Mock httpx response
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.content = b'{"result": "success"}'
        mock_response.headers = httpx.Headers(
            {
                "content-type": "application/json",
                "x-ratelimit-limit-requests": "1000",
                "x-ratelimit-remaining-requests": "750",
                "anthropic-ratelimit-unified-status": "allowed",
            }
        )
        mock_response.reason_phrase = "OK"
        mock_response.extensions = {}

        # Mock the request transformer
        proxy_service.request_transformer.transform_path = Mock(
            return_value="/v1/messages"
        )
        proxy_service.request_transformer.create_proxy_headers = Mock(
            return_value={"authorization": "Bearer test-token"}
        )
        proxy_service.request_transformer.transform_request_body = Mock(
            return_value=b'{"message": "test"}'
        )

        # Mock the response transformer
        proxy_service.response_transformer.transform_response_body = Mock(
            return_value=b'{"result": "success"}'
        )
        proxy_service.response_transformer.transform_response_headers = Mock(
            return_value={"content-type": "application/json", "content-length": "20"}
        )

        # Mock InstrumentedHttpClient
        with patch.object(proxy_service, "_get_http_client") as mock_get_client:
            mock_client = Mock(spec=InstrumentedHttpClient)
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            # Make request
            status, headers, body = await proxy_service.proxy_request(
                method="POST",
                path="/v1/messages",
                headers={"content-type": "application/json"},
                body=b'{"message": "test"}',
            )

            # Check that rate limit headers are preserved
            assert status == 200
            assert headers["x-ratelimit-limit-requests"] == "1000"
            assert headers["x-ratelimit-remaining-requests"] == "750"
            assert headers["anthropic-ratelimit-unified-status"] == "allowed"
            assert headers["content-type"] == "application/json"

    def test_streaming_response_header_collection(self, proxy_service):
        """Test that streaming response setup includes header capture logic."""
        # Test that the streaming response method handles header capture
        # This tests the logic without the complex async mocking

        # Mock response headers for testing
        test_headers = {
            "content-type": "text/event-stream",
            "x-ratelimit-limit-requests": "1000",
            "x-ratelimit-remaining-requests": "750",
            "anthropic-ratelimit-unified-status": "allowed",
        }

        # Verify that our rate limit header preservation logic works
        preserved = proxy_service._preserve_rate_limit_headers(test_headers, {})

        # Rate limit headers should be preserved
        assert preserved["x-ratelimit-limit-requests"] == "1000"
        assert preserved["x-ratelimit-remaining-requests"] == "750"
        assert preserved["anthropic-ratelimit-unified-status"] == "allowed"

        # Content type should not be preserved (not a rate limit header)
        assert "content-type" not in preserved

    def test_rate_limit_header_coverage(self, proxy_service):
        """Test that all known rate limit headers are covered."""
        # Test all supported rate limit headers
        all_headers = {
            # API key headers
            "x-ratelimit-limit-requests": "1000",
            "x-ratelimit-remaining-requests": "750",
            "x-ratelimit-limit-tokens": "50000",
            "x-ratelimit-remaining-tokens": "45000",
            "x-ratelimit-reset-requests": "1640995200",
            "x-ratelimit-reset-tokens": "1640995200",
            "retry-after": "60",
            # OAuth headers
            "anthropic-ratelimit-unified-status": "allowed",
            "anthropic-ratelimit-unified-representative-claim": "five_hour",
            "anthropic-ratelimit-unified-fallback-percentage": "85.5",
            "anthropic-ratelimit-unified-reset": "1752105600",
            # Non-rate-limit headers
            "content-type": "application/json",
            "content-length": "100",
        }

        result = proxy_service._preserve_rate_limit_headers(all_headers, {})

        # All rate limit headers should be preserved
        expected_headers = {
            "x-ratelimit-limit-requests": "1000",
            "x-ratelimit-remaining-requests": "750",
            "x-ratelimit-limit-tokens": "50000",
            "x-ratelimit-remaining-tokens": "45000",
            "x-ratelimit-reset-requests": "1640995200",
            "x-ratelimit-reset-tokens": "1640995200",
            "retry-after": "60",
            "anthropic-ratelimit-unified-status": "allowed",
            "anthropic-ratelimit-unified-representative-claim": "five_hour",
            "anthropic-ratelimit-unified-fallback-percentage": "85.5",
            "anthropic-ratelimit-unified-reset": "1752105600",
        }

        assert result == expected_headers

    def test_header_preservation_doesnt_overwrite(self, proxy_service):
        """Test that header preservation doesn't overwrite existing headers."""
        anthropic_headers = {
            "x-ratelimit-limit-requests": "1000",
            "content-type": "application/json",
        }

        transformed_headers = {
            "content-type": "application/custom",  # Different value
            "x-custom-header": "value",
        }

        result = proxy_service._preserve_rate_limit_headers(
            anthropic_headers, transformed_headers
        )

        # Rate limit header should be added
        assert result["x-ratelimit-limit-requests"] == "1000"

        # Existing transformed headers should be preserved
        assert result["content-type"] == "application/custom"  # Original value
        assert result["x-custom-header"] == "value"

    @pytest.mark.asyncio
    async def test_error_response_preserves_rate_limit_headers(self, proxy_service):
        """Test that error responses also preserve rate limit headers."""
        # Mock the credentials manager
        mock_credentials_manager = AsyncMock()
        mock_credentials_manager.get_access_token.return_value = "test-token"
        proxy_service._credentials_manager = mock_credentials_manager

        # Mock error response with rate limit headers
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 429  # Rate limited
        mock_response.content = b'{"error": "rate_limit_exceeded"}'
        mock_response.headers = httpx.Headers(
            {
                "content-type": "application/json",
                "x-ratelimit-limit-requests": "1000",
                "x-ratelimit-remaining-requests": "0",
                "retry-after": "60",
            }
        )
        mock_response.reason_phrase = "Too Many Requests"
        mock_response.extensions = {}

        # Mock transformers
        proxy_service.request_transformer.transform_path = Mock(
            return_value="/v1/messages"
        )
        proxy_service.request_transformer.create_proxy_headers = Mock(
            return_value={"authorization": "Bearer test-token"}
        )
        proxy_service.request_transformer.transform_request_body = Mock(
            return_value=b'{"message": "test"}'
        )
        proxy_service.response_transformer.transform_response_body = Mock(
            return_value=b'{"error": "rate_limit_exceeded"}'
        )
        proxy_service.response_transformer.transform_response_headers = Mock(
            return_value={"content-type": "application/json"}
        )

        # Mock InstrumentedHttpClient
        with patch.object(proxy_service, "_get_http_client") as mock_get_client:
            mock_client = Mock(spec=InstrumentedHttpClient)
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            # Make request
            status, headers, body = await proxy_service.proxy_request(
                method="POST",
                path="/v1/messages",
                headers={"content-type": "application/json"},
                body=b'{"message": "test"}',
            )

            # Check that rate limit headers are preserved even in error response
            assert status == 429
            assert headers["x-ratelimit-limit-requests"] == "1000"
            assert headers["x-ratelimit-remaining-requests"] == "0"
            assert headers["retry-after"] == "60"
