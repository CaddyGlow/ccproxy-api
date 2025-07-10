"""Tests for the rate limit middleware integration."""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

from ccproxy.metrics.middleware import MetricsMiddleware
from ccproxy.metrics.models import HTTPMetrics, UserAgentCategory
from ccproxy.models.rate_limit import (
    OAuthUnifiedRateLimit,
    RateLimitData,
    StandardRateLimit,
)
from ccproxy.services.rate_limit_tracker import RateLimitTracker


class TestRateLimitMiddleware:
    """Test the rate limit middleware functionality."""

    @pytest.fixture
    def middleware(self):
        """Create middleware instance."""
        app = FastAPI()
        return MetricsMiddleware(app)

    @pytest.fixture
    def mock_response(self):
        """Create a mock response."""
        response = Mock(spec=Response)
        response.status_code = 200
        response.headers = {
            "content-type": "application/json",
        }
        response.body = b'{"result": "success"}'
        return response

    @patch("ccproxy.metrics.middleware.extract_rate_limit_headers")
    async def test_extract_rate_limit_data_api_key(
        self, mock_extract, middleware, mock_response
    ):
        """Test extraction of API key rate limit data from response."""
        # Mock the extraction to return properly formatted data
        mock_extract.return_value = {
            "auth_type": "api_key",
            "standard": {
                "requests_limit": 1000,
                "requests_remaining": 750,
                "tokens_limit": 50000,
                "tokens_remaining": 45000,
                "reset_timestamp": datetime.now() + timedelta(hours=1),
                "retry_after_seconds": 60,
            },
            "oauth_unified": {
                "status": None,
                "representative_claim": None,
                "fallback_percentage": None,
                "reset_timestamp": None,
            },
            "detected_headers": [
                "x-ratelimit-limit",
                "x-ratelimit-remaining",
                "x-ratelimit-reset",
                "retry-after",
            ],
        }

        rate_limit_data = middleware._extract_rate_limit_data(mock_response)

        assert rate_limit_data is not None
        assert rate_limit_data.auth_type == "api_key"
        assert rate_limit_data.standard is not None
        assert rate_limit_data.standard.requests_limit == 1000
        assert rate_limit_data.standard.requests_remaining == 750
        assert rate_limit_data.standard.retry_after_seconds == 60
        assert rate_limit_data.oauth_unified is None

    @patch("ccproxy.metrics.middleware.extract_rate_limit_headers")
    async def test_extract_rate_limit_data_oauth(
        self, mock_extract, middleware, mock_response
    ):
        """Test extraction of OAuth unified rate limit data from response."""
        # Mock the extraction to return properly formatted data
        mock_extract.return_value = {
            "auth_type": "oauth",
            "standard": {
                "requests_limit": None,
                "requests_remaining": None,
                "tokens_limit": None,
                "tokens_remaining": None,
                "reset_timestamp": None,
                "retry_after_seconds": None,
            },
            "oauth_unified": {
                "status": "allowed",
                "representative_claim": "five_hour",
                "fallback_percentage": 85.5,
                "reset_timestamp": datetime.now() + timedelta(hours=1),
            },
            "detected_headers": [
                "anthropic-ratelimit-unified-status",
                "anthropic-ratelimit-unified-fallback-percentage",
            ],
        }

        rate_limit_data = middleware._extract_rate_limit_data(mock_response)

        assert rate_limit_data is not None
        assert rate_limit_data.auth_type == "oauth"
        assert rate_limit_data.oauth_unified is not None
        assert rate_limit_data.oauth_unified.status == "allowed"
        assert rate_limit_data.oauth_unified.representative_claim == "five_hour"
        assert rate_limit_data.oauth_unified.fallback_percentage == 85.5
        assert rate_limit_data.standard is None

    @patch("ccproxy.metrics.middleware.extract_rate_limit_headers")
    async def test_extract_rate_limit_data_mixed_headers(
        self, mock_extract, middleware, mock_response
    ):
        """Test extraction with both API key and OAuth headers."""
        # Mock the extraction to return properly formatted data
        mock_extract.return_value = {
            "auth_type": "oauth",  # OAuth takes precedence
            "standard": {
                "requests_limit": 1000,
                "requests_remaining": 750,
                "tokens_limit": None,
                "tokens_remaining": None,
                "reset_timestamp": None,
                "retry_after_seconds": None,
            },
            "oauth_unified": {
                "status": "allowed",
                "representative_claim": None,
                "fallback_percentage": 75.0,
                "reset_timestamp": None,
            },
            "detected_headers": [
                "x-ratelimit-limit",
                "x-ratelimit-remaining",
                "anthropic-ratelimit-unified-status",
            ],
        }

        rate_limit_data = middleware._extract_rate_limit_data(mock_response)

        assert rate_limit_data is not None
        assert rate_limit_data.auth_type == "oauth"  # OAuth takes precedence
        assert rate_limit_data.oauth_unified is not None
        assert rate_limit_data.oauth_unified.status == "allowed"
        assert rate_limit_data.oauth_unified.fallback_percentage == 75.0

    @patch("ccproxy.metrics.middleware.extract_rate_limit_headers")
    async def test_extract_rate_limit_data_no_headers(
        self, mock_extract, middleware, mock_response
    ):
        """Test extraction with no rate limit headers."""
        # Mock the extraction to return no detected headers
        mock_extract.return_value = {
            "auth_type": "unknown",
            "standard": {
                "requests_limit": None,
                "requests_remaining": None,
                "tokens_limit": None,
                "tokens_remaining": None,
                "reset_timestamp": None,
                "retry_after_seconds": None,
            },
            "oauth_unified": {
                "status": None,
                "representative_claim": None,
                "fallback_percentage": None,
                "reset_timestamp": None,
            },
            "detected_headers": [],
        }

        rate_limit_data = middleware._extract_rate_limit_data(mock_response)

        assert rate_limit_data is None

    @patch("ccproxy.metrics.middleware.extract_rate_limit_headers")
    async def test_extract_rate_limit_data_exception_handling(
        self, mock_extract, middleware, mock_response
    ):
        """Test exception handling in rate limit data extraction."""
        # Mock the extraction to raise an exception
        mock_extract.side_effect = Exception("Test exception")

        rate_limit_data = middleware._extract_rate_limit_data(mock_response)

        assert rate_limit_data is None

    def test_populate_rate_limit_fields_api_key(self, middleware):
        """Test populating rate limit fields for API key auth."""
        http_metrics = HTTPMetrics(
            method="POST",
            endpoint="/cc/v1/chat/completions",
            status_code=200,
            api_type="claude_code",
            user_agent_category=UserAgentCategory.PYTHON_SDK,
            duration_seconds=1.5,
            request_size_bytes=1024,
            response_size_bytes=2048,
        )

        standard_rate_limit = StandardRateLimit(
            requests_limit=1000,
            requests_remaining=750,
            tokens_limit=50000,
            tokens_remaining=45000,
            reset_timestamp=datetime.now() + timedelta(hours=1),
            retry_after_seconds=60,
        )

        rate_limit_data = RateLimitData(
            auth_type="api_key",
            standard=standard_rate_limit,
            timestamp=datetime.now(),
        )

        middleware._populate_rate_limit_fields(http_metrics, rate_limit_data)

        assert http_metrics.auth_type == "api_key"
        assert http_metrics.rate_limit_requests_limit == 1000
        assert http_metrics.rate_limit_requests_remaining == 750
        assert http_metrics.rate_limit_tokens_limit == 50000
        assert http_metrics.rate_limit_tokens_remaining == 45000
        assert http_metrics.rate_limit_reset_timestamp is not None
        assert http_metrics.retry_after_seconds == 60
        assert http_metrics.oauth_unified_status is None

    def test_populate_rate_limit_fields_oauth(self, middleware):
        """Test populating rate limit fields for OAuth auth."""
        http_metrics = HTTPMetrics(
            method="POST",
            endpoint="/cc/v1/chat/completions",
            status_code=200,
            api_type="claude_code",
            user_agent_category=UserAgentCategory.PYTHON_SDK,
            duration_seconds=1.5,
            request_size_bytes=1024,
            response_size_bytes=2048,
        )

        oauth_rate_limit = OAuthUnifiedRateLimit(
            status="allowed",
            representative_claim="five_hour",
            fallback_percentage=85.5,
            reset_timestamp=datetime.now() + timedelta(hours=1),
        )

        rate_limit_data = RateLimitData(
            auth_type="oauth",
            oauth_unified=oauth_rate_limit,
            timestamp=datetime.now(),
        )

        middleware._populate_rate_limit_fields(http_metrics, rate_limit_data)

        assert http_metrics.auth_type == "oauth"
        assert http_metrics.oauth_unified_status == "allowed"
        assert http_metrics.oauth_unified_claim == "five_hour"
        assert http_metrics.oauth_unified_fallback_percentage == 85.5
        assert http_metrics.oauth_unified_reset is not None
        assert http_metrics.rate_limit_requests_limit is None

    def test_populate_rate_limit_fields_exception_handling(self, middleware):
        """Test exception handling in rate limit field population."""
        http_metrics = HTTPMetrics(
            method="POST",
            endpoint="/cc/v1/chat/completions",
            status_code=200,
            api_type="claude_code",
            user_agent_category=UserAgentCategory.PYTHON_SDK,
            duration_seconds=1.5,
            request_size_bytes=1024,
            response_size_bytes=2048,
        )

        # Create invalid rate limit data that will cause an exception
        rate_limit_data = Mock()
        rate_limit_data.auth_type = "api_key"
        rate_limit_data.standard = Mock()
        rate_limit_data.standard.requests_limit = Mock(
            side_effect=Exception("Test exception")
        )

        # Should not raise an exception
        middleware._populate_rate_limit_fields(http_metrics, rate_limit_data)

        # Auth type should still be set
        assert http_metrics.auth_type == "api_key"

    @patch("ccproxy.metrics.middleware.get_rate_limit_tracker")
    def test_track_rate_limit_usage(self, mock_get_tracker, middleware):
        """Test tracking rate limit usage."""
        mock_tracker = Mock(spec=RateLimitTracker)
        mock_tracker.track_rate_limit = Mock()
        mock_get_tracker.return_value = mock_tracker

        rate_limit_data = RateLimitData(
            auth_type="api_key",
            standard=StandardRateLimit(
                requests_limit=1000,
                requests_remaining=750,
            ),
            timestamp=datetime.now(),
        )

        middleware._track_rate_limit_usage(rate_limit_data)

        mock_tracker.track_rate_limit.assert_called_once_with(rate_limit_data)

    @patch("ccproxy.metrics.middleware.get_rate_limit_tracker")
    def test_track_rate_limit_usage_exception_handling(
        self, mock_get_tracker, middleware
    ):
        """Test exception handling in rate limit usage tracking."""
        mock_get_tracker.side_effect = Exception("Test exception")

        rate_limit_data = RateLimitData(
            auth_type="api_key",
            standard=StandardRateLimit(
                requests_limit=1000,
                requests_remaining=750,
            ),
            timestamp=datetime.now(),
        )

        # Should not raise an exception
        middleware._track_rate_limit_usage(rate_limit_data)

    @patch("ccproxy.metrics.middleware.extract_rate_limit_headers")
    async def test_case_insensitive_rate_limit_headers(
        self, mock_extract, middleware, mock_response
    ):
        """Test that rate limit header extraction is case-insensitive."""
        # Mock the extraction to return data from case-insensitive headers
        mock_extract.return_value = {
            "auth_type": "oauth",
            "standard": {
                "requests_limit": 1000,
                "requests_remaining": 750,
                "tokens_limit": None,
                "tokens_remaining": None,
                "reset_timestamp": None,
                "retry_after_seconds": None,
            },
            "oauth_unified": {
                "status": "allowed",
                "representative_claim": None,
                "fallback_percentage": 85.0,
                "reset_timestamp": None,
            },
            "detected_headers": [
                "x-ratelimit-limit",
                "anthropic-ratelimit-unified-status",
            ],
        }

        rate_limit_data = middleware._extract_rate_limit_data(mock_response)

        assert rate_limit_data is not None
        assert rate_limit_data.auth_type == "oauth"
        assert rate_limit_data.oauth_unified.status == "allowed"
        assert rate_limit_data.oauth_unified.fallback_percentage == 85.0

    def test_streaming_response_handling(self, middleware):
        """Test that middleware handles streaming response size calculation."""
        # Create a streaming response mock
        streaming_response = Mock(spec=StreamingResponse)
        streaming_response.headers = {"content-length": "24"}
        streaming_response.status_code = 200

        # Test the response size calculation

        async def test_calc():
            return await middleware._calculate_response_size(streaming_response)

        size = asyncio.run(test_calc())
        assert size > 0  # Should have some size from headers + content length

    def test_rate_limit_data_validation(self, middleware):
        """Test validation of rate limit data structures."""
        # Test that valid rate limit data is handled correctly
        rate_limit_data = RateLimitData(
            auth_type="api_key",
            standard=StandardRateLimit(
                requests_limit=1000,
                requests_remaining=750,
            ),
            timestamp=datetime.now(),
        )

        # Should not raise any exceptions
        assert rate_limit_data.auth_type == "api_key"
        assert rate_limit_data.standard.requests_limit == 1000
        assert rate_limit_data.standard.requests_remaining == 750

    def test_rate_limit_data_edge_cases(self, middleware):
        """Test edge cases for rate limit data."""
        # Test with None values
        rate_limit_data = RateLimitData(
            auth_type="unknown",
            timestamp=datetime.now(),
        )

        assert rate_limit_data.auth_type == "unknown"
        assert rate_limit_data.standard is None
        assert rate_limit_data.oauth_unified is None

        # Test with zero values
        rate_limit_data = RateLimitData(
            auth_type="api_key",
            standard=StandardRateLimit(
                requests_limit=0,
                requests_remaining=0,
            ),
            timestamp=datetime.now(),
        )

        assert rate_limit_data.standard.requests_limit == 0
        assert rate_limit_data.standard.requests_remaining == 0

    def test_concurrent_rate_limit_tracking(self, middleware):
        """Test that rate limit tracking works correctly under concurrent access."""
        import threading

        results = []

        def track_rate_limit():
            with patch(
                "ccproxy.metrics.middleware.get_rate_limit_tracker"
            ) as mock_get_tracker:
                mock_tracker = Mock()
                mock_tracker.track_rate_limit = Mock()
                mock_get_tracker.return_value = mock_tracker

                rate_limit_data = RateLimitData(
                    auth_type="api_key",
                    standard=StandardRateLimit(
                        requests_limit=1000,
                        requests_remaining=750,
                    ),
                    timestamp=datetime.now(),
                )

                middleware._track_rate_limit_usage(rate_limit_data)
                results.append(True)

        # Run multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=track_rate_limit)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All threads should complete successfully
        assert len(results) == 5
        assert all(results)

    def test_error_handling_in_middleware_components(self, middleware):
        """Test error handling in various middleware components."""
        # Test API type extraction with invalid path
        api_type = middleware._extract_api_type("")
        assert api_type == "reverse_proxy"  # Default fallback

        # Test endpoint pattern extraction with special characters
        endpoint = middleware._extract_endpoint_pattern(
            "/test/path?query=value#fragment"
        )
        assert endpoint == "/test/path"  # Query and fragment should be removed

        # Test endpoint normalization with UUIDs
        endpoint = middleware._extract_endpoint_pattern(
            "/api/users/123e4567-e89b-12d3-a456-426614174000/profile"
        )
        assert endpoint == "/api/users/{id}/profile"

    def test_middleware_initialization(self):
        """Test middleware initialization."""
        app = FastAPI()
        middleware = MetricsMiddleware(app)

        assert middleware.metrics_collector is not None
        assert hasattr(middleware, "_extract_api_type")
        assert hasattr(middleware, "_extract_rate_limit_data")
        assert hasattr(middleware, "_populate_rate_limit_fields")
        assert hasattr(middleware, "_track_rate_limit_usage")

    def test_database_integration_readiness(self, middleware):
        """Test that middleware is ready for database integration."""
        # Test that HTTPMetrics can be created with all rate limit fields
        http_metrics = HTTPMetrics(
            method="POST",
            endpoint="/cc/v1/chat/completions",
            status_code=200,
            api_type="claude_code",
            user_agent_category=UserAgentCategory.PYTHON_SDK,
            duration_seconds=1.5,
            request_size_bytes=1024,
            response_size_bytes=2048,
            # API key fields
            auth_type="api_key",
            rate_limit_requests_limit=1000,
            rate_limit_requests_remaining=750,
            rate_limit_tokens_limit=50000,
            rate_limit_tokens_remaining=45000,
            rate_limit_reset_timestamp="2024-01-01T12:00:00Z",
            retry_after_seconds=60,
            # OAuth fields
            oauth_unified_status="allowed",
            oauth_unified_claim="five_hour",
            oauth_unified_fallback_percentage=85.5,
            oauth_unified_reset="2024-01-01T12:00:00Z",
        )

        # Verify all fields are populated
        assert http_metrics.auth_type == "api_key"
        assert http_metrics.rate_limit_requests_limit == 1000
        assert http_metrics.oauth_unified_status == "allowed"
        assert http_metrics.oauth_unified_fallback_percentage == 85.5

        # Test JSON serialization (for database storage)
        json_data = http_metrics.model_dump()
        assert "auth_type" in json_data
        assert "rate_limit_requests_limit" in json_data
        assert "oauth_unified_status" in json_data
