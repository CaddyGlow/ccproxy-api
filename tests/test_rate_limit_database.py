"""Tests for the database schema changes for rate limiting."""

from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from ccproxy.metrics.models import HTTPMetrics, UserAgentCategory
from ccproxy.metrics.storage import MetricsStorage
from ccproxy.models.rate_limit import (
    OAuthUnifiedRateLimit,
    RateLimitData,
    StandardRateLimit,
)


class TestHTTPMetricsRateLimitFields:
    """Test the HTTPMetrics model with rate limit fields."""

    def test_create_http_metrics_with_api_key_rate_limits(self):
        """Test creating HTTPMetrics with API key rate limit fields."""
        metrics = HTTPMetrics(
            method="POST",
            endpoint="/cc/v1/chat/completions",
            status_code=200,
            api_type="claude_code",
            user_agent_category=UserAgentCategory.PYTHON_SDK,
            duration_seconds=1.5,
            request_size_bytes=1024,
            response_size_bytes=2048,
            auth_type="api_key",
            rate_limit_requests_limit=1000,
            rate_limit_requests_remaining=750,
            rate_limit_tokens_limit=50000,
            rate_limit_tokens_remaining=45000,
            rate_limit_reset_timestamp="2024-01-01T12:00:00Z",
            retry_after_seconds=60,
        )

        assert metrics.auth_type == "api_key"
        assert metrics.rate_limit_requests_limit == 1000
        assert metrics.rate_limit_requests_remaining == 750
        assert metrics.rate_limit_tokens_limit == 50000
        assert metrics.rate_limit_tokens_remaining == 45000
        assert metrics.rate_limit_reset_timestamp == "2024-01-01T12:00:00Z"
        assert metrics.retry_after_seconds == 60

        # OAuth fields should be None
        assert metrics.oauth_unified_status is None
        assert metrics.oauth_unified_claim is None
        assert metrics.oauth_unified_fallback_percentage is None
        assert metrics.oauth_unified_reset is None

    def test_create_http_metrics_with_oauth_rate_limits(self):
        """Test creating HTTPMetrics with OAuth unified rate limit fields."""
        metrics = HTTPMetrics(
            method="POST",
            endpoint="/cc/v1/chat/completions",
            status_code=200,
            api_type="claude_code",
            user_agent_category=UserAgentCategory.PYTHON_SDK,
            duration_seconds=1.5,
            request_size_bytes=1024,
            response_size_bytes=2048,
            auth_type="oauth",
            oauth_unified_status="allowed",
            oauth_unified_claim="five_hour",
            oauth_unified_fallback_percentage=85.5,
            oauth_unified_reset="2024-01-01T12:00:00Z",
        )

        assert metrics.auth_type == "oauth"
        assert metrics.oauth_unified_status == "allowed"
        assert metrics.oauth_unified_claim == "five_hour"
        assert metrics.oauth_unified_fallback_percentage == 85.5
        assert metrics.oauth_unified_reset == "2024-01-01T12:00:00Z"

        # API key fields should be None
        assert metrics.rate_limit_requests_limit is None
        assert metrics.rate_limit_requests_remaining is None
        assert metrics.rate_limit_tokens_limit is None
        assert metrics.rate_limit_tokens_remaining is None
        assert metrics.rate_limit_reset_timestamp is None
        assert metrics.retry_after_seconds is None

    def test_create_http_metrics_with_both_rate_limit_types(self):
        """Test creating HTTPMetrics with both API key and OAuth rate limit fields."""
        metrics = HTTPMetrics(
            method="POST",
            endpoint="/cc/v1/chat/completions",
            status_code=200,
            api_type="claude_code",
            user_agent_category=UserAgentCategory.PYTHON_SDK,
            duration_seconds=1.5,
            request_size_bytes=1024,
            response_size_bytes=2048,
            auth_type="oauth",
            # API key fields
            rate_limit_requests_limit=1000,
            rate_limit_requests_remaining=750,
            rate_limit_tokens_limit=50000,
            rate_limit_tokens_remaining=45000,
            # OAuth fields
            oauth_unified_status="allowed",
            oauth_unified_claim="five_hour",
            oauth_unified_fallback_percentage=85.5,
        )

        # Both types should be present
        assert metrics.auth_type == "oauth"
        assert metrics.rate_limit_requests_limit == 1000
        assert metrics.rate_limit_requests_remaining == 750
        assert metrics.oauth_unified_status == "allowed"
        assert metrics.oauth_unified_fallback_percentage == 85.5

    def test_create_http_metrics_without_rate_limits(self):
        """Test creating HTTPMetrics without rate limit fields."""
        metrics = HTTPMetrics(
            method="POST",
            endpoint="/cc/v1/chat/completions",
            status_code=200,
            api_type="claude_code",
            user_agent_category=UserAgentCategory.PYTHON_SDK,
            duration_seconds=1.5,
            request_size_bytes=1024,
            response_size_bytes=2048,
        )

        # All rate limit fields should be None
        assert metrics.auth_type is None
        assert metrics.rate_limit_requests_limit is None
        assert metrics.rate_limit_requests_remaining is None
        assert metrics.rate_limit_tokens_limit is None
        assert metrics.rate_limit_tokens_remaining is None
        assert metrics.rate_limit_reset_timestamp is None
        assert metrics.retry_after_seconds is None
        assert metrics.oauth_unified_status is None
        assert metrics.oauth_unified_claim is None
        assert metrics.oauth_unified_fallback_percentage is None
        assert metrics.oauth_unified_reset is None

    def test_http_metrics_field_validation(self):
        """Test field validation for rate limit fields."""
        base_metrics = {
            "method": "POST",
            "endpoint": "/cc/v1/chat/completions",
            "status_code": 200,
            "api_type": "claude_code",
            "user_agent_category": UserAgentCategory.PYTHON_SDK,
            "duration_seconds": 1.5,
            "request_size_bytes": 1024,
            "response_size_bytes": 2048,
        }

        # Test that negative values are actually allowed for rate limit fields
        # (rate limits can have negative values in some edge cases)
        metrics = HTTPMetrics(
            **base_metrics,
            rate_limit_requests_limit=-1,
            rate_limit_requests_remaining=-1,
            retry_after_seconds=-1,
            oauth_unified_fallback_percentage=-1.0,
        )
        assert metrics.rate_limit_requests_limit == -1
        assert metrics.rate_limit_requests_remaining == -1
        assert metrics.retry_after_seconds == -1
        assert metrics.oauth_unified_fallback_percentage == -1.0

        # Test that invalid user agent category raises ValidationError
        invalid_metrics = base_metrics.copy()
        invalid_metrics["user_agent_category"] = "invalid_category"
        with pytest.raises(ValidationError):
            HTTPMetrics(**invalid_metrics)

        # Test that negative duration raises ValidationError (has ge=0 constraint)
        invalid_duration_metrics = base_metrics.copy()
        invalid_duration_metrics["duration_seconds"] = -1.0
        with pytest.raises(ValidationError):
            HTTPMetrics(**invalid_duration_metrics)

        # Test percentage above 100 (should be allowed)
        metrics = HTTPMetrics(
            **base_metrics,
            oauth_unified_fallback_percentage=105.0,
        )
        assert metrics.oauth_unified_fallback_percentage == 105.0

    def test_http_metrics_serialization(self):
        """Test serialization of HTTPMetrics with rate limit fields."""
        metrics = HTTPMetrics(
            method="POST",
            endpoint="/cc/v1/chat/completions",
            status_code=200,
            api_type="claude_code",
            user_agent_category=UserAgentCategory.PYTHON_SDK,
            duration_seconds=1.5,
            request_size_bytes=1024,
            response_size_bytes=2048,
            auth_type="api_key",
            rate_limit_requests_limit=1000,
            rate_limit_requests_remaining=750,
            oauth_unified_status="allowed",
            oauth_unified_fallback_percentage=85.5,
        )

        # Test JSON serialization
        json_data = metrics.model_dump()
        assert json_data["auth_type"] == "api_key"
        assert json_data["rate_limit_requests_limit"] == 1000
        assert json_data["rate_limit_requests_remaining"] == 750
        assert json_data["oauth_unified_status"] == "allowed"
        assert json_data["oauth_unified_fallback_percentage"] == 85.5

        # Test deserialization
        new_metrics = HTTPMetrics.model_validate(json_data)
        assert new_metrics.auth_type == "api_key"
        assert new_metrics.rate_limit_requests_limit == 1000
        assert new_metrics.oauth_unified_status == "allowed"

    def test_http_metrics_extra_fields_forbidden(self):
        """Test that extra fields are forbidden in HTTPMetrics."""
        base_metrics = {
            "method": "POST",
            "endpoint": "/cc/v1/chat/completions",
            "status_code": 200,
            "api_type": "claude_code",
            "user_agent_category": UserAgentCategory.PYTHON_SDK,
            "duration_seconds": 1.5,
            "request_size_bytes": 1024,
            "response_size_bytes": 2048,
        }

        # Test that extra fields are rejected
        with pytest.raises(ValidationError):
            HTTPMetrics(
                **base_metrics,
                extra_field="not_allowed",
            )

    def test_http_metrics_boundary_values(self):
        """Test HTTPMetrics with boundary values for rate limit fields."""
        base_metrics = {
            "method": "POST",
            "endpoint": "/cc/v1/chat/completions",
            "status_code": 200,
            "api_type": "claude_code",
            "user_agent_category": UserAgentCategory.PYTHON_SDK,
            "duration_seconds": 1.5,
            "request_size_bytes": 1024,
            "response_size_bytes": 2048,
        }

        # Test zero values
        metrics = HTTPMetrics(
            **base_metrics,
            rate_limit_requests_limit=0,
            rate_limit_requests_remaining=0,
            rate_limit_tokens_limit=0,
            rate_limit_tokens_remaining=0,
            retry_after_seconds=0,
            oauth_unified_fallback_percentage=0.0,
        )

        assert metrics.rate_limit_requests_limit == 0
        assert metrics.rate_limit_requests_remaining == 0
        assert metrics.oauth_unified_fallback_percentage == 0.0

        # Test very large values
        metrics = HTTPMetrics(
            **base_metrics,
            rate_limit_requests_limit=999999999,
            rate_limit_tokens_limit=999999999,
            oauth_unified_fallback_percentage=999.999,
        )

        assert metrics.rate_limit_requests_limit == 999999999
        assert metrics.rate_limit_tokens_limit == 999999999
        assert metrics.oauth_unified_fallback_percentage == 999.999


class TestMetricsStorageRateLimitFields:
    """Test the MetricsStorage with rate limit fields."""

    @pytest.fixture
    def mock_storage(self):
        """Create a mock metrics storage."""
        storage = Mock(spec=MetricsStorage)
        storage.store_request_log = Mock()
        return storage

    def test_store_request_log_with_api_key_rate_limits(self, mock_storage):
        """Test storing request log with API key rate limit data."""
        metrics = HTTPMetrics(
            method="POST",
            endpoint="/cc/v1/chat/completions",
            status_code=200,
            api_type="claude_code",
            user_agent_category=UserAgentCategory.PYTHON_SDK,
            duration_seconds=1.5,
            request_size_bytes=1024,
            response_size_bytes=2048,
            auth_type="api_key",
            rate_limit_requests_limit=1000,
            rate_limit_requests_remaining=750,
            rate_limit_tokens_limit=50000,
            rate_limit_tokens_remaining=45000,
            rate_limit_reset_timestamp="2024-01-01T12:00:00Z",
            retry_after_seconds=60,
        )

        mock_storage.store_request_log(metrics)

        mock_storage.store_request_log.assert_called_once_with(metrics)
        stored_metrics = mock_storage.store_request_log.call_args[0][0]
        assert stored_metrics.auth_type == "api_key"
        assert stored_metrics.rate_limit_requests_limit == 1000
        assert stored_metrics.rate_limit_requests_remaining == 750

    def test_store_request_log_with_oauth_rate_limits(self, mock_storage):
        """Test storing request log with OAuth rate limit data."""
        metrics = HTTPMetrics(
            method="POST",
            endpoint="/cc/v1/chat/completions",
            status_code=200,
            api_type="claude_code",
            user_agent_category=UserAgentCategory.PYTHON_SDK,
            duration_seconds=1.5,
            request_size_bytes=1024,
            response_size_bytes=2048,
            auth_type="oauth",
            oauth_unified_status="allowed",
            oauth_unified_claim="five_hour",
            oauth_unified_fallback_percentage=85.5,
            oauth_unified_reset="2024-01-01T12:00:00Z",
        )

        mock_storage.store_request_log(metrics)

        mock_storage.store_request_log.assert_called_once_with(metrics)
        stored_metrics = mock_storage.store_request_log.call_args[0][0]
        assert stored_metrics.auth_type == "oauth"
        assert stored_metrics.oauth_unified_status == "allowed"
        assert stored_metrics.oauth_unified_fallback_percentage == 85.5

    def test_store_request_log_without_rate_limits(self, mock_storage):
        """Test storing request log without rate limit data."""
        metrics = HTTPMetrics(
            method="POST",
            endpoint="/cc/v1/chat/completions",
            status_code=200,
            api_type="claude_code",
            user_agent_category=UserAgentCategory.PYTHON_SDK,
            duration_seconds=1.5,
            request_size_bytes=1024,
            response_size_bytes=2048,
        )

        mock_storage.store_request_log(metrics)

        mock_storage.store_request_log.assert_called_once_with(metrics)
        stored_metrics = mock_storage.store_request_log.call_args[0][0]
        assert stored_metrics.auth_type is None
        assert stored_metrics.rate_limit_requests_limit is None
        assert stored_metrics.oauth_unified_status is None

    def test_store_request_log_with_mixed_rate_limits(self, mock_storage):
        """Test storing request log with mixed rate limit data."""
        metrics = HTTPMetrics(
            method="POST",
            endpoint="/cc/v1/chat/completions",
            status_code=200,
            api_type="claude_code",
            user_agent_category=UserAgentCategory.PYTHON_SDK,
            duration_seconds=1.5,
            request_size_bytes=1024,
            response_size_bytes=2048,
            auth_type="oauth",
            rate_limit_requests_limit=1000,
            rate_limit_requests_remaining=750,
            oauth_unified_status="allowed",
            oauth_unified_fallback_percentage=85.5,
        )

        mock_storage.store_request_log(metrics)

        mock_storage.store_request_log.assert_called_once_with(metrics)
        stored_metrics = mock_storage.store_request_log.call_args[0][0]
        assert stored_metrics.auth_type == "oauth"
        assert stored_metrics.rate_limit_requests_limit == 1000
        assert stored_metrics.oauth_unified_status == "allowed"


class TestRateLimitDataValidation:
    """Test validation of rate limit data structures."""

    def test_valid_standard_rate_limit(self):
        """Test creating valid StandardRateLimit objects."""
        reset_time = datetime.now() + timedelta(hours=1)

        standard = StandardRateLimit(
            requests_limit=1000,
            requests_remaining=750,
            tokens_limit=50000,
            tokens_remaining=45000,
            reset_timestamp=reset_time,
            retry_after_seconds=60,
        )

        assert standard.requests_limit == 1000
        assert standard.requests_remaining == 750
        assert standard.tokens_limit == 50000
        assert standard.tokens_remaining == 45000
        assert standard.reset_timestamp == reset_time
        assert standard.retry_after_seconds == 60

    def test_valid_oauth_unified_rate_limit(self):
        """Test creating valid OAuthUnifiedRateLimit objects."""
        reset_time = datetime.now() + timedelta(hours=1)

        oauth = OAuthUnifiedRateLimit(
            status="allowed",
            representative_claim="five_hour",
            fallback_percentage=85.5,
            reset_timestamp=reset_time,
        )

        assert oauth.status == "allowed"
        assert oauth.representative_claim == "five_hour"
        assert oauth.fallback_percentage == 85.5
        assert oauth.reset_timestamp == reset_time

    def test_valid_rate_limit_data_with_standard(self):
        """Test creating valid RateLimitData with standard rate limit."""
        reset_time = datetime.now() + timedelta(hours=1)
        timestamp = datetime.now()

        standard = StandardRateLimit(
            requests_limit=1000,
            requests_remaining=750,
            tokens_limit=50000,
            tokens_remaining=45000,
            reset_timestamp=reset_time,
        )

        rate_limit_data = RateLimitData(
            auth_type="api_key",
            standard=standard,
            timestamp=timestamp,
        )

        assert rate_limit_data.auth_type == "api_key"
        assert rate_limit_data.standard == standard
        assert rate_limit_data.oauth_unified is None
        assert rate_limit_data.timestamp == timestamp

    def test_valid_rate_limit_data_with_oauth(self):
        """Test creating valid RateLimitData with OAuth rate limit."""
        reset_time = datetime.now() + timedelta(hours=1)
        timestamp = datetime.now()

        oauth = OAuthUnifiedRateLimit(
            status="allowed",
            representative_claim="five_hour",
            fallback_percentage=85.5,
            reset_timestamp=reset_time,
        )

        rate_limit_data = RateLimitData(
            auth_type="oauth",
            oauth_unified=oauth,
            timestamp=timestamp,
        )

        assert rate_limit_data.auth_type == "oauth"
        assert rate_limit_data.oauth_unified == oauth
        assert rate_limit_data.standard is None
        assert rate_limit_data.timestamp == timestamp

    def test_rate_limit_data_with_both_types(self):
        """Test creating RateLimitData with both standard and OAuth rate limits."""
        reset_time = datetime.now() + timedelta(hours=1)
        timestamp = datetime.now()

        standard = StandardRateLimit(
            requests_limit=1000,
            requests_remaining=750,
        )

        oauth = OAuthUnifiedRateLimit(
            status="allowed",
            fallback_percentage=85.5,
        )

        rate_limit_data = RateLimitData(
            auth_type="oauth",
            standard=standard,
            oauth_unified=oauth,
            timestamp=timestamp,
        )

        assert rate_limit_data.auth_type == "oauth"
        assert rate_limit_data.standard == standard
        assert rate_limit_data.oauth_unified == oauth
        assert rate_limit_data.timestamp == timestamp

    def test_rate_limit_data_without_rate_limits(self):
        """Test creating RateLimitData without rate limit objects."""
        timestamp = datetime.now()

        rate_limit_data = RateLimitData(
            auth_type="unknown",
            timestamp=timestamp,
        )

        assert rate_limit_data.auth_type == "unknown"
        assert rate_limit_data.standard is None
        assert rate_limit_data.oauth_unified is None
        assert rate_limit_data.timestamp == timestamp

    def test_rate_limit_data_serialization(self):
        """Test serialization of RateLimitData objects."""
        reset_time = datetime.now() + timedelta(hours=1)
        timestamp = datetime.now()

        standard = StandardRateLimit(
            requests_limit=1000,
            requests_remaining=750,
            reset_timestamp=reset_time,
        )

        rate_limit_data = RateLimitData(
            auth_type="api_key",
            standard=standard,
            timestamp=timestamp,
        )

        # Test JSON serialization
        json_data = rate_limit_data.model_dump()
        assert json_data["auth_type"] == "api_key"
        assert json_data["standard"]["requests_limit"] == 1000
        assert json_data["oauth_unified"] is None

        # Test deserialization
        new_rate_limit_data = RateLimitData.model_validate(json_data)
        assert new_rate_limit_data.auth_type == "api_key"
        assert new_rate_limit_data.standard.requests_limit == 1000
        assert new_rate_limit_data.oauth_unified is None

    def test_rate_limit_data_edge_cases(self):
        """Test RateLimitData with edge cases."""
        timestamp = datetime.now()

        # Test with zero values
        standard = StandardRateLimit(
            requests_limit=0,
            requests_remaining=0,
            tokens_limit=0,
            tokens_remaining=0,
        )

        rate_limit_data = RateLimitData(
            auth_type="api_key",
            standard=standard,
            timestamp=timestamp,
        )

        assert rate_limit_data.standard.requests_limit == 0
        assert rate_limit_data.standard.requests_remaining == 0

        # Test with very large values
        standard = StandardRateLimit(
            requests_limit=999999999,
            requests_remaining=999999999,
        )

        rate_limit_data = RateLimitData(
            auth_type="api_key",
            standard=standard,
            timestamp=timestamp,
        )

        assert rate_limit_data.standard.requests_limit == 999999999
        assert rate_limit_data.standard.requests_remaining == 999999999

    def test_rate_limit_data_timestamp_validation(self):
        """Test timestamp validation in RateLimitData."""
        # Test with current timestamp
        timestamp = datetime.now()
        rate_limit_data = RateLimitData(
            auth_type="api_key",
            timestamp=timestamp,
        )
        assert rate_limit_data.timestamp == timestamp

        # Test with past timestamp
        past_timestamp = datetime.now() - timedelta(hours=1)
        rate_limit_data = RateLimitData(
            auth_type="api_key",
            timestamp=past_timestamp,
        )
        assert rate_limit_data.timestamp == past_timestamp

        # Test with future timestamp
        future_timestamp = datetime.now() + timedelta(hours=1)
        rate_limit_data = RateLimitData(
            auth_type="api_key",
            timestamp=future_timestamp,
        )
        assert rate_limit_data.timestamp == future_timestamp

    def test_rate_limit_data_optional_fields(self):
        """Test RateLimitData with optional fields."""
        timestamp = datetime.now()

        # Test with minimal StandardRateLimit
        standard = StandardRateLimit()
        rate_limit_data = RateLimitData(
            auth_type="api_key",
            standard=standard,
            timestamp=timestamp,
        )

        assert rate_limit_data.standard.requests_limit is None
        assert rate_limit_data.standard.requests_remaining is None
        assert rate_limit_data.standard.tokens_limit is None
        assert rate_limit_data.standard.tokens_remaining is None
        assert rate_limit_data.standard.reset_timestamp is None
        assert rate_limit_data.standard.retry_after_seconds is None

        # Test with minimal OAuthUnifiedRateLimit
        oauth = OAuthUnifiedRateLimit()
        rate_limit_data = RateLimitData(
            auth_type="oauth",
            oauth_unified=oauth,
            timestamp=timestamp,
        )

        assert rate_limit_data.oauth_unified.status is None
        assert rate_limit_data.oauth_unified.representative_claim is None
        assert rate_limit_data.oauth_unified.fallback_percentage is None
        assert rate_limit_data.oauth_unified.reset_timestamp is None
