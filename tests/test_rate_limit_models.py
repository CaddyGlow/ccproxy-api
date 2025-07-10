"""Tests for rate limit Pydantic models."""

from datetime import UTC, datetime, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from ccproxy.models.rate_limit import (
    OAuthUnifiedRateLimit,
    RateLimitData,
    RateLimitStatus,
    StandardRateLimit,
)


@pytest.mark.unit
class TestStandardRateLimit:
    """Test StandardRateLimit model."""

    def test_valid_standard_rate_limit(self):
        """Test creating a valid StandardRateLimit."""
        reset_time = datetime.now(UTC)
        rate_limit = StandardRateLimit(
            requests_limit=1000,
            requests_remaining=750,
            tokens_limit=50000,
            tokens_remaining=30000,
            reset_timestamp=reset_time,
            retry_after_seconds=60,
        )

        assert rate_limit.requests_limit == 1000
        assert rate_limit.requests_remaining == 750
        assert rate_limit.tokens_limit == 50000
        assert rate_limit.tokens_remaining == 30000
        assert rate_limit.reset_timestamp == reset_time
        assert rate_limit.retry_after_seconds == 60

    def test_standard_rate_limit_with_none_values(self):
        """Test StandardRateLimit with None values."""
        rate_limit = StandardRateLimit()

        assert rate_limit.requests_limit is None
        assert rate_limit.requests_remaining is None
        assert rate_limit.tokens_limit is None
        assert rate_limit.tokens_remaining is None
        assert rate_limit.reset_timestamp is None
        assert rate_limit.retry_after_seconds is None

    def test_standard_rate_limit_partial_data(self):
        """Test StandardRateLimit with partial data."""
        rate_limit = StandardRateLimit(
            requests_limit=1000,
            requests_remaining=750,
        )

        assert rate_limit.requests_limit == 1000
        assert rate_limit.requests_remaining == 750
        assert rate_limit.tokens_limit is None
        assert rate_limit.tokens_remaining is None

    def test_standard_rate_limit_invalid_retry_after(self):
        """Test StandardRateLimit with invalid retry_after_seconds."""
        with pytest.raises(ValidationError) as exc_info:
            StandardRateLimit(retry_after_seconds=-1)

        assert "retry_after_seconds must be non-negative" in str(exc_info.value)

    def test_standard_rate_limit_zero_retry_after(self):
        """Test StandardRateLimit with zero retry_after_seconds."""
        rate_limit = StandardRateLimit(retry_after_seconds=0)
        assert rate_limit.retry_after_seconds == 0

    def test_standard_rate_limit_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            StandardRateLimit(extra_field="not_allowed")


@pytest.mark.unit
class TestOAuthUnifiedRateLimit:
    """Test OAuthUnifiedRateLimit model."""

    def test_valid_oauth_rate_limit(self):
        """Test creating a valid OAuthUnifiedRateLimit."""
        reset_time = datetime.now(UTC)
        rate_limit = OAuthUnifiedRateLimit(
            status="allowed",
            representative_claim="five_hour",
            fallback_percentage=75.5,
            reset_timestamp=reset_time,
        )

        assert rate_limit.status == "allowed"
        assert rate_limit.representative_claim == "five_hour"
        assert rate_limit.fallback_percentage == 75.5
        assert rate_limit.reset_timestamp == reset_time

    def test_oauth_rate_limit_with_none_values(self):
        """Test OAuthUnifiedRateLimit with None values."""
        rate_limit = OAuthUnifiedRateLimit()

        assert rate_limit.status is None
        assert rate_limit.representative_claim is None
        assert rate_limit.fallback_percentage is None
        assert rate_limit.reset_timestamp is None

    def test_oauth_rate_limit_denied_status(self):
        """Test OAuthUnifiedRateLimit with denied status."""
        rate_limit = OAuthUnifiedRateLimit(status="denied")
        assert rate_limit.status == "denied"

    def test_oauth_rate_limit_invalid_status(self):
        """Test OAuthUnifiedRateLimit with invalid status."""
        with pytest.raises(ValidationError) as exc_info:
            OAuthUnifiedRateLimit(status="invalid")

        assert "status must be 'allowed' or 'denied'" in str(exc_info.value)

    def test_oauth_rate_limit_invalid_fallback_percentage_negative(self):
        """Test OAuthUnifiedRateLimit with negative fallback_percentage."""
        with pytest.raises(ValidationError) as exc_info:
            OAuthUnifiedRateLimit(fallback_percentage=-1.0)

        assert "fallback_percentage must be between 0 and 100" in str(exc_info.value)

    def test_oauth_rate_limit_invalid_fallback_percentage_over_100(self):
        """Test OAuthUnifiedRateLimit with fallback_percentage over 100."""
        with pytest.raises(ValidationError) as exc_info:
            OAuthUnifiedRateLimit(fallback_percentage=101.0)

        assert "fallback_percentage must be between 0 and 100" in str(exc_info.value)

    def test_oauth_rate_limit_edge_fallback_percentages(self):
        """Test OAuthUnifiedRateLimit with edge case fallback_percentage values."""
        rate_limit_0 = OAuthUnifiedRateLimit(fallback_percentage=0.0)
        assert rate_limit_0.fallback_percentage == 0.0

        rate_limit_100 = OAuthUnifiedRateLimit(fallback_percentage=100.0)
        assert rate_limit_100.fallback_percentage == 100.0

    def test_oauth_rate_limit_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            OAuthUnifiedRateLimit(extra_field="not_allowed")


@pytest.mark.unit
class TestRateLimitData:
    """Test RateLimitData model."""

    def test_valid_rate_limit_data_api_key(self):
        """Test creating valid RateLimitData with API key auth."""
        timestamp = datetime.now(UTC)
        standard_limit = StandardRateLimit(
            requests_limit=1000,
            requests_remaining=750,
        )

        rate_limit_data = RateLimitData(
            auth_type="api_key",
            standard=standard_limit,
            timestamp=timestamp,
        )

        assert rate_limit_data.auth_type == "api_key"
        assert rate_limit_data.standard == standard_limit
        assert rate_limit_data.oauth_unified is None
        assert rate_limit_data.timestamp == timestamp

    def test_valid_rate_limit_data_oauth(self):
        """Test creating valid RateLimitData with OAuth auth."""
        timestamp = datetime.now(UTC)
        oauth_limit = OAuthUnifiedRateLimit(
            status="allowed",
            representative_claim="five_hour",
        )

        rate_limit_data = RateLimitData(
            auth_type="oauth",
            oauth_unified=oauth_limit,
            timestamp=timestamp,
        )

        assert rate_limit_data.auth_type == "oauth"
        assert rate_limit_data.oauth_unified == oauth_limit
        assert rate_limit_data.standard is None
        assert rate_limit_data.timestamp == timestamp

    def test_valid_rate_limit_data_unknown(self):
        """Test creating valid RateLimitData with unknown auth."""
        timestamp = datetime.now(UTC)

        rate_limit_data = RateLimitData(
            auth_type="unknown",
            timestamp=timestamp,
        )

        assert rate_limit_data.auth_type == "unknown"
        assert rate_limit_data.standard is None
        assert rate_limit_data.oauth_unified is None
        assert rate_limit_data.timestamp == timestamp

    def test_rate_limit_data_with_both_types(self):
        """Test RateLimitData with both standard and oauth_unified data."""
        timestamp = datetime.now(UTC)
        standard_limit = StandardRateLimit(requests_limit=1000)
        oauth_limit = OAuthUnifiedRateLimit(status="allowed")

        rate_limit_data = RateLimitData(
            auth_type="api_key",
            standard=standard_limit,
            oauth_unified=oauth_limit,
            timestamp=timestamp,
        )

        assert rate_limit_data.standard == standard_limit
        assert rate_limit_data.oauth_unified == oauth_limit

    def test_rate_limit_data_invalid_auth_type(self):
        """Test RateLimitData with invalid auth_type."""
        timestamp = datetime.now(UTC)

        with pytest.raises(ValidationError):
            RateLimitData(
                auth_type="invalid_type",  # type: ignore
                timestamp=timestamp,
            )

    def test_rate_limit_data_missing_timestamp(self):
        """Test RateLimitData without timestamp."""
        with pytest.raises(ValidationError):
            RateLimitData(auth_type="api_key")

    def test_rate_limit_data_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        timestamp = datetime.now(UTC)

        with pytest.raises(ValidationError):
            RateLimitData(
                auth_type="api_key",
                timestamp=timestamp,
                extra_field="not_allowed",
            )


@pytest.mark.unit
class TestRateLimitStatus:
    """Test RateLimitStatus model."""

    def test_valid_rate_limit_status(self):
        """Test creating a valid RateLimitStatus."""
        exhaustion_time = datetime.now(UTC)
        status = RateLimitStatus(
            auth_type="api_key",
            is_limited=True,
            utilization_percentage=85.5,
            time_until_reset=3600,
            estimated_exhaustion_time=exhaustion_time,
        )

        assert status.auth_type == "api_key"
        assert status.is_limited is True
        assert status.utilization_percentage == 85.5
        assert status.time_until_reset == 3600
        assert status.estimated_exhaustion_time == exhaustion_time

    def test_rate_limit_status_not_limited(self):
        """Test RateLimitStatus when not limited."""
        status = RateLimitStatus(
            auth_type="oauth",
            is_limited=False,
            utilization_percentage=25.0,
        )

        assert status.auth_type == "oauth"
        assert status.is_limited is False
        assert status.utilization_percentage == 25.0
        assert status.time_until_reset is None
        assert status.estimated_exhaustion_time is None

    def test_rate_limit_status_minimal(self):
        """Test RateLimitStatus with minimal data."""
        status = RateLimitStatus(
            auth_type="unknown",
            is_limited=False,
        )

        assert status.auth_type == "unknown"
        assert status.is_limited is False
        assert status.utilization_percentage is None
        assert status.time_until_reset is None
        assert status.estimated_exhaustion_time is None

    def test_rate_limit_status_invalid_utilization_percentage_negative(self):
        """Test RateLimitStatus with negative utilization_percentage."""
        with pytest.raises(ValidationError) as exc_info:
            RateLimitStatus(
                auth_type="api_key",
                is_limited=True,
                utilization_percentage=-1.0,
            )

        assert "utilization_percentage must be between 0 and 100" in str(exc_info.value)

    def test_rate_limit_status_invalid_utilization_percentage_over_100(self):
        """Test RateLimitStatus with utilization_percentage over 100."""
        with pytest.raises(ValidationError) as exc_info:
            RateLimitStatus(
                auth_type="api_key",
                is_limited=True,
                utilization_percentage=101.0,
            )

        assert "utilization_percentage must be between 0 and 100" in str(exc_info.value)

    def test_rate_limit_status_edge_utilization_percentages(self):
        """Test RateLimitStatus with edge case utilization_percentage values."""
        status_0 = RateLimitStatus(
            auth_type="api_key",
            is_limited=False,
            utilization_percentage=0.0,
        )
        assert status_0.utilization_percentage == 0.0

        status_100 = RateLimitStatus(
            auth_type="api_key",
            is_limited=True,
            utilization_percentage=100.0,
        )
        assert status_100.utilization_percentage == 100.0

    def test_rate_limit_status_invalid_time_until_reset(self):
        """Test RateLimitStatus with negative time_until_reset."""
        with pytest.raises(ValidationError) as exc_info:
            RateLimitStatus(
                auth_type="api_key",
                is_limited=True,
                time_until_reset=-1,
            )

        assert "time_until_reset must be non-negative" in str(exc_info.value)

    def test_rate_limit_status_zero_time_until_reset(self):
        """Test RateLimitStatus with zero time_until_reset."""
        status = RateLimitStatus(
            auth_type="api_key",
            is_limited=True,
            time_until_reset=0,
        )
        assert status.time_until_reset == 0

    def test_rate_limit_status_missing_required_fields(self):
        """Test RateLimitStatus missing required fields."""
        with pytest.raises(ValidationError):
            RateLimitStatus(is_limited=True)

        with pytest.raises(ValidationError):
            RateLimitStatus(auth_type="api_key")

    def test_rate_limit_status_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            RateLimitStatus(
                auth_type="api_key",
                is_limited=True,
                extra_field="not_allowed",
            )


@pytest.mark.unit
class TestRateLimitModelSerialization:
    """Test serialization and deserialization of rate limit models."""

    def test_standard_rate_limit_serialization(self):
        """Test StandardRateLimit serialization."""
        reset_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        rate_limit = StandardRateLimit(
            requests_limit=1000,
            requests_remaining=750,
            tokens_limit=50000,
            tokens_remaining=30000,
            reset_timestamp=reset_time,
            retry_after_seconds=60,
        )

        data = rate_limit.model_dump()
        assert data["requests_limit"] == 1000
        assert data["requests_remaining"] == 750
        assert data["tokens_limit"] == 50000
        assert data["tokens_remaining"] == 30000
        assert data["reset_timestamp"] == reset_time
        assert data["retry_after_seconds"] == 60

    def test_oauth_unified_rate_limit_serialization(self):
        """Test OAuthUnifiedRateLimit serialization."""
        reset_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        rate_limit = OAuthUnifiedRateLimit(
            status="allowed",
            representative_claim="five_hour",
            fallback_percentage=75.5,
            reset_timestamp=reset_time,
        )

        data = rate_limit.model_dump()
        assert data["status"] == "allowed"
        assert data["representative_claim"] == "five_hour"
        assert data["fallback_percentage"] == 75.5
        assert data["reset_timestamp"] == reset_time

    def test_rate_limit_data_serialization(self):
        """Test RateLimitData serialization."""
        timestamp = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        standard_limit = StandardRateLimit(requests_limit=1000)

        rate_limit_data = RateLimitData(
            auth_type="api_key",
            standard=standard_limit,
            timestamp=timestamp,
        )

        data = rate_limit_data.model_dump()
        assert data["auth_type"] == "api_key"
        assert data["standard"]["requests_limit"] == 1000
        assert data["oauth_unified"] is None
        assert data["timestamp"] == timestamp

    def test_rate_limit_status_serialization(self):
        """Test RateLimitStatus serialization."""
        exhaustion_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        status = RateLimitStatus(
            auth_type="api_key",
            is_limited=True,
            utilization_percentage=85.5,
            time_until_reset=3600,
            estimated_exhaustion_time=exhaustion_time,
        )

        data = status.model_dump()
        assert data["auth_type"] == "api_key"
        assert data["is_limited"] is True
        assert data["utilization_percentage"] == 85.5
        assert data["time_until_reset"] == 3600
        assert data["estimated_exhaustion_time"] == exhaustion_time
