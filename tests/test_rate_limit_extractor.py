"""Tests for the rate limit header extraction utility."""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from ccproxy.utils.rate_limit_extractor import (
    detect_auth_type,
    extract_rate_limit_headers,
    parse_oauth_unified_rate_limits,
    parse_standard_rate_limits,
)


class TestExtractRateLimitHeaders:
    """Test the main rate limit header extraction function."""

    def test_extract_standard_api_key_headers(self):
        """Test extraction of standard API key rate limit headers."""
        headers = {
            "X-RateLimit-Limit": "1000",
            "X-RateLimit-Remaining": "999",
            "X-RateLimit-Reset": "1672531200",
            "Retry-After": "60",
        }

        result = extract_rate_limit_headers(headers)

        assert result["auth_type"] == "api_key"
        assert result["standard"]["limit"] == 1000
        assert result["standard"]["remaining"] == 999
        assert result["standard"]["reset_timestamp"] == 1672531200
        assert result["standard"]["retry_after"] == 60
        assert result["oauth_unified"]["status"] is None
        assert len(result["detected_headers"]) == 4

    def test_extract_oauth_unified_headers(self):
        """Test extraction of OAuth unified rate limit headers."""
        headers = {
            "Anthropic-RateLimit-Unified-Status": "allowed",
            "Anthropic-RateLimit-Unified-Representative-Claim": "five_hour",
            "Anthropic-RateLimit-Unified-Fallback-Percentage": "85.5",
            "Anthropic-RateLimit-Unified-Reset": "1672531200",
        }

        result = extract_rate_limit_headers(headers)

        assert result["auth_type"] == "oauth"
        assert result["oauth_unified"]["status"] == "allowed"
        assert result["oauth_unified"]["representative_claim"] == "five_hour"
        assert result["oauth_unified"]["fallback_percentage"] == 85.5
        assert result["oauth_unified"]["reset_timestamp"] == 1672531200
        assert result["standard"]["limit"] is None
        assert len(result["detected_headers"]) == 4

    def test_extract_mixed_headers(self):
        """Test extraction with both standard and OAuth headers present."""
        headers = {
            "X-RateLimit-Limit": "1000",
            "X-RateLimit-Remaining": "999",
            "Anthropic-RateLimit-Unified-Status": "allowed",
            "Anthropic-RateLimit-Unified-Fallback-Percentage": "75.0",
        }

        result = extract_rate_limit_headers(headers)

        # OAuth headers take precedence in auth type detection
        assert result["auth_type"] == "oauth"
        assert result["standard"]["limit"] == 1000
        assert result["standard"]["remaining"] == 999
        assert result["oauth_unified"]["status"] == "allowed"
        assert result["oauth_unified"]["fallback_percentage"] == 75.0
        assert len(result["detected_headers"]) == 4

    def test_extract_empty_headers(self):
        """Test extraction with no rate limit headers."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer token123",
        }

        result = extract_rate_limit_headers(headers)

        assert result["auth_type"] == "unknown"
        assert result["standard"]["limit"] is None
        assert result["oauth_unified"]["status"] is None
        assert len(result["detected_headers"]) == 0

    def test_extract_case_insensitive_headers(self):
        """Test extraction with case-insensitive header names."""
        headers = {
            "x-ratelimit-limit": "1000",
            "X-RATELIMIT-REMAINING": "999",
            "anthropic-ratelimit-unified-status": "allowed",
            "ANTHROPIC-RATELIMIT-UNIFIED-FALLBACK-PERCENTAGE": "80.0",
        }

        result = extract_rate_limit_headers(headers)

        assert result["auth_type"] == "oauth"
        assert result["standard"]["limit"] == 1000
        assert result["standard"]["remaining"] == 999
        assert result["oauth_unified"]["status"] == "allowed"
        assert result["oauth_unified"]["fallback_percentage"] == 80.0
        assert len(result["detected_headers"]) == 4

    def test_extract_malformed_headers(self):
        """Test extraction with malformed header values."""
        headers = {
            "X-RateLimit-Limit": "not_a_number",
            "X-RateLimit-Remaining": "invalid",
            "X-RateLimit-Reset": "invalid_timestamp",
            "Anthropic-RateLimit-Unified-Fallback-Percentage": "not_a_float",
        }

        result = extract_rate_limit_headers(headers)

        assert result["auth_type"] == "oauth"  # OAuth headers take precedence
        assert result["standard"]["limit"] is None
        assert result["standard"]["remaining"] is None
        assert result["standard"]["reset"] is None
        assert result["oauth_unified"]["fallback_percentage"] is None
        assert len(result["detected_headers"]) == 4

    def test_extract_partial_headers(self):
        """Test extraction with only partial rate limit headers."""
        headers = {
            "X-RateLimit-Limit": "1000",
            "Anthropic-RateLimit-Unified-Status": "allowed",
        }

        result = extract_rate_limit_headers(headers)

        assert result["auth_type"] == "oauth"
        assert result["standard"]["limit"] == 1000
        assert result["standard"]["remaining"] is None
        assert result["oauth_unified"]["status"] == "allowed"
        assert result["oauth_unified"]["fallback_percentage"] is None
        assert len(result["detected_headers"]) == 2


class TestParseStandardRateLimits:
    """Test the standard rate limit header parsing."""

    def test_parse_complete_standard_headers(self):
        """Test parsing complete standard rate limit headers."""
        headers = {
            "X-RateLimit-Limit": "1000",
            "X-RateLimit-Remaining": "750",
            "X-RateLimit-Reset": "1672531200",
            "Retry-After": "120",
        }

        result = parse_standard_rate_limits(headers)

        assert result["limit"] == 1000
        assert result["remaining"] == 750
        assert result["reset_timestamp"] == 1672531200
        assert result["retry_after"] == 120
        assert isinstance(result["reset"], datetime)
        assert result["reset"].tzinfo is not None  # Should have timezone info

    def test_parse_case_insensitive_headers(self):
        """Test parsing with case-insensitive header names."""
        headers = {
            "x-ratelimit-limit": "500",
            "X-RATELIMIT-REMAINING": "400",
            "x-RateLimit-Reset": "1672531200",
            "RETRY-AFTER": "60",
        }

        result = parse_standard_rate_limits(headers)

        assert result["limit"] == 500
        assert result["remaining"] == 400
        assert result["reset_timestamp"] == 1672531200
        assert result["retry_after"] == 60

    def test_parse_partial_headers(self):
        """Test parsing with only some headers present."""
        headers = {
            "X-RateLimit-Limit": "1000",
            "X-RateLimit-Remaining": "999",
        }

        result = parse_standard_rate_limits(headers)

        assert result["limit"] == 1000
        assert result["remaining"] == 999
        assert result["reset"] is None
        assert result["reset_timestamp"] is None
        assert result["retry_after"] is None

    def test_parse_malformed_values(self):
        """Test parsing with malformed header values."""
        headers = {
            "X-RateLimit-Limit": "not_a_number",
            "X-RateLimit-Remaining": "invalid",
            "X-RateLimit-Reset": "invalid_timestamp",
            "Retry-After": "not_an_integer",
        }

        result = parse_standard_rate_limits(headers)

        assert result["limit"] is None
        assert result["remaining"] is None
        assert result["reset"] is None
        assert result["reset_timestamp"] is None
        assert result["retry_after"] is None

    def test_parse_empty_headers(self):
        """Test parsing with no relevant headers."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer token123",
        }

        result = parse_standard_rate_limits(headers)

        assert result["limit"] is None
        assert result["remaining"] is None
        assert result["reset"] is None
        assert result["reset_timestamp"] is None
        assert result["retry_after"] is None

    def test_parse_negative_values(self):
        """Test parsing with negative values."""
        headers = {
            "X-RateLimit-Limit": "-1000",
            "X-RateLimit-Remaining": "-10",
            "Retry-After": "-60",
        }

        result = parse_standard_rate_limits(headers)

        assert result["limit"] == -1000
        assert result["remaining"] == -10
        assert result["retry_after"] == -60

    def test_parse_zero_values(self):
        """Test parsing with zero values."""
        headers = {
            "X-RateLimit-Limit": "0",
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": "0",
            "Retry-After": "0",
        }

        result = parse_standard_rate_limits(headers)

        assert result["limit"] == 0
        assert result["remaining"] == 0
        assert result["reset_timestamp"] == 0
        assert result["retry_after"] == 0
        assert isinstance(result["reset"], datetime)

    def test_parse_very_large_timestamp(self):
        """Test parsing with very large timestamp values."""
        headers = {
            "X-RateLimit-Reset": "999999999999",  # Very large timestamp
        }

        result = parse_standard_rate_limits(headers)

        # Should handle large timestamps gracefully
        assert result["reset_timestamp"] == 999999999999
        # Reset datetime might be None if timestamp is too large
        if result["reset"] is not None:
            assert isinstance(result["reset"], datetime)


class TestParseOAuthUnifiedRateLimits:
    """Test the OAuth unified rate limit header parsing."""

    def test_parse_complete_oauth_headers(self):
        """Test parsing complete OAuth unified rate limit headers."""
        headers = {
            "Anthropic-RateLimit-Unified-Status": "allowed",
            "Anthropic-RateLimit-Unified-Representative-Claim": "five_hour",
            "Anthropic-RateLimit-Unified-Fallback-Percentage": "85.5",
            "Anthropic-RateLimit-Unified-Reset": "1672531200",
        }

        result = parse_oauth_unified_rate_limits(headers)

        assert result["status"] == "allowed"
        assert result["representative_claim"] == "five_hour"
        assert result["fallback_percentage"] == 85.5
        assert result["reset_timestamp"] == 1672531200
        assert isinstance(result["reset"], datetime)
        assert result["reset"].tzinfo is not None

    def test_parse_case_insensitive_oauth_headers(self):
        """Test parsing with case-insensitive OAuth header names."""
        headers = {
            "anthropic-ratelimit-unified-status": "denied",
            "ANTHROPIC-RATELIMIT-UNIFIED-REPRESENTATIVE-CLAIM": "one_hour",
            "anthropic-RateLimit-Unified-Fallback-Percentage": "75.0",
            "ANTHROPIC-RATELIMIT-UNIFIED-RESET": "1672531200",
        }

        result = parse_oauth_unified_rate_limits(headers)

        assert result["status"] == "denied"
        assert result["representative_claim"] == "one_hour"
        assert result["fallback_percentage"] == 75.0
        assert result["reset_timestamp"] == 1672531200

    def test_parse_partial_oauth_headers(self):
        """Test parsing with only some OAuth headers present."""
        headers = {
            "Anthropic-RateLimit-Unified-Status": "allowed",
            "Anthropic-RateLimit-Unified-Fallback-Percentage": "90.0",
        }

        result = parse_oauth_unified_rate_limits(headers)

        assert result["status"] == "allowed"
        assert result["representative_claim"] is None
        assert result["fallback_percentage"] == 90.0
        assert result["reset"] is None
        assert result["reset_timestamp"] is None

    def test_parse_malformed_oauth_values(self):
        """Test parsing with malformed OAuth header values."""
        headers = {
            "Anthropic-RateLimit-Unified-Status": "invalid_status",
            "Anthropic-RateLimit-Unified-Fallback-Percentage": "not_a_float",
            "Anthropic-RateLimit-Unified-Reset": "invalid_timestamp",
        }

        result = parse_oauth_unified_rate_limits(headers)

        assert result["status"] == "invalid_status"  # Still stored, but warning logged
        assert result["representative_claim"] is None
        assert result["fallback_percentage"] is None
        assert result["reset"] is None
        assert result["reset_timestamp"] is None

    def test_parse_empty_oauth_headers(self):
        """Test parsing with no OAuth headers."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer token123",
        }

        result = parse_oauth_unified_rate_limits(headers)

        assert result["status"] is None
        assert result["representative_claim"] is None
        assert result["fallback_percentage"] is None
        assert result["reset"] is None
        assert result["reset_timestamp"] is None

    def test_parse_valid_oauth_status_values(self):
        """Test parsing with valid OAuth status values."""
        for status in ["allowed", "denied"]:
            headers = {
                "Anthropic-RateLimit-Unified-Status": status,
            }

            result = parse_oauth_unified_rate_limits(headers)
            assert result["status"] == status

    def test_parse_boundary_fallback_percentages(self):
        """Test parsing with boundary fallback percentage values."""
        test_cases = [
            ("0.0", 0.0),
            ("100.0", 100.0),
            ("0.1", 0.1),
            ("99.9", 99.9),
            ("50.5", 50.5),
        ]

        for percentage_str, expected in test_cases:
            headers = {
                "Anthropic-RateLimit-Unified-Fallback-Percentage": percentage_str,
            }

            result = parse_oauth_unified_rate_limits(headers)
            assert result["fallback_percentage"] == expected

    def test_parse_negative_fallback_percentage(self):
        """Test parsing with negative fallback percentage."""
        headers = {
            "Anthropic-RateLimit-Unified-Fallback-Percentage": "-10.0",
        }

        result = parse_oauth_unified_rate_limits(headers)
        assert result["fallback_percentage"] == -10.0


class TestDetectAuthType:
    """Test the authentication type detection."""

    def test_detect_api_key_auth(self):
        """Test detection of API key authentication."""
        headers = {
            "X-RateLimit-Limit": "1000",
            "X-RateLimit-Remaining": "999",
        }

        result = detect_auth_type(headers)
        assert result == "api_key"

    def test_detect_oauth_auth(self):
        """Test detection of OAuth authentication."""
        headers = {
            "Anthropic-RateLimit-Unified-Status": "allowed",
            "Anthropic-RateLimit-Unified-Fallback-Percentage": "85.0",
        }

        result = detect_auth_type(headers)
        assert result == "oauth"

    def test_detect_oauth_precedence(self):
        """Test that OAuth headers take precedence over API key headers."""
        headers = {
            "X-RateLimit-Limit": "1000",
            "X-RateLimit-Remaining": "999",
            "Anthropic-RateLimit-Unified-Status": "allowed",
        }

        result = detect_auth_type(headers)
        assert result == "oauth"

    def test_detect_unknown_auth(self):
        """Test detection when no rate limit headers are present."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer token123",
        }

        result = detect_auth_type(headers)
        assert result == "unknown"

    def test_detect_case_insensitive_api_key(self):
        """Test case-insensitive detection of API key headers."""
        headers = {
            "x-ratelimit-limit": "1000",
            "X-RATELIMIT-REMAINING": "999",
        }

        result = detect_auth_type(headers)
        assert result == "api_key"

    def test_detect_case_insensitive_oauth(self):
        """Test case-insensitive detection of OAuth headers."""
        headers = {
            "anthropic-ratelimit-unified-status": "allowed",
            "ANTHROPIC-RATELIMIT-UNIFIED-FALLBACK-PERCENTAGE": "85.0",
        }

        result = detect_auth_type(headers)
        assert result == "oauth"

    def test_detect_single_header_types(self):
        """Test detection with single headers of each type."""
        # Test each individual header type
        api_key_headers = [
            "x-ratelimit-limit",
            "x-ratelimit-remaining",
            "x-ratelimit-reset",
            "retry-after",
        ]

        oauth_headers = [
            "anthropic-ratelimit-unified-status",
            "anthropic-ratelimit-unified-representative-claim",
            "anthropic-ratelimit-unified-fallback-percentage",
            "anthropic-ratelimit-unified-reset",
        ]

        for header in api_key_headers:
            headers = {header: "test_value"}
            result = detect_auth_type(headers)
            assert result == "api_key", f"Failed for header: {header}"

        for header in oauth_headers:
            headers = {header: "test_value"}
            result = detect_auth_type(headers)
            assert result == "oauth", f"Failed for header: {header}"

    def test_detect_empty_headers(self):
        """Test detection with empty headers dictionary."""
        headers = {}

        result = detect_auth_type(headers)
        assert result == "unknown"

    def test_detect_with_irrelevant_headers(self):
        """Test detection with only irrelevant headers."""
        headers = {
            "content-type": "application/json",
            "user-agent": "test-client",
            "accept": "application/json",
        }

        result = detect_auth_type(headers)
        assert result == "unknown"
