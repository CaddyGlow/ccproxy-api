"""Rate limit header extraction utilities for Claude Code Proxy API.

This module provides utilities for parsing standard API key rate limit headers
and OAuth unified rate limit headers from HTTP responses.
"""

from datetime import UTC, datetime
from typing import Any

from ccproxy.utils.logging import get_logger


logger = get_logger(__name__)


def extract_rate_limit_headers(headers: dict[str, str]) -> dict[str, Any]:
    """Extract rate limit headers from HTTP response headers.

    This function parses both standard API key rate limit headers and OAuth
    unified rate limit headers, automatically detecting the authentication type.

    Args:
        headers: HTTP response headers dictionary

    Returns:
        Dictionary containing parsed rate limit information with the following
        structure:
        {
            'auth_type': str,  # 'api_key', 'oauth', or 'unknown'
            'standard': dict,  # Standard API key rate limit headers
            'oauth_unified': dict,  # OAuth unified rate limit headers
            'detected_headers': list[str],  # List of rate limit headers found
        }

    Example:
        >>> headers = {
        ...     'x-ratelimit-limit': '1000',
        ...     'x-ratelimit-remaining': '999',
        ...     'x-ratelimit-reset': '1672531200'
        ... }
        >>> result = extract_rate_limit_headers(headers)
        >>> result['auth_type']
        'api_key'
        >>> result['standard']['limit']
        1000
    """
    logger.debug(f"Extracting rate limit headers from {len(headers)} headers")

    # Parse both types of headers
    standard_limits = parse_standard_rate_limits(headers)
    oauth_unified_limits = parse_oauth_unified_rate_limits(headers)

    # Detect authentication type
    auth_type = detect_auth_type(headers)

    # Collect all detected rate limit headers
    detected_headers = []

    # Standard headers
    standard_header_names = [
        "x-ratelimit-limit",
        "x-ratelimit-remaining",
        "x-ratelimit-reset",
        "retry-after",
    ]

    # OAuth unified headers
    oauth_header_names = [
        "anthropic-ratelimit-unified-status",
        "anthropic-ratelimit-unified-representative-claim",
        "anthropic-ratelimit-unified-fallback-percentage",
        "anthropic-ratelimit-unified-reset",
    ]

    # Check for detected headers (case-insensitive)
    all_possible_headers = standard_header_names + oauth_header_names
    header_keys_lower = {k.lower(): k for k in headers}

    for header_name in all_possible_headers:
        if header_name.lower() in header_keys_lower:
            detected_headers.append(header_name)

    result = {
        "auth_type": auth_type,
        "standard": standard_limits,
        "oauth_unified": oauth_unified_limits,
        "detected_headers": detected_headers,
    }

    logger.debug(
        f"Extracted rate limits: auth_type={auth_type}, "
        f"detected_headers={detected_headers}"
    )

    return result


def parse_standard_rate_limits(headers: dict[str, str]) -> dict[str, Any]:
    """Parse standard API key rate limit headers.

    Extracts rate limit information from standard headers typically used
    with API key authentication.

    Args:
        headers: HTTP response headers dictionary

    Returns:
        Dictionary containing parsed standard rate limit information:
        {
            'limit': int | None,  # Total requests allowed
            'remaining': int | None,  # Remaining requests in current window
            'reset': datetime | None,  # Reset time as datetime object
            'reset_timestamp': int | None,  # Raw reset timestamp
            'retry_after': int | None,  # Retry after seconds
        }

    Example:
        >>> headers = {
        ...     'x-ratelimit-limit': '1000',
        ...     'x-ratelimit-remaining': '999',
        ...     'x-ratelimit-reset': '1672531200'
        ... }
        >>> result = parse_standard_rate_limits(headers)
        >>> result['limit']
        1000
        >>> result['remaining']
        999
    """
    logger.debug("Parsing standard rate limit headers")

    def get_header_case_insensitive(header_name: str) -> str | None:
        """Get header value case-insensitively."""
        for key, value in headers.items():
            if key.lower() == header_name.lower():
                return value
        return None

    # Parse rate limit
    limit = None
    limit_str = get_header_case_insensitive("x-ratelimit-limit")
    if limit_str:
        try:
            limit = int(limit_str)
        except ValueError as e:
            logger.warning(f"Failed to parse rate limit: {limit_str}, error: {e}")

    # Parse remaining requests
    remaining = None
    remaining_str = get_header_case_insensitive("x-ratelimit-remaining")
    if remaining_str:
        try:
            remaining = int(remaining_str)
        except ValueError as e:
            logger.warning(
                f"Failed to parse remaining requests: {remaining_str}, error: {e}"
            )

    # Parse reset timestamp
    reset = None
    reset_timestamp = None
    reset_str = get_header_case_insensitive("x-ratelimit-reset")
    if reset_str:
        try:
            reset_timestamp = int(reset_str)
            reset = datetime.fromtimestamp(reset_timestamp, tz=UTC)
        except (ValueError, OSError) as e:
            logger.warning(f"Failed to parse reset timestamp: {reset_str}, error: {e}")

    # Parse retry after
    retry_after = None
    retry_after_str = get_header_case_insensitive("retry-after")
    if retry_after_str:
        try:
            retry_after = int(retry_after_str)
        except ValueError as e:
            logger.warning(
                f"Failed to parse retry-after: {retry_after_str}, error: {e}"
            )

    result = {
        "limit": limit,
        "remaining": remaining,
        "reset": reset,
        "reset_timestamp": reset_timestamp,
        "retry_after": retry_after,
    }

    logger.debug(f"Parsed standard rate limits: {result}")
    return result


def parse_oauth_unified_rate_limits(headers: dict[str, str]) -> dict[str, Any]:
    """Parse OAuth unified rate limit headers.

    Extracts rate limit information from OAuth unified headers used
    with OAuth authentication.

    Args:
        headers: HTTP response headers dictionary

    Returns:
        Dictionary containing parsed OAuth unified rate limit information:
        {
            'status': str | None,  # 'allowed' or 'denied'
            'representative_claim': str | None,  # e.g., 'five_hour'
            'fallback_percentage': float | None,  # Fallback percentage
            'reset': datetime | None,  # Reset time as datetime object
            'reset_timestamp': int | None,  # Raw reset timestamp
        }

    Example:
        >>> headers = {
        ...     'anthropic-ratelimit-unified-status': 'allowed',
        ...     'anthropic-ratelimit-unified-representative-claim': 'five_hour',
        ...     'anthropic-ratelimit-unified-fallback-percentage': '85.5',
        ...     'anthropic-ratelimit-unified-reset': '1672531200'
        ... }
        >>> result = parse_oauth_unified_rate_limits(headers)
        >>> result['status']
        'allowed'
        >>> result['fallback_percentage']
        85.5
    """
    logger.debug("Parsing OAuth unified rate limit headers")

    def get_header_case_insensitive(header_name: str) -> str | None:
        """Get header value case-insensitively."""
        for key, value in headers.items():
            if key.lower() == header_name.lower():
                return value
        return None

    # Parse status
    status = get_header_case_insensitive("anthropic-ratelimit-unified-status")
    if status and status not in ["allowed", "denied"]:
        logger.warning(f"Unknown OAuth unified status: {status}")

    # Parse representative claim
    representative_claim = get_header_case_insensitive(
        "anthropic-ratelimit-unified-representative-claim"
    )

    # Parse fallback percentage
    fallback_percentage = None
    fallback_str = get_header_case_insensitive(
        "anthropic-ratelimit-unified-fallback-percentage"
    )
    if fallback_str:
        try:
            fallback_percentage = float(fallback_str)
        except ValueError as e:
            logger.warning(
                f"Failed to parse fallback percentage: {fallback_str}, error: {e}"
            )

    # Parse reset timestamp
    reset = None
    reset_timestamp = None
    reset_str = get_header_case_insensitive("anthropic-ratelimit-unified-reset")
    if reset_str:
        try:
            reset_timestamp = int(reset_str)
            reset = datetime.fromtimestamp(reset_timestamp, tz=UTC)
        except (ValueError, OSError) as e:
            logger.warning(
                f"Failed to parse OAuth unified reset timestamp: {reset_str}, "
                f"error: {e}"
            )

    result = {
        "status": status,
        "representative_claim": representative_claim,
        "fallback_percentage": fallback_percentage,
        "reset": reset,
        "reset_timestamp": reset_timestamp,
    }

    logger.debug(f"Parsed OAuth unified rate limits: {result}")
    return result


def detect_auth_type(headers: dict[str, str]) -> str:
    """Detect authentication type from rate limit headers.

    Determines the authentication type used based on the presence of
    specific rate limit headers.

    Args:
        headers: HTTP response headers dictionary

    Returns:
        Authentication type: 'api_key', 'oauth', or 'unknown'

    Example:
        >>> headers = {'x-ratelimit-limit': '1000'}
        >>> detect_auth_type(headers)
        'api_key'
        >>> headers = {'anthropic-ratelimit-unified-status': 'allowed'}
        >>> detect_auth_type(headers)
        'oauth'
    """
    logger.debug("Detecting authentication type from headers")

    # Convert headers to lowercase for case-insensitive comparison
    headers_lower = {k.lower(): v for k, v in headers.items()}

    # Check for OAuth unified headers
    oauth_headers = [
        "anthropic-ratelimit-unified-status",
        "anthropic-ratelimit-unified-representative-claim",
        "anthropic-ratelimit-unified-fallback-percentage",
        "anthropic-ratelimit-unified-reset",
    ]

    if any(header.lower() in headers_lower for header in oauth_headers):
        logger.debug("Detected OAuth authentication from unified headers")
        return "oauth"

    # Check for standard API key headers
    standard_headers = [
        "x-ratelimit-limit",
        "x-ratelimit-remaining",
        "x-ratelimit-reset",
        "retry-after",
    ]

    if any(header.lower() in headers_lower for header in standard_headers):
        logger.debug("Detected API key authentication from standard headers")
        return "api_key"

    logger.debug("Could not detect authentication type")
    return "unknown"


# Unit tests using pytest conventions
if __name__ == "__main__":
    import pytest

    def test_extract_rate_limit_headers_standard() -> None:
        """Test extraction of standard rate limit headers."""
        headers: dict[str, str] = {
            "x-ratelimit-limit": "1000",
            "x-ratelimit-remaining": "999",
            "x-ratelimit-reset": "1672531200",
            "retry-after": "60",
        }

        result = extract_rate_limit_headers(headers)

        assert result["auth_type"] == "api_key"
        assert result["standard"]["limit"] == 1000
        assert result["standard"]["remaining"] == 999
        assert result["standard"]["retry_after"] == 60
        assert result["standard"]["reset_timestamp"] == 1672531200
        assert isinstance(result["standard"]["reset"], datetime)
        assert "x-ratelimit-limit" in result["detected_headers"]
        assert "x-ratelimit-remaining" in result["detected_headers"]
        assert "x-ratelimit-reset" in result["detected_headers"]
        assert "retry-after" in result["detected_headers"]

    def test_extract_rate_limit_headers_oauth() -> None:
        """Test extraction of OAuth unified rate limit headers."""
        headers: dict[str, str] = {
            "anthropic-ratelimit-unified-status": "allowed",
            "anthropic-ratelimit-unified-representative-claim": "five_hour",
            "anthropic-ratelimit-unified-fallback-percentage": "85.5",
            "anthropic-ratelimit-unified-reset": "1672531200",
        }

        result = extract_rate_limit_headers(headers)

        assert result["auth_type"] == "oauth"
        assert result["oauth_unified"]["status"] == "allowed"
        assert result["oauth_unified"]["representative_claim"] == "five_hour"
        assert result["oauth_unified"]["fallback_percentage"] == 85.5
        assert result["oauth_unified"]["reset_timestamp"] == 1672531200
        assert isinstance(result["oauth_unified"]["reset"], datetime)
        assert "anthropic-ratelimit-unified-status" in result["detected_headers"]

    def test_parse_standard_rate_limits() -> None:
        """Test parsing of standard rate limit headers."""
        headers: dict[str, str] = {
            "x-ratelimit-limit": "1000",
            "x-ratelimit-remaining": "999",
            "x-ratelimit-reset": "1672531200",
            "retry-after": "60",
        }

        result = parse_standard_rate_limits(headers)

        assert result["limit"] == 1000
        assert result["remaining"] == 999
        assert result["retry_after"] == 60
        assert result["reset_timestamp"] == 1672531200
        assert isinstance(result["reset"], datetime)
        assert result["reset"].tzinfo == UTC

    def test_parse_oauth_unified_rate_limits() -> None:
        """Test parsing of OAuth unified rate limit headers."""
        headers: dict[str, str] = {
            "anthropic-ratelimit-unified-status": "allowed",
            "anthropic-ratelimit-unified-representative-claim": "five_hour",
            "anthropic-ratelimit-unified-fallback-percentage": "85.5",
            "anthropic-ratelimit-unified-reset": "1672531200",
        }

        result = parse_oauth_unified_rate_limits(headers)

        assert result["status"] == "allowed"
        assert result["representative_claim"] == "five_hour"
        assert result["fallback_percentage"] == 85.5
        assert result["reset_timestamp"] == 1672531200
        assert isinstance(result["reset"], datetime)
        assert result["reset"].tzinfo == UTC

    def test_detect_auth_type() -> None:
        """Test authentication type detection."""
        # Test API key detection
        api_key_headers = {"x-ratelimit-limit": "1000"}
        assert detect_auth_type(api_key_headers) == "api_key"

        # Test OAuth detection
        oauth_headers = {"anthropic-ratelimit-unified-status": "allowed"}
        assert detect_auth_type(oauth_headers) == "oauth"

        # Test unknown detection
        unknown_headers = {"content-type": "application/json"}
        assert detect_auth_type(unknown_headers) == "unknown"

    def test_case_insensitive_headers() -> None:
        """Test case-insensitive header parsing."""
        headers: dict[str, str] = {
            "X-RateLimit-Limit": "1000",
            "X-RATELIMIT-REMAINING": "999",
            "x-ratelimit-reset": "1672531200",
        }

        result = parse_standard_rate_limits(headers)

        assert result["limit"] == 1000
        assert result["remaining"] == 999
        assert result["reset_timestamp"] == 1672531200

    def test_malformed_headers() -> None:
        """Test handling of malformed headers."""
        headers: dict[str, str] = {
            "x-ratelimit-limit": "not_a_number",
            "x-ratelimit-remaining": "",
            "x-ratelimit-reset": "invalid_timestamp",
            "retry-after": "sixty",
        }

        result = parse_standard_rate_limits(headers)

        # Should handle malformed values gracefully
        assert result["limit"] is None
        assert result["remaining"] is None
        assert result["reset"] is None
        assert result["reset_timestamp"] is None
        assert result["retry_after"] is None

    def test_empty_headers() -> None:
        """Test handling of empty headers."""
        headers: dict[str, str] = {}

        result = extract_rate_limit_headers(headers)

        assert result["auth_type"] == "unknown"
        assert result["standard"]["limit"] is None
        assert result["oauth_unified"]["status"] is None
        assert result["detected_headers"] == []

    def test_mixed_headers() -> None:
        """Test handling of mixed standard and OAuth headers."""
        headers: dict[str, str] = {
            "x-ratelimit-limit": "1000",
            "anthropic-ratelimit-unified-status": "allowed",
        }

        result = extract_rate_limit_headers(headers)

        # OAuth headers take precedence in auth type detection
        assert result["auth_type"] == "oauth"
        assert result["standard"]["limit"] == 1000
        assert result["oauth_unified"]["status"] == "allowed"
        assert len(result["detected_headers"]) == 2

    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])
