"""OAuth-related mocks for ProxyService testing.

These fixtures intercept HTTP calls to OAuth endpoints for testing
authentication and token management functionality.
"""

import pytest
from pytest_httpx import HTTPXMock


@pytest.fixture
def mock_external_oauth_endpoints(httpx_mock: HTTPXMock) -> HTTPXMock:
    """Mock OAuth token endpoints for authentication testing.

    This fixture intercepts HTTP calls to OAuth token endpoints and returns
    mock responses for testing ProxyService OAuth functionality.

    Mocking Strategy: External HTTP interception via httpx_mock
    Use Case: Testing OAuth token exchange and refresh flows
    HTTP Calls: Intercepted OAuth endpoint calls

    Args:
        httpx_mock: HTTPXMock fixture for HTTP interception

    Returns:
        HTTPXMock configured with OAuth responses
    """
    # Mock token exchange
    httpx_mock.add_response(
        url="https://api.anthropic.com/oauth/token",
        json={
            "access_token": "test_access_token",
            "refresh_token": "test_refresh_token",
            "expires_in": 3600,
            "token_type": "Bearer",
        },
        status_code=200,
    )

    # Mock token refresh
    httpx_mock.add_response(
        url="https://api.anthropic.com/oauth/refresh",
        json={
            "access_token": "new_test_access_token",
            "expires_in": 3600,
            "token_type": "Bearer",
        },
        status_code=200,
    )

    return httpx_mock


@pytest.fixture
def mock_external_oauth_endpoints_error(httpx_mock: HTTPXMock) -> HTTPXMock:
    """Mock OAuth token endpoints with error responses.

    This fixture intercepts HTTP calls to OAuth endpoints and returns
    error responses for testing authentication error handling.

    Mocking Strategy: External HTTP interception via httpx_mock
    Use Case: Testing OAuth error handling (invalid credentials, expired tokens)
    HTTP Calls: Intercepted OAuth endpoint calls with errors

    Args:
        httpx_mock: HTTPXMock fixture for HTTP interception

    Returns:
        HTTPXMock configured with OAuth error responses
    """
    # Mock token exchange error
    httpx_mock.add_response(
        url="https://api.anthropic.com/oauth/token",
        json={
            "error": "invalid_grant",
            "error_description": "The provided authorization grant is invalid",
        },
        status_code=400,
    )

    # Mock token refresh error
    httpx_mock.add_response(
        url="https://api.anthropic.com/oauth/refresh",
        json={
            "error": "invalid_grant",
            "error_description": "The refresh token is invalid or expired",
        },
        status_code=400,
    )

    return httpx_mock
