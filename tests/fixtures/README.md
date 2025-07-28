# Test Fixtures Organization

This directory contains organized test fixtures that provide clear separation between different mocking strategies used in the ccproxy test suite.

## Structure Overview

```
tests/fixtures/
├── claude_sdk/           # Claude SDK service mocking
│   ├── internal_mocks.py # AsyncMock for dependency injection
│   └── responses.py      # Standard response data
├── proxy_service/        # Proxy service mocking  
│   └── oauth_mocks.py    # OAuth endpoint HTTP mocks
├── external_apis/        # External API HTTP mocking
│   └── anthropic_api.py  # api.anthropic.com HTTP intercepts
└── README.md            # This documentation
```

## Mocking Strategies

### 1. Internal Service Mocking (Claude SDK)

**Purpose**: Mock ClaudeSDKService for dependency injection testing  
**Location**: `tests/fixtures/claude_sdk/internal_mocks.py`  
**Technology**: AsyncMock from unittest.mock  
**Use Case**: Testing API endpoints that depend on Claude SDK without HTTP calls

**Fixtures**:
- `mock_internal_claude_sdk_service` - Standard completion mocking
- `mock_internal_claude_sdk_service_streaming` - Streaming response mocking  
- `mock_internal_claude_sdk_service_unavailable` - Service unavailable simulation

**Example Usage**:
```python
def test_api_endpoint(client: TestClient, mock_internal_claude_sdk_service: AsyncMock):
    # Test API endpoint with mocked Claude SDK dependency
    response = client.post("/sdk/v1/messages", json=request_data)
    assert response.status_code == 200
```

### 2. External API Mocking (HTTP Interception)

**Purpose**: Intercept HTTP calls to external APIs  
**Location**: `tests/fixtures/external_apis/anthropic_api.py`  
**Technology**: pytest-httpx (HTTPXMock)  
**Use Case**: Testing ProxyService and components making direct HTTP calls

**Fixtures**:
- `mock_external_anthropic_api` - Standard API responses
- `mock_external_anthropic_api_streaming` - SSE streaming responses
- `mock_external_anthropic_api_error` - Error response simulation
- `mock_external_anthropic_api_unavailable` - Service unavailable simulation

**Example Usage**:
```python
def test_proxy_service(mock_external_anthropic_api: HTTPXMock):
    # Test ProxyService with intercepted HTTP calls to api.anthropic.com
    service = ProxyService()
    response = await service.forward_request(request_data)
    assert response.status_code == 200
```

### 3. OAuth Service Mocking

**Purpose**: Mock OAuth token endpoints for authentication testing  
**Location**: `tests/fixtures/proxy_service/oauth_mocks.py`  
**Technology**: pytest-httpx (HTTPXMock)  
**Use Case**: Testing OAuth flows and credential management

**Fixtures**:
- `mock_external_oauth_endpoints` - Success token exchange/refresh
- `mock_external_oauth_endpoints_error` - OAuth error responses

## Usage

Use descriptive fixture names for clear intent:

```python
def test_endpoint(mock_internal_claude_sdk_service: AsyncMock):
    # Testing with internal service dependency injection
    pass

def test_proxy(mock_external_anthropic_api: HTTPXMock):  
    # Testing with external HTTP interception
    pass
```

## Response Data Management

Standard response data is centralized in `tests/fixtures/claude_sdk/responses.py`:

```python
from tests.fixtures.claude_sdk.responses import (
    CLAUDE_SDK_STANDARD_COMPLETION,
    CLAUDE_SDK_STREAMING_EVENTS,
    SUPPORTED_CLAUDE_MODELS
)
```

## Key Benefits

1. **Clear Purpose**: Fixture names indicate mocking strategy and scope
2. **Organized Structure**: Related fixtures grouped by service/strategy  
3. **Maintainability**: Centralized response data and clear documentation
4. **Type Safety**: Proper type hints and documentation for each fixture

## Common Patterns

### Internal Service Testing
Use when testing FastAPI endpoints that inject ClaudeSDKService:
- API route handlers
- Dependency injection scenarios
- Service layer unit tests

### External API Testing  
Use when testing components that make HTTP calls:
- ProxyService HTTP forwarding
- OAuth authentication flows
- Error handling for external API failures

### Mixed Testing
Some tests may need both strategies for comprehensive coverage:
```python
def test_complete_flow(
    mock_internal_claude_sdk_service: AsyncMock,
    mock_external_oauth_endpoints: HTTPXMock
):
    # Test both internal service dependencies and external HTTP calls
    pass
```
