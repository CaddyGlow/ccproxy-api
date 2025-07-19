# Service Layer Mocking Migration Guide

This guide explains the restructured service layer mocking and how to migrate to the new organized fixtures.

## Summary of Changes

### Before: Confusing Naming
```python
# OLD - unclear purpose
mock_claude_service     # Internal service mock? External API mock?
mock_claude            # Which Claude? SDK or API?
mock_oauth             # What kind of OAuth mocking?
```

### After: Clear Intent  
```python
# NEW - clear mocking strategy and scope
mock_internal_claude_sdk_service    # Internal dependency injection mock
mock_external_anthropic_api         # External HTTP interception mock  
mock_external_oauth_endpoints       # External OAuth endpoint mocks
```

## Mocking Strategy Separation

### 1. Internal Service Mocking
**Purpose**: Mock services for dependency injection testing  
**Technology**: AsyncMock from unittest.mock  
**Location**: `tests/fixtures/claude_sdk/internal_mocks.py`

**When to Use**: Testing API endpoints that inject ClaudeSDKService
```python
def test_api_endpoint(client: TestClient, mock_internal_claude_sdk_service: AsyncMock):
    # Tests FastAPI endpoint with mocked dependency injection
    response = client.post("/sdk/v1/messages", json=request_data)
    assert response.status_code == 200
```

### 2. External API Mocking  
**Purpose**: Intercept HTTP calls to external APIs  
**Technology**: pytest-httpx (HTTPXMock)  
**Location**: `tests/fixtures/external_apis/anthropic_api.py`

**When to Use**: Testing ProxyService and HTTP forwarding
```python  
def test_proxy_service(mock_external_anthropic_api: HTTPXMock):
    # Tests HTTP calls to api.anthropic.com
    service = ProxyService()
    response = await service.forward_request(request_data)
    assert response.status_code == 200
```

### 3. OAuth Service Mocking
**Purpose**: Mock OAuth token endpoints  
**Technology**: pytest-httpx (HTTPXMock)  
**Location**: `tests/fixtures/proxy_service/oauth_mocks.py`

**When to Use**: Testing authentication flows
```python
def test_oauth_flow(mock_external_oauth_endpoints: HTTPXMock):
    # Tests OAuth token exchange/refresh endpoints  
    client = OAuthClient()
    tokens = await client.exchange_tokens(auth_code)
    assert tokens.access_token
```

## Migration Options

### Option 1: No Change Required (Recommended)
Existing tests continue working with backward compatibility aliases:

```python
# ✅ Existing code works unchanged
def test_endpoint(mock_claude_service: AsyncMock):
    # Uses mock_internal_claude_sdk_service internally
    pass

def test_api(mock_claude: HTTPXMock):  
    # Uses mock_external_anthropic_api internally
    pass
```

### Option 2: Migrate to New Names (Improved Clarity)
For new tests or when refactoring, use descriptive names:

```python
# ✅ New way - explicit intent  
def test_endpoint(mock_internal_claude_sdk_service: AsyncMock):
    # Clear: testing with internal service dependency injection
    pass

def test_proxy(mock_external_anthropic_api: HTTPXMock):
    # Clear: testing with external HTTP interception  
    pass
```

## Directory Structure

```
tests/fixtures/
├── claude_sdk/
│   ├── __init__.py
│   ├── internal_mocks.py      # AsyncMock for ClaudeSDKService  
│   └── responses.py           # Standard response data
├── external_apis/
│   ├── __init__.py
│   └── anthropic_api.py       # HTTPXMock for api.anthropic.com
├── proxy_service/
│   ├── __init__.py
│   └── oauth_mocks.py         # HTTPXMock for OAuth endpoints
├── README.md                  # Detailed documentation
└── MIGRATION_GUIDE.md         # This guide
```

## Fixture Mapping

| Old Name | New Name | Purpose |
|----------|----------|---------|
| `mock_claude_service` | `mock_internal_claude_sdk_service` | Internal dependency injection |
| `mock_claude_service_streaming` | `mock_internal_claude_sdk_service_streaming` | Internal streaming dependency |
| `mock_claude_service_unavailable` | `mock_internal_claude_sdk_service_unavailable` | Internal unavailable simulation |
| `mock_claude` | `mock_external_anthropic_api` | External HTTP interception |  
| `mock_claude_stream` | `mock_external_anthropic_api_streaming` | External streaming HTTP |
| `mock_oauth` | `mock_external_oauth_endpoints` | External OAuth endpoints |

## Best Practices

### For New Tests
1. **Choose the right strategy**: Internal mocking for dependency injection, external mocking for HTTP calls
2. **Use descriptive names**: `mock_internal_*` vs `mock_external_*` makes intent clear
3. **Combine when needed**: Some tests may need both internal and external mocks

### For Existing Tests  
1. **No rush to migrate**: Backward compatibility aliases work perfectly
2. **Migrate during refactoring**: When touching test code, consider upgrading names
3. **Keep consistency**: Within a test file, prefer consistent naming style

## Common Patterns

### Pure Internal Testing
```python
def test_sdk_endpoint(mock_internal_claude_sdk_service: AsyncMock):
    # Test FastAPI endpoint with mocked ClaudeSDKService
    mock_internal_claude_sdk_service.create_completion.return_value = {...}
    response = client.post("/sdk/v1/messages", json=request_data)
    assert response.status_code == 200
```

### Pure External Testing  
```python
def test_proxy_forwarding(mock_external_anthropic_api: HTTPXMock):
    # Test ProxyService HTTP forwarding to api.anthropic.com
    service = ProxyService()
    response = await service.forward_request(request_data)
    assert response.status_code == 200
```

### Mixed Testing
```python  
def test_complete_flow(
    mock_internal_claude_sdk_service: AsyncMock,
    mock_external_oauth_endpoints: HTTPXMock
):
    # Test both internal dependencies and external HTTP calls
    pass
```

## Troubleshooting

### Import Errors
```python
# ❌ Old direct imports won't work
from tests.fixtures.claude_sdk.internal_mocks import mock_claude_service

# ✅ Use as pytest fixtures  
def test_example(mock_internal_claude_sdk_service: AsyncMock):
    pass
```

### Type Errors
```python
# ✅ Proper type hints
def test_internal(mock_internal_claude_sdk_service: AsyncMock):
    pass

def test_external(mock_external_anthropic_api: HTTPXMock):  
    pass
```

### Backward Compatibility Issues
If old fixture names stop working, check:
1. Fixture aliases are properly imported in `conftest.py`
2. No circular import issues
3. Pytest can discover the fixtures

## Benefits

1. **Clear Intent**: Fixture names indicate mocking strategy
2. **Organized Code**: Related fixtures grouped by service/purpose
3. **Maintainability**: Centralized response data and documentation  
4. **Type Safety**: Proper type hints and clear interfaces
5. **Backward Compatible**: Existing tests work without changes
6. **Future Proof**: Easy to extend with new mocking strategies

## Questions?

See `tests/fixtures/README.md` for detailed documentation, or check the existing test files for usage examples.
