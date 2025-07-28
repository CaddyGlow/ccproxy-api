# Simplified Testing Guide for CCProxy

## Philosophy
Keep it simple. Test what matters, mock what's external, don't overthink it.

## Quick Start
```bash
# Run all tests
make test

# Run specific test categories
pytest tests/unit/api/           # API endpoint tests
pytest tests/unit/auth/          # Authentication tests
pytest tests/integration/       # Integration tests

# Run with coverage
make test-coverage

# Run with real APIs (optional, slow)
pytest -m real_api
```

## Test Structure

**Organized by functionality** - As the test suite grew beyond 30+ files, we moved from a flat structure to organized categories while maintaining the same testing philosophy.

```
tests/
├── conftest.py              # Shared fixtures + backward compatibility
├── unit/                    # Unit tests organized by component
│   ├── api/                 # API endpoint tests
│   │   ├── test_api.py      # Core API endpoints
│   │   ├── test_mcp_route.py # MCP permission routes
│   │   ├── test_metrics_api.py # Metrics collection endpoints
│   │   ├── test_reset_endpoint.py # Reset endpoint
│   │   ├── test_confirmation_routes.py # Confirmation routes
│   ├── services/            # Service layer tests
│   │   ├── test_adapters.py # OpenAI↔Anthropic conversion
│   │   ├── test_streaming.py # Streaming functionality
│   │   ├── test_docker.py   # Docker integration
│   │   ├── test_confirmation_service.py # Confirmation service
│   │   ├── test_scheduler*.py # Scheduler components
│   │   └── test_*.py        # Other service tests
│   ├── auth/                # Authentication tests
│   │   └── test_auth.py     # Auth + OAuth2 together
│   ├── config/              # Configuration tests
│   │   ├── test_claude_sdk_*.py # Claude SDK configuration
│   │   └── test_terminal_handler.py # Terminal handling
│   ├── utils/               # Utility tests
│   │   ├── test_duckdb_*.py # Database utilities
│   │   └── test_version_checker.py # Version checking
│   └── cli/                 # CLI command tests
│       ├── test_cli_*.py    # CLI command implementations  
│       └── test_cli_confirmation_handler.py # Confirmation CLI handling
├── integration/             # Integration tests
│   ├── test_*_integration.py # Cross-component integration tests
│   └── test_confirmation_integration.py # Full confirmation flows
├── factories/               # Factory pattern implementations
│   ├── __init__.py         # Factory exports
│   ├── fastapi_factory.py  # FastAPI app/client factories
│   ├── README.md           # Factory documentation
│   └── MIGRATION_GUIDE.md  # Factory migration guide
├── fixtures/               # Organized mock responses and utilities
│   ├── auth/               # Authentication fixtures and utilities
│   ├── claude_sdk/         # Claude SDK mocking
│   ├── external_apis/      # External API mocking
│   ├── proxy_service/      # Proxy service mocking
│   ├── responses.json      # Legacy mock data (still works)
│   ├── README.md           # Complete fixture documentation
│   └── MIGRATION_GUIDE.md  # Migration strategies
├── helpers/                # Test helper utilities
└── .gitignore              # Excludes coverage reports
```

## Writing Tests

### What to Mock (External Only)
- **External APIs**: Claude API responses (using `mock_external_anthropic_api`)
- **OAuth endpoints**: Token endpoints (using `mock_external_oauth_endpoints`)  
- **Docker subprocess calls**: Process execution mocking
- **Nothing else**: Keep mocking minimal and focused

### What NOT to Mock
- **Internal services**: Use dependency injection with `mock_internal_claude_sdk_service`
- **Adapters**: Test real adapter logic
- **Configuration**: Use test settings
- **Middleware**: Test real middleware behavior
- **Any internal components**: Only mock external boundaries

### New Mocking Strategy (Clear Separation)
- **Internal Mocks**: `mock_internal_claude_sdk_service` - AsyncMock for dependency injection
- **External Mocks**: `mock_external_anthropic_api` - HTTPXMock for HTTP interception
- **OAuth Mocks**: `mock_external_oauth_endpoints` - OAuth flow simulation

## Type Safety and Code Quality

**REQUIREMENT**: All test files MUST pass type checking and linting. This is not optional.

### Type Safety Requirements
1. **All test files MUST pass mypy type checking** - No `Any` types unless absolutely necessary
2. **All test files MUST pass ruff formatting and linting** - Code must be properly formatted
3. **Add proper type hints to all test functions and fixtures** - Include return types and parameter types
4. **Import necessary types** - Use `from typing import` for type annotations

### Required Type Annotations
- **Test functions**: Must have `-> None` return type annotation
- **Fixtures**: Must have proper return type hints
- **Parameters**: Must have type hints where not inferred from fixtures
- **Variables**: Add type hints for complex objects when not obvious

### Examples with Proper Typing

#### Basic Test Function with Types
```python
from typing import Any
import pytest
from fastapi.testclient import TestClient
from httpx_mock import HTTPXMock

def test_openai_endpoint(client: TestClient, mock_claude: HTTPXMock) -> None:
    """Test OpenAI-compatible endpoint"""
    response = client.post("/v1/chat/completions", json={
        "model": "claude-3-5-sonnet-20241022",
        "messages": [{"role": "user", "content": "Hello"}]
    })
    assert response.status_code == 200
    data: dict[str, Any] = response.json()
    assert "choices" in data
```

#### Fixture with Type Annotations
```python
from typing import Generator
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

@pytest.fixture
def app() -> FastAPI:
    """Create test FastAPI application"""
    from ccproxy.main import create_app
    return create_app()

@pytest.fixture
def client(app: FastAPI) -> Generator[TestClient, None, None]:
    """Create test client"""
    with TestClient(app) as test_client:
        yield test_client
```

#### Testing with Complex Types
```python
from typing import Any, Dict, List
from pathlib import Path
import pytest

def test_config_loading(tmp_path: Path) -> None:
    """Test configuration file loading"""
    config_file: Path = tmp_path / "config.toml"
    config_file.write_text("port = 8080")

    from ccproxy.config.settings import Settings
    settings: Settings = Settings(_config_file=config_file)
    assert settings.port == 8080
```

### Quality Checks Commands
```bash
# Type checking (MUST pass)
make typecheck
uv run mypy tests/

# Linting and formatting (MUST pass)
make lint
make format
uv run ruff check tests/
uv run ruff format tests/

# Run all quality checks
make pre-commit
```

### Common Type Annotations for Tests
- `TestClient` - FastAPI test client
- `HTTPXMock` - Mock for HTTP requests
- `Path` - File system paths
- `dict[str, Any]` - JSON response data
- `Generator[T, None, None]` - Fixture generators
- `-> None` - Test function return type

### Basic Test Pattern
```python
from fastapi.testclient import TestClient
from httpx_mock import HTTPXMock

def test_openai_endpoint(client: TestClient, mock_claude: HTTPXMock) -> None:
    """Test OpenAI-compatible endpoint"""
    response = client.post("/v1/chat/completions", json={
        "model": "claude-3-5-sonnet-20241022",
        "messages": [{"role": "user", "content": "Hello"}]
    })
    assert response.status_code == 200
    assert "choices" in response.json()
```

### Testing with Auth
```python
from fastapi.testclient import TestClient

def test_with_auth_token(client_with_auth: TestClient) -> None:
    """Test endpoint requiring authentication"""
    response = client_with_auth.post("/v1/messages",
        json={"messages": [{"role": "user", "content": "Hi"}]},
        headers={"Authorization": "Bearer test-token"}
    )
    assert response.status_code == 200
```

### Testing Streaming
```python
from fastapi.testclient import TestClient
from httpx_mock import HTTPXMock

def test_streaming_response(client: TestClient, mock_claude_stream: HTTPXMock) -> None:
    """Test SSE streaming"""
    with client.stream("POST", "/v1/chat/completions",
                      json={"stream": True, "model": "claude-3-5-sonnet-20241022",
                           "messages": [{"role": "user", "content": "Hello"}]}) as response:
        for line in response.iter_lines():
            assert line.startswith("data: ")
```

## Fixtures Architecture

### NEW: Factory Pattern (Recommended for New Tests)

#### Factory Fixtures
- `fastapi_app_factory` - Creates FastAPI apps with any configuration  
- `fastapi_client_factory` - Creates test clients with any configuration

#### Authentication Modes (Composable)
- `auth_mode_none` - No authentication required
- `auth_mode_bearer_token` - Bearer token without server config
- `auth_mode_configured_token` - Bearer token with server-configured token  
- `auth_mode_credentials` - OAuth credentials flow
- `auth_mode_credentials_with_fallback` - Credentials with bearer fallback

#### Auth Utilities
- `auth_settings_factory` - Creates settings for any auth mode
- `auth_headers_factory` - Generates headers for any auth mode
- `invalid_auth_headers_factory` - Creates invalid headers for testing
- `auth_test_utils` - Helper functions for auth response validation

#### Service Mocks (Clear Naming)
- `mock_internal_claude_sdk_service` - AsyncMock for dependency injection
- `mock_external_anthropic_api` - HTTPXMock for HTTP interception  
- `mock_external_oauth_endpoints` - OAuth endpoint mocking

#### Convenience Fixtures (Pre-configured)
- `client_no_auth` - No authentication required
- `client_bearer_auth` - Bearer token authentication
- `client_configured_auth` - Server-configured token auth
- `client_credentials_auth` - OAuth credentials authentication

### Legacy Fixtures (Backward Compatibility)

#### Core Fixtures (Still Work)
- `app()` - Test FastAPI application
- `client(app)` - Test client for API calls  
- `client_with_auth(app)` - Client with auth enabled

#### Response Fixtures (Still Work)
- `claude_responses()` - Standard Claude responses
- `mock_claude_stream()` - Streaming responses

#### Legacy Aliases (For Migration)
- `mock_claude_service` → `mock_internal_claude_sdk_service`
- `mock_claude` → `mock_external_anthropic_api`
- `mock_oauth` → `mock_external_oauth_endpoints`

## Test Markers
- `@pytest.mark.unit` - Fast unit tests (default)
- `@pytest.mark.real_api` - Tests using real APIs (slow)
- `@pytest.mark.docker` - Tests requiring Docker

## Best Practices

1. **Keep tests focused** - One test, one behavior
2. **Use descriptive names** - `test_what_when_expected`
3. **Minimal setup** - Use factories and fixtures, avoid duplication
4. **Real components** - Only mock external services (clear separation)
5. **Fast by default** - Real API tests are optional
6. **NEW: Use factory pattern** - For complex scenarios with multiple configurations
7. **NEW: Use composable auth** - Mix and match auth modes as needed
8. **NEW: Parametrized testing** - Test multiple scenarios in one test function

## Common Patterns

### NEW: Factory Pattern for Complex Scenarios
```python
from fastapi.testclient import TestClient

def test_complex_scenario(fastapi_client_factory, auth_mode_bearer_token,
                         mock_internal_claude_sdk_service) -> None:
    """Test authenticated endpoint with mocked service."""
    client = fastapi_client_factory.create_client(
        auth_mode=auth_mode_bearer_token,
        claude_service_mock=mock_internal_claude_sdk_service
    )
    response = client.post("/v1/messages", json={
        "model": "claude-3-5-sonnet-20241022",
        "messages": [{"role": "user", "content": "Hello"}]
    })
    assert response.status_code == 200
```

### NEW: Parametrized Testing (Multiple Scenarios)
```python
import pytest
from fastapi.testclient import TestClient

@pytest.mark.parametrize("auth_mode_fixture", [
    "auth_mode_none", "auth_mode_bearer_token", "auth_mode_configured_token"
])
def test_endpoint_all_auth_modes(request, auth_mode_fixture, fastapi_client_factory,
                                auth_headers_factory) -> None:
    """Test endpoint with different authentication modes."""
    auth_mode = request.getfixturevalue(auth_mode_fixture)
    client = fastapi_client_factory.create_client(auth_mode=auth_mode)

    headers = auth_headers_factory(auth_mode) if auth_mode else {}
    response = client.get("/api/models", headers=headers)
    assert response.status_code == 200
```

### NEW: Composable Authentication Testing
```python
from fastapi.testclient import TestClient

def test_auth_endpoint(client_bearer_auth: TestClient, auth_headers_factory,
                      auth_mode_bearer_token) -> None:
    """Test endpoint with bearer token authentication."""
    headers = auth_headers_factory(auth_mode_bearer_token)
    response = client_bearer_auth.post("/v1/messages",
        json={"messages": [{"role": "user", "content": "Hello"}]},
        headers=headers
    )
    assert response.status_code == 200
```

### Testing Error Cases (Updated)
```python
from typing import Any
from fastapi.testclient import TestClient

def test_invalid_model_error(fastapi_client_factory,
                           mock_internal_claude_sdk_service) -> None:
    """Test error handling with internal service mock."""
    # Configure mock to raise validation error
    from ccproxy.core.errors import ValidationError
    mock_internal_claude_sdk_service.create_completion.side_effect = \
        ValidationError("Invalid model specified")

    client = fastapi_client_factory.create_client(
        claude_service_mock=mock_internal_claude_sdk_service
    )
    response = client.post("/v1/messages", json={
        "model": "invalid-model",
        "messages": [{"role": "user", "content": "Hello"}]
    })
    assert response.status_code == 400
```

### Testing Metrics Collection
```python
from typing import Any
from fastapi.testclient import TestClient
from httpx_mock import HTTPXMock

def test_metrics_collected(client: TestClient, mock_claude: HTTPXMock, app) -> None:
    # Make request
    client.post("/v1/messages", json={
        "model": "claude-3-5-sonnet-20241022",
        "messages": [{"role": "user", "content": "Hello"}]
    })
    # Check metrics
    metrics: list[dict[str, Any]] = app.state.metrics_collector.get_metrics()
    assert len(metrics) > 0
```

### Testing with Temp Files
```python
from pathlib import Path
import pytest

def test_config_loading(tmp_path: Path) -> None:
    config_file: Path = tmp_path / "config.toml"
    config_file.write_text("port = 8080")

    from ccproxy.config.settings import Settings
    settings: Settings = Settings(_config_file=config_file)
    assert settings.port == 8080
```

## Running Tests

### Make Commands
```bash
make test              # Run all tests
make test-unit         # Fast tests only
make test-coverage     # With coverage report
make test-watch        # Auto-run on changes
```

### Direct pytest
```bash
pytest -v                          # Verbose output
pytest -k "test_auth"              # Run matching tests
pytest --lf                        # Run last failed
pytest -x                          # Stop on first failure
pytest --pdb                       # Debug on failure
```

## Debugging Tests

### Print Debugging
```python
from typing import Any
from fastapi.testclient import TestClient
import pytest

def test_something(client: TestClient, capsys: pytest.CaptureFixture[str]) -> None:
    response = client.post("/v1/messages", json={
        "model": "claude-3-5-sonnet-20241022",
        "messages": [{"role": "user", "content": "Hello"}]
    })
    data: dict[str, Any] = response.json()
    print(f"Response: {data}")  # Will show in pytest output
    captured = capsys.readouterr()
```

### Interactive Debugging
```python
from fastapi.testclient import TestClient

def test_something(client: TestClient) -> None:
    response = client.post("/v1/messages", json={
        "model": "claude-3-5-sonnet-20241022",
        "messages": [{"role": "user", "content": "Hello"}]
    })
    import pdb; pdb.set_trace()  # Debugger breakpoint
```

## For New Developers

1. **Start here**: Read this file and `tests/conftest.py`
2. **Run tests**: `make test` to ensure everything works
3. **Add new test**: Copy existing test pattern, modify as needed
4. **Mock external only**: Don't mock internal components
5. **Ask questions**: Tests should be obvious, if not, improve them

## Factory Pattern Migration

### Quick Migration Guide

**All existing tests continue working unchanged** - Migration is optional but recommended for new tests.

See [`FIXTURE_MIGRATION_GUIDE.md`](./FIXTURE_MIGRATION_GUIDE.md) for comprehensive migration examples.

### Key Changes Summary

#### Before (Old Pattern)
```python
# Limited combinations, fixture explosion
def test_auth(client_with_auth: TestClient) -> None:
    response = client_with_auth.post("/v1/messages")
```

#### After (New Pattern - Recommended)
```python  
# Infinite combinations, composable
def test_auth(fastapi_client_factory, auth_mode_bearer_token,
              auth_headers_factory) -> None:
    client = fastapi_client_factory.create_client(auth_mode=auth_mode_bearer_token)
    headers = auth_headers_factory(auth_mode_bearer_token)
    response = client.post("/v1/messages", headers=headers)
```

#### Benefits of Migration
- **Scalability**: Linear vs exponential fixture growth
- **Clarity**: Clear naming (`mock_internal_claude_sdk_service` vs `mock_claude_service`)  
- **Composability**: Test any combination of features
- **Type Safety**: Full type annotations and mypy compliance
- **No Test Skips**: Proper configurations for all auth modes

## For LLMs/AI Assistants

When writing tests for this project:

### Required (Unchanged)
1. **MUST include proper type hints** - All test functions need `-> None` return type
2. **MUST pass mypy and ruff checks** - Type safety and formatting are required  
3. Keep tests simple and focused
4. Follow the naming convention: `test_what_when_expected()`
5. Import necessary types: `TestClient`, `HTTPXMock`, `Path`, etc.

### Recommended (New)
6. **Use factory pattern** - For complex scenarios: `fastapi_client_factory.create_client()`
7. **Use composable auth** - Auth modes: `auth_mode_bearer_token`, `auth_mode_none`, etc.
8. **Use clear mock naming** - `mock_internal_claude_sdk_service`, `mock_external_anthropic_api`
9. **Use parametrized testing** - Test multiple scenarios in one function
10. **Prefer convenience fixtures** - `client_bearer_auth`, `client_no_auth` for simple cases

### Legacy Support (Backward Compatibility)
- All existing fixtures still work: `client`, `client_with_auth`, `mock_claude_service`
- Use existing patterns in `tests/` as reference
- Only mock external HTTP calls using `httpx_mock`
- Use fixtures from `conftest.py`, don't create new combinatorial ones

**Type Safety Checklist:**
- [ ] All test functions have `-> None` return type
- [ ] All parameters have type hints (especially fixtures)
- [ ] Complex variables have explicit type annotations
- [ ] Proper imports from `typing` module
- [ ] Code passes `make typecheck` and `make lint`

**Factory Pattern Checklist:**
- [ ] Use `fastapi_client_factory` for complex test scenarios
- [ ] Use auth modes (`auth_mode_bearer_token`) instead of manual auth setup
- [ ] Use clear service mock names (`mock_internal_claude_sdk_service`)
- [ ] Consider parametrized testing for multiple scenarios
- [ ] Use convenience fixtures (`client_bearer_auth`) for simple cases

Remember: **Simple tests that actually test real behavior > Complex tests with lots of mocks.**

**Migration is optional** - all existing tests continue working. Use new patterns for better maintainability and testing capabilities.
