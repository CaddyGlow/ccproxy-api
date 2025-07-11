# Simplified Testing Guide for CCProxy

## Philosophy
Keep it simple. Test what matters, mock what's external, don't overthink it.

## Quick Start
```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_api.py

# Run with coverage
make test-coverage

# Run with real APIs (optional, slow)
pytest -m real_api
```

## Test Structure
```
tests/
├── conftest.py              # Shared fixtures (minimal)
├── test_api.py              # All API endpoint tests
├── test_auth.py             # Auth + OAuth2 together
├── test_streaming.py        # Streaming functionality
├── test_metrics.py          # Metrics collection
├── test_adapters.py         # OpenAI↔Anthropic conversion
└── fixtures/                # Mock responses
    └── responses.json       # All mock data in one file
```

## Writing Tests

### What to Mock (External Only)
- Claude API responses (using httpx_mock)
- OAuth token endpoints
- Docker subprocess calls
- Nothing else

### What NOT to Mock
- Internal services
- Adapters
- Configuration
- Middleware
- Any internal components

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

## Fixtures (from conftest.py)

### Core Fixtures
- `app()` - Test FastAPI application
- `client(app)` - Test client for API calls
- `mock_claude(httpx_mock)` - Mocked Claude API
- `client_with_auth(app)` - Client with auth enabled

### Response Fixtures
- `claude_responses()` - Standard Claude responses
- `mock_claude_stream()` - Streaming responses

## Test Markers
- `@pytest.mark.unit` - Fast unit tests (default)
- `@pytest.mark.real_api` - Tests using real APIs (slow)
- `@pytest.mark.docker` - Tests requiring Docker

## Best Practices

1. **Keep tests focused** - One test, one behavior
2. **Use descriptive names** - `test_what_when_expected`
3. **Minimal setup** - Use fixtures, avoid duplication
4. **Real components** - Only mock external services
5. **Fast by default** - Real API tests are optional

## Common Patterns

### Testing Error Cases
```python
from typing import Any
from fastapi.testclient import TestClient
from httpx_mock import HTTPXMock

def test_invalid_model_error(client: TestClient, mock_claude: HTTPXMock) -> None:
    mock_claude.add_response(status_code=400, json={"error": "Invalid model"})
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

## For LLMs/AI Assistants

When writing tests for this project:
1. **MUST include proper type hints** - All test functions need `-> None` return type
2. **MUST pass mypy and ruff checks** - Type safety and formatting are required
3. Use the existing test patterns in `tests/`
4. Only mock external HTTP calls using `httpx_mock`
5. Use fixtures from `conftest.py`, don't create new ones
6. Keep tests simple and focused
7. Follow the naming convention: `test_what_when_expected()`
8. Import necessary types: `TestClient`, `HTTPXMock`, `Path`, etc.

**Type Safety Checklist:**
- [ ] All test functions have `-> None` return type
- [ ] All parameters have type hints (especially fixtures)
- [ ] Complex variables have explicit type annotations
- [ ] Proper imports from `typing` module
- [ ] Code passes `make typecheck` and `make lint`

Remember: Simple tests that actually test real behavior > Complex tests with lots of mocks.
