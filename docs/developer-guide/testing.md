# Testing

## Overview

The Claude Code Proxy API includes comprehensive test coverage using pytest with different test categories.

## Test Categories

Tests are organized using pytest markers:

- `unit` - Fast unit tests (< 1s each)
- `integration` - Integration tests (< 30s each)
- `slow` - Slow tests (> 30s each)
- `docker` - Docker-dependent tests
- `network` - Network-access tests

## Running Tests

### All Tests
```bash
pytest
```

### Specific Test Categories
```bash
pytest -m unit        # Fast unit tests
pytest -m integration # Integration tests
pytest -m slow        # Slow tests
pytest -m docker      # Docker tests
pytest -m network     # Network tests
```

### With Coverage
```bash
pytest --cov=claude_code_proxy --cov-report=html
```

### Specific Test Files
```bash
pytest tests/test_api.py
pytest tests/test_services.py
pytest tests/test_streaming.py
```

## Test Structure

```
tests/
├── test_api.py           # API endpoint tests
├── test_services.py      # Service layer tests
├── test_streaming.py     # Streaming functionality tests
├── test_config.py        # Configuration tests
└── conftest.py          # Test fixtures
```

## Writing Tests

### Unit Test Example
```python
import pytest
from claude_code_proxy.services.translator import OpenAITranslator

def test_translate_request():
    translator = OpenAITranslator()
    openai_request = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}]
    }
    anthropic_request = translator.translate_request(openai_request)
    assert anthropic_request["model"] == "claude-3-5-sonnet-20241022"
```

### Integration Test Example
```python
import pytest
from fastapi.testclient import TestClient
from claude_code_proxy.main import create_app

@pytest.mark.integration
def test_chat_completion_endpoint():
    app = create_app()
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello"}]
        }
    )
    assert response.status_code == 200
```

## Test Configuration

Tests use environment variables for configuration:

```bash
export CLAUDE_CLI_PATH="/path/to/claude"
export BEARER_TOKEN="test-token"
```
