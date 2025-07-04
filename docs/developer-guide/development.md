# Development Guide

## Development Environment Setup

### Prerequisites

- **Python 3.11+** (3.12 recommended)
- **uv** (preferred package manager) or pip
- **Git** for version control
- **Docker** (optional, for containerized development)
- **Node.js** (if using Claude CLI via npm)

### Quick Setup with devenv (Recommended)

If you have Nix and devenv installed:

```bash
# Clone the repository
git clone https://github.com/your-org/claude-proxy.git
cd claude-proxy

# Enter development environment
devenv shell

# Dependencies will be automatically installed
```

### Manual Setup with uv

```bash
# Clone the repository
git clone https://github.com/your-org/claude-proxy.git
cd claude-proxy

# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate
```

### Manual Setup with pip

```bash
# Clone the repository
git clone https://github.com/your-org/claude-proxy.git
cd claude-proxy

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
pip install -e .[dev]
```

## Project Structure

```
claude-proxy/
├── claude_code_proxy/          # Main application package
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── cli.py                  # CLI interface
│   ├── exceptions.py           # Custom exceptions
│   ├── api/                    # API endpoints
│   │   ├── v1/                 # Anthropic-compatible endpoints
│   │   └── openai/             # OpenAI-compatible endpoints
│   ├── config/                 # Configuration management
│   ├── models/                 # Data models
│   ├── services/               # Business logic
│   └── utils/                  # Utility functions
├── tests/                      # Test suite
├── docs/                       # Documentation
├── examples/                   # Usage examples
├── pyproject.toml             # Project configuration
├── uv.lock                    # Dependency lock file
├── CLAUDE.md                  # Claude Code instructions
└── README.md                  # Project documentation
```

## Development Workflow

### Initial Setup

1. **Check repository state**:
```bash
git status  # Should be clean with no staged files
```

2. **Install pre-commit hooks** (if available):
```bash
pre-commit install
```

3. **Verify installation**:
```bash
# Test CLI installation
ccproxy --version

# Test application startup
python -m claude_code_proxy.main
```

### Code Quality Requirements

The project enforces strict code quality standards:

#### Type Checking with mypy

```bash
# Run type checking
mypy .

# Type checking is configured in pyproject.toml with strict mode
# All code must pass type checking before committing
```

#### Code Formatting with ruff

```bash
# Format code
ruff format .

# Check formatting
ruff format --check .

# The project uses 88-character line length
```

#### Linting with ruff

```bash
# Run linting
ruff check .

# Fix automatically fixable issues
ruff check --fix .

# Linting rules are configured in pyproject.toml
```

#### Run All Quality Checks

```bash
# Recommended workflow
ruff format . && ruff check . && mypy .

# Or use the Makefile target (if available)
make quality
```

### Testing

#### Test Structure

```
tests/
├── unit/                      # Fast unit tests (<1s each)
├── integration/               # Integration tests (<30s each)
├── slow/                      # Slow tests (>30s each)
├── fixtures/                  # Test fixtures
└── conftest.py               # Pytest configuration
```

#### Running Tests

```bash
# Run all tests
pytest

# Run specific test types
pytest -m unit        # Fast unit tests only
pytest -m integration # Integration tests only
pytest -m slow        # Slow tests only

# Run with coverage
pytest --cov=claude_code_proxy

# Run specific test file
pytest tests/test_api.py

# Run specific test function
pytest tests/test_api.py::test_chat_completion
```

#### Test Markers

The project uses pytest markers to categorize tests:

- `unit`: Fast unit tests that don't require external dependencies
- `integration`: Integration tests that may use Docker or external services
- `slow`: Slow tests that take significant time or resources
- `docker`: Tests that require Docker to be available
- `network`: Tests that require network access
- `regression`: Regression tests for specific bug fixes
- `smoke`: Basic smoke tests for critical functionality
- `asyncio`: Tests that use asyncio

#### Writing Tests

```python
import pytest
from claude_code_proxy.models.requests import ChatCompletionRequest

@pytest.mark.unit
def test_chat_completion_request_validation():
    """Test request validation."""
    request = ChatCompletionRequest(
        model="claude-3-5-sonnet-20241022",
        messages=[
            {"role": "user", "content": "Hello"}
        ]
    )
    assert request.model == "claude-3-5-sonnet-20241022"

@pytest.mark.integration
async def test_api_endpoint(client):
    """Test API endpoint integration."""
    response = await client.post(
        "/v1/chat/completions",
        json={
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello"}]
        }
    )
    assert response.status_code == 200
```

### Development Server

#### Running the Development Server

```bash
# Using uvicorn directly (recommended for development)
uvicorn claude_code_proxy.main:app --reload --host 0.0.0.0 --port 8000

# Using the CLI
ccproxy run --reload --host 0.0.0.0 --port 8000

# Using Python module
python -m uvicorn claude_code_proxy.main:app --reload
```

#### Development Configuration

Create `.env.dev` file:

```bash
# Development environment variables
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=DEBUG
RELOAD=true
WORKERS=1

# Claude configuration
CLAUDE_CLI_PATH=/usr/local/bin/claude  # Optional
```

#### Hot Reloading

The development server supports hot reloading:

```bash
# Enable auto-reload for file changes
uvicorn claude_code_proxy.main:app --reload

# Watch specific directories
uvicorn claude_code_proxy.main:app --reload --reload-dir ./claude_code_proxy
```

## Debugging

### Debugging with Python Debugger

```python
# Add breakpoint in code
import pdb; pdb.set_trace()

# Or use the built-in breakpoint() function (Python 3.7+)
breakpoint()
```

### Debugging with VS Code

Create `.vscode/launch.json`:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug FastAPI",
            "type": "python",
            "request": "launch",
            "module": "uvicorn",
            "args": [
                "claude_code_proxy.main:app",
                "--reload",
                "--host", "0.0.0.0",
                "--port", "8000"
            ],
            "console": "integratedTerminal",
            "env": {
                "LOG_LEVEL": "DEBUG"
            }
        },
        {
            "name": "Debug Tests",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": ["tests/", "-v"],
            "console": "integratedTerminal"
        }
    ]
}
```

### Logging Configuration

#### Development Logging

```python
import logging

# Configure development logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Use structured logging for better debugging
logger = logging.getLogger(__name__)
logger.info("API request received", extra={
    "method": "POST",
    "path": "/v1/chat/completions",
    "client_ip": "127.0.0.1"
})
```

#### Log Analysis

```bash
# View application logs
tail -f logs/app.log

# Search for errors
grep ERROR logs/app.log

# Filter by log level
grep -E "(ERROR|WARNING)" logs/app.log

# Monitor logs in real-time
tail -f logs/app.log | grep -E "(ERROR|WARNING)"
```

## Development Tools

### IDE Configuration

#### VS Code Settings

Create `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "./.venv/bin/python",
    "python.formatting.provider": "none",
    "python.linting.enabled": false,
    "ruff.enable": true,
    "ruff.organizeImports": true,
    "mypy.enabled": true,
    "files.associations": {
        "*.py": "python"
    },
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

#### PyCharm Configuration

1. Set Python interpreter to `.venv/bin/python`
2. Enable ruff for formatting and linting
3. Enable mypy for type checking
4. Configure test runner to use pytest

### Git Hooks

#### Pre-commit Configuration

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.0.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
```

## Contributing Guidelines

### Code Style and Conventions

#### Python Code Style

- **Line Length**: 88 characters (configured in ruff)
- **Imports**: Use absolute imports, organize with ruff
- **Type Hints**: Required for all functions and methods
- **Docstrings**: Use Google-style docstrings
- **Naming**: Follow PEP 8 conventions

#### Example Code Style

```python
"""Module docstring describing the module purpose."""

from typing import Any, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from claude_code_proxy.config.settings import get_settings


class ChatRequest(BaseModel):
    """Chat completion request model."""
    
    model: str
    messages: list[dict[str, Any]]
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.7


async def create_chat_completion(
    request: ChatRequest,
    settings: Optional[Any] = None,
) -> dict[str, Any]:
    """
    Create a chat completion.
    
    Args:
        request: The chat completion request
        settings: Optional settings override
        
    Returns:
        Chat completion response
        
    Raises:
        HTTPException: If request validation fails
    """
    if settings is None:
        settings = get_settings()
        
    # Implementation here
    return {"response": "example"}
```

### Git Workflow

#### Branch Naming

- **Feature branches**: `feature/description`
- **Bug fixes**: `fix/description`
- **Documentation**: `docs/description`
- **Refactoring**: `refactor/description`

#### Commit Messages

Follow conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Examples:
```
feat(api): add OpenAI-compatible chat endpoint
fix(streaming): handle connection errors gracefully
docs(readme): update installation instructions
refactor(models): simplify request validation
```

#### Pull Request Process

1. **Create feature branch** from `main`
2. **Make changes** following code quality standards
3. **Run all quality checks** (tests, linting, type checking)
4. **Create pull request** with clear description
5. **Address review feedback**
6. **Merge** after approval

### Code Review Checklist

#### Before Submitting PR

- [ ] All tests pass (`pytest`)
- [ ] Type checking passes (`mypy .`)
- [ ] Linting passes (`ruff check .`)
- [ ] Code is formatted (`ruff format .`)
- [ ] Documentation is updated
- [ ] No sensitive information in code

#### Code Review Focus

- [ ] Code correctness and logic
- [ ] Error handling and edge cases
- [ ] Performance implications
- [ ] Security considerations
- [ ] API design and usability
- [ ] Test coverage and quality

## Performance Optimization

### Development Performance Tips

#### Async Best Practices

```python
# Good: Use async/await consistently
async def fetch_data():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.example.com")
        return response.json()

# Avoid: Blocking calls in async functions
async def bad_fetch_data():
    response = requests.get("https://api.example.com")  # Blocking!
    return response.json()
```

#### Memory Management

```python
# Use generators for large datasets
def process_large_dataset():
    for item in large_dataset:
        yield process_item(item)

# Close resources properly
async def with_resources():
    async with httpx.AsyncClient() as client:
        # Client will be properly closed
        pass
```

### Profiling and Monitoring

#### Profile CPU Usage

```python
import cProfile
import pstats

# Profile a function
profiler = cProfile.Profile()
profiler.enable()

# Your code here
result = your_function()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative').print_stats(10)
```

#### Memory Profiling

```bash
# Install memory profiler
pip install memory-profiler

# Profile memory usage
python -m memory_profiler your_script.py
```

#### Performance Testing

```python
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def performance_test():
    """Test API performance under load."""
    start_time = time.time()
    
    # Simulate concurrent requests
    tasks = []
    for _ in range(100):
        task = asyncio.create_task(make_api_request())
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    
    end_time = time.time()
    print(f"Processed {len(results)} requests in {end_time - start_time:.2f}s")
```

## Troubleshooting Common Issues

### Development Environment Issues

#### Virtual Environment Problems

```bash
# Reset virtual environment
rm -rf .venv
uv sync

# Check Python version
python --version  # Should be 3.11+

# Verify installation
pip list | grep claude-code-proxy
```

#### Import Errors

```bash
# Install in development mode
pip install -e .

# Check Python path
python -c "import sys; print(sys.path)"

# Verify package installation
python -c "import claude_code_proxy; print('OK')"
```

#### Port Conflicts

```bash
# Check what's using port 8000
lsof -i :8000
netstat -tulpn | grep :8000

# Use different port
uvicorn claude_code_proxy.main:app --port 8001
```

### Testing Issues

#### Test Database/Dependencies

```bash
# Clean test environment
pytest --cache-clear

# Run tests with verbose output
pytest -v -s

# Run specific failing test
pytest tests/test_specific.py::test_function -v
```

#### Async Test Issues

```python
# Ensure proper async test setup
import pytest
import asyncio

@pytest.mark.asyncio
async def test_async_function():
    result = await async_function()
    assert result is not None
```

### Docker Development Issues

#### Container Build Problems

```bash
# Build with no cache
docker build --no-cache -t claude-proxy .

# Check build logs
docker build -t claude-proxy . 2>&1 | tee build.log

# Inspect layers
docker history claude-proxy
```

#### Volume Mount Issues

```bash
# Check file permissions
ls -la ~/.config/claude

# Fix permissions
chmod 644 ~/.config/claude/*
```

## Advanced Development Topics

### Custom Middleware

```python
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

class CustomMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        response = await call_next(request)
        
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
```

### Custom Exception Handlers

```python
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

@app.exception_handler(CustomException)
async def custom_exception_handler(request: Request, exc: CustomException):
    return JSONResponse(
        status_code=400,
        content={"error": {"type": "custom_error", "message": str(exc)}}
    )
```

### Background Tasks

```python
from fastapi import BackgroundTasks

async def send_notification(message: str):
    # Background task implementation
    pass

@app.post("/endpoint")
async def endpoint(background_tasks: BackgroundTasks):
    background_tasks.add_task(send_notification, "Task completed")
    return {"status": "success"}
```

### WebSocket Support (Future Feature)

```python
from fastapi import WebSocket

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Echo: {data}")
    except WebSocketDisconnect:
        pass
```