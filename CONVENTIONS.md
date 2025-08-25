# CCProxy Coding Conventions

## 1. Guiding Principles

Our primary goal is to build a robust, maintainable, scalable, and secure CCProxy API Server. These conventions are rooted in the following principles:

* **Clarity over Cleverness:** Code should be easy to read and understand
* **Explicit over Implicit:** Be clear about intentions and dependencies
* **Consistency:** Follow established patterns within the project
* **Single Responsibility Principle:** Each module, class, or function should have one clear purpose
* **Loose Coupling, High Cohesion:** Modules should be independent but related components within a module should be grouped
* **Testability:** Write code that is inherently easy to unit and integration test
* **Pythonic:** Embrace PEP 8 and the Zen of Python (`import this`)

## 2. General Python Conventions

* **PEP 8 Compliance:** Adhere strictly to PEP 8
  * Use `ruff format` for auto-formatting to ensure consistent style
  * Line length limit is **88 characters** (ruff's default)
* **Python Version:** Target **Python 3.11+**. Utilize modern features like union types (`X | Y`)
* **No Mutable Default Arguments:** Avoid using mutable objects as default arguments
  * **Bad:** `def foo(items=[])`
  * **Good:** `def foo(items: list | None = None): if items is None: items = []`

## 3. Naming Conventions

* **Packages/Directories:** `snake_case` (e.g., `api`, `claude_sdk`, `auth`)
* **Modules:** `snake_case` (e.g., `manager.py`, `client.py`)
* **Classes:** `CamelCase` (e.g., `ProxyService`, `OpenAIAdapter`)
  * **Abstract Base Classes:** Suffix with `ABC` or `Protocol`
  * **Pydantic Models:** `CamelCase` (e.g., `MessageCreateParams`)
* **Functions/Methods/Variables:** `snake_case` (e.g., `handle_request`, `get_access_token`)
* **Constants:** `UPPER_SNAKE_CASE` (e.g., `DEFAULT_PORT`, `API_VERSION`)
* **Private Members:** `_single_leading_underscore` for internal use

## 4. Imports

* **Ordering:** Standard library → Third-party → First-party → Relative
* **Absolute Imports Preferred:** Use absolute imports for modules within `ccproxy`
  * **Good:** `from ccproxy.auth.manager import AuthManager`
* **Relative Imports:** Use for modules within the same package
  * **Good (inside `plugins/claude_api/`):** `from .models import ClaudeModel`
* **`__all__` in `__init__.py`:** Define to explicitly expose public API

## 5. Typing

Type hints are mandatory for clarity and maintainability:

* **All Function Signatures:** Type-hint all parameters and return values
* **Class Attributes:** Use type hints, especially for Pydantic models
* **Union Types:** Use `Type | None` for optional values (Python 3.11+)
* **Type Aliases:** Define in `core/types.py` for complex types

## 6. Plugin Architecture

### Plugin Structure
Each plugin must follow the delegation pattern:

```python
plugins/
├── plugin_name/
│   ├── __init__.py
│   ├── adapter.py          # Main plugin interface
│   ├── plugin.py           # Plugin declaration
│   ├── transformers/       # Request/response transformation
│   │   ├── request.py
│   │   └── response.py
│   ├── detection_service.py # Provider capability detection
│   ├── format_adapter.py   # Protocol conversion (if needed)
│   └── auth/               # Authentication (if needed)
│       └── manager.py
```

### Delegation Pattern
Adapters must delegate to ProxyService:

```python
class ProviderAdapter(BaseAdapter):
    async def handle_request(self, request, endpoint, method):
        context = self._build_provider_context()
        return await self.proxy_service.handle_provider_request(
            request, endpoint, method, context
        )
```

## 7. Error Handling

* **Custom Exceptions:** Inherit from `ccproxy.core.errors.CCProxyError`
* **Catch Specific Exceptions:** Never use bare `except:`
* **Chain Exceptions:** Use `raise NewError(...) from original`
* **FastAPI HTTPException:** Use in routes with appropriate status codes

## 8. Asynchronous Programming

* **`async`/`await`:** Use consistently for all I/O operations
* **Libraries:** Prefer `httpx` for HTTP, `asyncio` for concurrency
* **No Blocking Code:** Never use blocking I/O in async functions

## 9. Testing

* **Framework:** `pytest` with `pytest-asyncio`
* **Structure:** Tests in `tests/` with clear organization:
  * `tests/unit/` - Fast, isolated unit tests
  * `tests/integration/` - Component interaction tests
* **Markers:** Use `@pytest.mark.unit`, `@pytest.mark.integration`
* **Fixtures:** Share in `conftest.py`
* **Coverage:** High coverage on critical paths (auth, API endpoints, plugins)

## 10. Configuration

* **Pydantic Settings:** All config in `config/settings.py`
* **Environment Variables:** Use `__` for nesting (e.g., `SERVER__LOG_LEVEL`)
* **Priority:** CLI args → Environment → TOML files → Defaults

## 11. Security

* **Input Validation:** All API inputs validated with Pydantic
* **No Secrets in Code:** Use environment variables
* **Authentication:** Enforce via middleware
* **CORS:** Configure properly in production

## 12. Tooling

Core tools enforced via pre-commit and CI:

* **Package Manager:** `uv` (via Makefile only)
* **Formatter:** `ruff format`
* **Linter:** `ruff check`
* **Type Checker:** `mypy`
* **Test Runner:** `pytest`

## 13. Development Workflow

### Required Before Commits
```bash
make pre-commit  # Comprehensive checks + auto-fixes
make test        # Run tests with coverage
```

### Key Makefile Targets

| Category | Target | Description |
|----------|--------|-------------|
| **Setup** | `make setup` | Complete dev environment setup |
| **Quality** | `make pre-commit` | All checks with auto-fixes |
| | `make check` | Lint + typecheck + format check |
| | `make format` | Format code |
| | `make lint` | Linting only |
| | `make typecheck` | Type checking |
| **Testing** | `make test` | Full test suite with coverage |
| | `make test-unit` | Fast unit tests only |
| | `make test-integration` | Integration tests |
| **CI** | `make ci` | Full CI pipeline |
| **Build** | `make build` | Build Python package |
| | `make docker-build` | Build Docker image |
| **Dev** | `make dev` | Start dev server with debug logging |

## 14. Documentation

* **Docstrings:** Required for all public APIs (Google style)
* **Comments:** Explain *why*, not *what*
* **TODO/FIXME:** Use consistently with explanations

## 15. Git Workflow

* **Commits:** Follow Conventional Commits (feat:, fix:, docs:, etc.)
* **Branches:** Use feature branches (`feature/`, `fix/`, `docs/`)
* **No `git add .`:** Only stage specific files

## 16. Project-Specific Patterns

### Provider Context Pattern
```python
context = ProviderContext(
    provider_name="...",
    target_base_url="...",
    request_transformer=...,
    response_transformer=...,
    auth_manager=...,
    supports_streaming=True
)
```

### Transformer Pattern
```python
class RequestTransformer:
    def transform_headers(self, headers, **kwargs): ...
    def transform_body(self, body): ...  # Often passthrough
```

### Environment Variables
* Config: `SERVER__LOG_LEVEL=debug`
* Features: `CCPROXY_VERBOSE_API=true`
* Logging: `CCPROXY_REQUEST_LOG_DIR=/tmp/ccproxy/request`
