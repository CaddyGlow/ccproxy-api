# `ccproxy` Coding Conventions

## 1\. Guiding Principles

Our primary goal is to build a robust, maintainable, scalable, and secure Claude Code Proxy API Server. These conventions are rooted in the following principles:

  * **Clarity over Cleverness:** Code should be easy to read and understand, even by someone new to the project.
  * **Explicit over Implicit:** Be clear about intentions and dependencies.
  * **Consistency:** Follow established patterns within the project.
  * **Single Responsibility Principle (SRP):** Each module, class, or function should have one clear purpose.
  * **Loose Coupling, High Cohesion:** Modules should be independent but related components within a module should be grouped.
  * **Testability:** Write code that is inherently easy to unit and integrate test.
  * **Pythonic:** Embrace PEP 8 and the Zen of Python (`import this`).

## 2\. General Python Conventions

  * **PEP 8 Compliance:** Adhere strictly to PEP 8 – The Style Guide for Python Code.
      * Use `Black` for auto-formatting to ensure consistent style.
      * Line length limit is **88 characters** (Black's default).
  * **Python Version:** Target **Python 3.10+**. Utilize features like union types (`X | Y`) where applicable.
  * **No Mutable Default Arguments:** Avoid using mutable objects (lists, dicts, sets) as default arguments in function definitions.
      * **Bad:** `def foo(items=[])`
      * **Good:** `def foo(items: Optional[List] = None): if items is None: items = []`

## 3\. Naming Conventions

Consistency in naming is crucial for navigability within our domain-driven structure.

  * **Packages/Directories:** `snake_case` (e.g., `api_server`, `claude_sdk`, `auth/oauth`).
  * **Modules (.py files):** `snake_case` (e.g., `manager.py`, `client.py`, `interfaces.py`).
  * **Classes:** `CamelCase` (e.g., `ProxyService`, `OpenAIAdapter`, `CredentialsManager`).
      * **Abstract Base Classes (ABCs) / Protocols:** Suffix with `ABC` or `Protocol` respectively (e.g., `HTTPClientABC`, `RequestTransformerProtocol`).
      * **Pydantic Models:** `CamelCase` (e.g., `MessageCreateParams`, `OpenAIChatCompletionRequest`).
  * **Functions, Methods, Variables:** `snake_case` (e.g., `handle_request`, `get_access_token`, `max_tokens`).
  * **Constants (Global):** `UPPER_SNAKE_CASE` (e.g., `DEFAULT_PORT`, `API_VERSION`).
  * **Private/Internal Members:**
      * **`_single_leading_underscore`:** For internal use within a module or class. Do not import or access these directly from outside their defined scope.
      * **`__double_leading_underscore` (Name Mangling):** Reserve this for preventing name clashes in inheritance hierarchies, rarely used.

## 4\. Imports

Imports should be clean, organized, and explicit.

  * **Ordering:** Use `isort` for consistent import ordering. General order:
    1.  Standard library imports.
    2.  Third-party library imports.
    3.  First-party (`ccproxy` project) imports.
    4.  Relative imports.
  * **Absolute Imports Preferred:** Use absolute imports for modules within the `ccproxy` project whenever possible, especially when crossing domain boundaries.
      * **Good:** `from ccproxy.auth.manager import AuthManager`
      * **Avoid (if not within the same sub-domain):** `from ..auth.manager import AuthManager`
  * **Relative Imports for Siblings:** Use relative imports for modules within the same logical sub-package/directory.
      * **Good (inside `adapters/openai/`):** `from .models import OpenAIModel`
  * **`__all__` in `__init__.py`:** Each package's `__init__.py` file **must** define an `__all__` list to explicitly expose its public API. This controls what `from package import *` imports and guides explicit imports.
      * **Example (`ccproxy/auth/__init__.py`):**
        ```python
        from .manager import AuthManager
        from .models import Credentials
        from .storage.base import TokenStorage

        __all__ = ["AuthManager", "Credentials", "TokenStorage"]
        ```
  * **Minimize Imports:** Only import what you need. Avoid `from module import *`.

## 5\. Typing

Type hints are mandatory for clarity, maintainability, and static analysis.

  * **All Function Signatures:** All function parameters and return values must be type-hinted.
  * **Class Attributes:** Use type hints for class attributes, especially Pydantic models.
  * **`from __future__ import annotations`:** Use this at the top of every module to enable postponed evaluation of type annotations, especially useful for forward references and preventing circular import issues.
  * **`Optional` Types:** Use `Type | None` for optional values (Python 3.10+) or `Optional[Type]` (from `typing`) for older versions/clarity.
  * **`Annotated` (Pydantic v2):** Use `Annotated` for `Field` and other Pydantic-specific metadata.
      * **Example:** `model: Annotated[str, Field(description="Model ID")]`
  * **Generics and Protocols:** Use `TypeVar`, `Generic`, `Protocol` (from `typing`) in `core/interfaces.py` and other modules where abstract types are defined.
  * **Type Aliases:** Use `TypeAlias` (from `typing`) for complex type hints in `core/types.py` or domain-specific `models.py` files to improve readability.

## 6\. Docstrings and Comments

Code should be self-documenting first, then supplemented by comments and docstrings.

  * **Docstrings:**
      * Every **public module, class, method, and function** must have a docstring.
      * Use **Google Style Docstrings** or **Sphinx Style** consistently throughout the project.
      * Describe the purpose, arguments, return values, and any exceptions raised.
  * **Comments:**
      * Explain *why* a particular piece of code exists or was chosen, not *what* it does (which should be clear from the code itself).
      * Avoid redundant comments that simply re-state the code.
      * Use comments for complex algorithms, workarounds, or non-obvious logic.
  * **TODO/FIXME/HACK:** Use these markers consistently for areas needing attention. Explain briefly what needs to be done.
      * `# TODO: Implement rate limiting here.`
      * `# FIXME: This logic has a known race condition under high load.`
      * `# HACK: Temporary workaround for issue #123.`

## 7\. Error Handling

Define clear error boundaries and handle exceptions gracefully.

  * **Custom Exceptions:**
      * All custom exceptions **must** inherit from `core.errors.ClaudeProxyError` or its more specific sub-classes defined in `core/errors.py`.
      * Domain-specific exceptions should be defined within their respective domain's `exceptions.py` module (e.g., `auth/exceptions.py`, `docker/exceptions.py`).
  * **Catch Specific Exceptions:** Always catch specific exception types, not bare `except Exception:`.
  * **Propagate with Context:** When re-raising or wrapping exceptions, include the original exception as the cause using `raise NewError(...) from OriginalError`.
  * **FastAPI `HTTPException`:** In API routes, raise `fastapi.HTTPException` with appropriate `status_code` and `detail` (which should be a dictionary conforming to our API error models). Internal services should raise custom exceptions, and the `api/middleware/errors.py` should convert them to `HTTPException`.

## 8\. Asynchronous Programming

Adhere to modern `asyncio` patterns.

  * **`async` / `await`:** Use `async def` and `await` consistently for all asynchronous operations (I/O-bound tasks).
  * **Asynchronous Libraries:** Prefer `httpx` for HTTP requests, `anyio` for high-level async primitives, and `asyncio` for low-level tasks.
  * **Concurrency:** Use `asyncio.gather` for parallel independent tasks. Be mindful of CPU-bound tasks in `async` functions (offload to `ThreadPoolExecutor` if necessary).

## 9\. Testing

Tests are integral to the development process.

* **Framework:** Use `pytest`.
* **Structure:** All tests live in `tests/` directory with descriptive filenames
  - `test_api_*.py` - API endpoint tests
  - `test_auth.py` - Authentication tests  
  - `test_*.py` - Other component tests
* **Fixtures:** Use `conftest.py` for shared fixtures
* **Mocking:** Use `unittest.mock` for external dependencies
* **Naming:** Test files: `test_feature.py`. Test functions: `test_specific_behavior()`
* **Coverage:** Aim for high coverage on critical paths (auth, API endpoints)

## 10\. Dependency Management

  * **`pyproject.toml`:** Use `pyproject.toml` (e.g., with Poetry, PDM, or Rye) for project metadata and primary dependency management. This is the source of truth for dependencies.
  * **`requirements.txt` / `requirements-dev.txt`:** Generate these from `pyproject.toml` for deployment and development environments respectively.
  * **Pin Dependencies:** Pin exact versions of production dependencies to ensure reproducible builds.

## 11\. Configuration

  * **Centralized Settings:** All configurable parameters must be defined in Pydantic `BaseSettings` models within `config/settings.py`.
  * **Precedence:** Environment variables should override `.env` file settings, which override config file settings, which override default values. This should be handled by `config/loader.py`.
  * **Type-Safe:** Leverage Pydantic's type validation for configuration values.

## 12\. Security Considerations

  * **Input Validation:** All API inputs **must** be validated using Pydantic models.
  * **Sensitive Data:** Never log raw API keys, tokens, or other sensitive user data. Mask or redact.
  * **Authentication:** Enforce authentication using `api/middleware/auth.py` and `auth/dependencies.py` where required.
  * **CORS:** Properly configure CORS origins in `api/middleware/cors.py` to only allow trusted clients.
  * **Least Privilege:** When running Docker containers, use `docker/adapter.py` and `docker/builder.py` to configure the least necessary privileges (e.g., specific UID/GID mapping, limited volumes).
  * **Dependency Scanning:** Regularly scan dependencies for known vulnerabilities.

## 13\. Tooling**


The `ccproxy` project leverages a modern, streamlined Python development toolchain to ensure high code quality, consistency, and efficient workflows. Our core tools are `uv` for package management and `ruff` for all code formatting, linting, and import sorting.

These tools are enforced via `pre-commit` hooks for local development and validated in GitHub Actions CI pipelines.

## **13.1. Core Tooling Stack**

* **Package Management & Dependency Resolution:** `uv`
    * Replaces traditional `pip` and dependency resolvers.
    * Handles installing, syncing, and publishing packages.
    * **Usage:** Orchestrated exclusively via the `Makefile` targets. Developers should **not** invoke `uv` directly.
* **Code Formatting:** `ruff format`
    * Ensures consistent code style across the entire codebase.
    * **Configuration:** Handled automatically by `ruff`'s default sensible settings, and `pyproject.toml` for project-specific overrides if needed (e.g., line length, though we use `Black`'s standard 88).
    * **Enforcement:**
        * **Local:** `pre-commit` hook (`ruff-format` hook).
        * **CI:** Part of the `make ci` and `make check` targets.
* **Linting:** `ruff check`
    * Identifies potential bugs, stylistic errors, and enforces best practices.
    * **Configuration:** Configured via `pyproject.toml`.
    * **Enforcement:**
        * **Local:** `pre-commit` hook (`ruff` hook with `--fix` arg).
        * **CI:** Part of the `make ci` and `make check` targets.
* **Import Sorting:** `ruff check --select I` (integrated into `ruff lint-fix`)
    * Automatically organizes import statements according to PEP 8.
    * **Configuration:** Handled by `ruff`'s import sorting capabilities.
    * **Enforcement:**
        * **Local:** `pre-commit` hook (`ruff` hook with `--fix` arg, or `make lint-fix`).
        * **CI:** Part of the `make ci` and `make check` targets.
* **Static Type Checking:** `MyPy`
    * Ensures type correctness and catches type-related errors early.
    * **Configuration:** Configured via `pyproject.toml` (refer to `[tool.mypy]` section). Specific `additional_dependencies` are listed in `.pre-commit-config.yaml` for MyPy's virtual environment.
    * **Enforcement:**
        * **Local:** `pre-commit` hook (`mypy` hook).
        * **CI:** Part of the `make ci` and `make check` targets.
* **Test Runner:** `pytest`
    * The standard framework for writing and running tests.
    * **Coverage:** Integrated with `pytest-cov` to generate test coverage reports.
    * **Enforcement:**
        * **Local:** `make test`.
        * **CI:** Part of the `make ci` target, and coverage reports are uploaded to Codecov.

## **14. Workflow Automation with Makefile**

To ensure **consistency and reproducibility** across all development environments, **all primary development tasks are orchestrated via `Makefile` targets.**

**Developers are required to use `make <target>`** instead of directly invoking `uv`, `ruff`, `mypy`, `pytest`, `mkdocs`, `docker`, or `docker-compose`.

### **14.1. Key `Makefile` Targets:**

| Category          | Makefile Target(s)      | Description                                                                                                                                           |
| :---------------- | :---------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Setup/Install** | `make install`          | Installs production dependencies.                                                                                                                     |
|                   | `make dev-install`      | Installs all development dependencies (including pre-commit hooks). **Run this after cloning the repo.** |
| **Cleanup** | `make clean`            | Removes build artifacts, `__pycache__`, coverage files, `node_modules`, etc.                                                                          |
| **Code Quality** | `make format`           | Formats all Python code using `ruff format`.                                                                                                          |
|                   | `make format-check`     | Checks if code is formatted correctly (used in CI).                                                                                                   |
|                   | `make lint`             | Runs `ruff check` for linting.                                                                                                                        |
|                   | `make lint-fix`         | Runs `ruff check --fix` and `ruff check --select I --fix` to fix linting and import issues automatically.                                             |
|                   | `make typecheck`        | Runs `mypy` for static type checking.                                                                                                                 |
|                   | `make check`            | Runs `lint`, `typecheck`, and `format-check`.                                                                                                         |
|                   | `make pre-commit`       | Manually runs all `pre-commit` hooks against all files (useful for staging changes).                                                                  |
| **Testing** | `make test`             | Runs the full `pytest` suite with coverage.                                                                                                           |
|                   | `make test-unit`        | Runs unit tests only.                                                                                                                                 |
|                   | `make test-integration` | Runs integration tests only.                                                                                                                          |
| **CI Automation** | `make ci`               | **The comprehensive CI pipeline target.** Runs `pre-commit` hooks and the full `test` suite, mirroring the GitHub Actions `CI` workflow.              |
| **Builds** | `make build`            | Builds the Python distributable package.                                                                                                              |
|                   | `make docker-build`     | Builds the Docker image locally.                                                                                                                      |
|                   | `make docker-run`       | Runs the locally built Docker image.                                                                                                                  |
|                   | `make docker-compose-up`| Starts the project using `docker-compose` (often for development).                                                                                    |
|                   | `make docker-compose-down`| Stops `docker-compose` services.                                                                                                                      |
| **Development** | `make dev`              | Starts the FastAPI development server with auto-reload (using `uv run fastapi dev`).                                                                  |
|                   | `make setup`            | One-time setup: runs `dev-install` and prints guidance.                                                                                               |
| **Documentation** | `make docs-install`     | Installs documentation-specific dependencies.                                                                                                         |
|                   | `make docs-build`       | Builds the project documentation.                                                                                                                     |
|                   | `make docs-serve`       | Serves the documentation locally for preview.                                                                                                         |
|                   | `make docs-clean`       | Cleans documentation build files.                                                                                                                     |
|                   | `make docs-deploy`      | Helper target, documentation deployment is typically handled by GitHub Actions (`docs.yml`).                                                            |
| **Help** | `make help`             | Displays all available `Makefile` targets and their descriptions.                                                                                     |

### **14.2. GitHub Actions CI Pipelines (`.github/workflows/`):**

* **`ci.yml` (Continuous Integration):**
    * Triggered on `push` to `main` and `develop`, and on `pull_request` to `main` and `develop`.
    * Installs `uv`.
    * Sets up multiple Python versions (`3.10`, `3.11`, `3.12`, `3.13`) for compatibility testing.
    * Installs dependencies via `make dev-install`.
    * **Executes `make ci`**, ensuring local and CI environments run the same suite of checks.
    * Builds documentation (`make docs-build`).
    * Uploads coverage reports to Codecov.
* **`build.yml` (Docker Image Build):**
    * Triggered on `push` to `main` or when `CI` workflow completes successfully on `main`.
    * Handles Docker login, metadata extraction, and multi-platform image building and pushing to `ghcr.io`.
    * Generates artifact attestations.
* **`release.yml` (Release Workflow):**
    * Triggered on `release` creation (when a new Git tag is pushed as a release).
    * **`build-package` job:** Installs dependencies (`make install`), builds the Python package (`make build`), uploads `dist/` artifacts, and publishes to PyPI using `uv publish`.
    * **`build-release-docker` job:** Builds and pushes Docker images to `ghcr.io` with release-specific tags (semver, major.minor, major).
    * **`create-release` job:** Downloads package artifacts and uploads them as assets to the GitHub Release.
* **`docs.yml` (Documentation Workflow):**
    * Triggered on `push` to `main` or `dev` (if `docs/**` or relevant code files change) and on `pull_request`.
    * Installs `uv`.
    * Sets up Python `3.13`.
    * Installs documentation dependencies via `uv sync --group docs`.
    * Builds documentation (`uv run mkdocs build`).
    * Deploys to GitHub Pages (`main` branch `push` only).
    * Includes a `check` job to validate documentation links by starting a local server and running `linkchecker`.

---
