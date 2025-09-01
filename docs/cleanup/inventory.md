# Codebase Inventory and Baseline (Phase 1)

Last updated: 2025-09-01

Purpose: Establish a clear baseline after the plugin migration, locate legacy or redundant code, and capture quality gates before removals/refactors.

## Summary

- Core and plugins are both present; plugins are wired via entry points.
- Lint passes; type check currently fails with widespread mypy errors (baseline captured).
- Several compatibility shims remain in core; candidates for consolidation or removal in later phases.

## Plugin Entry Points (pyproject)

- access_log → `plugins.access_log.plugin:factory`
- claude_api → `plugins.claude_api.plugin:factory`
- claude_sdk → `plugins.claude_sdk.plugin:factory`
- codex → `plugins.codex.plugin:factory`
- permissions → `plugins.permissions.plugin:factory`
- request_tracer → `plugins.request_tracer.plugin:factory`
- duckdb_storage → `plugins.duckdb_storage.plugin:factory`
- analytics → `plugins.analytics.plugin:factory`
- dashboard → `plugins.dashboard.plugin:factory`

## Code Map (core)

- `ccproxy/api/`: app factory, middleware (errors, CORS, hooks, request_id), routes (`health`, `plugins`).
- `ccproxy/services/`: service container and adapter base/http adapters; request_context.
- `ccproxy/plugins/`: plugin runtime, declaration, protocol, factory, discovery, middleware.
- `ccproxy/core/`: async utilities, http/http_client, logging, errors, system, transformers, interfaces, types, constants.
- `ccproxy/auth/`: auth protocols, managers, OAuth router/routes, storage backends, models.
- `ccproxy/cli/`: CLI entry (`main.py`), commands (serve, auth, config, plugins), docker helpers.
- `ccproxy/config/`: `Settings` and nested models (server, logging, security, cors, http, docker, scheduler, binary, discovery, validators, constants). reverse_proxy, observability, plugin_dir, and hooks removed.
- `ccproxy/streaming/`: streaming interfaces and helpers.
- `ccproxy/utils/`: helpers (model mapping, binary resolver, version checker, caching, id generator, startup helpers, cost_calculator, CORS, models_provider).

Notable legacy-leaning areas:

- `ccproxy/adapters/openai/*`: an in-core OpenAI adapter potentially overlapped by plugin-based format adapters (`plugins/codex`, `plugins/claude_*`).
- Backward-compat in `api/dependencies.py` (metrics stub, DuckDB aliases) and `api/middleware/hooks.py` mentions coexistence with prior ProxyService hook emission.

## Code Map (plugins)

- Provider plugins: `plugins/claude_api`, `plugins/claude_sdk`, `plugins/codex`, `plugins/oauth_*`.
- Observability/ops: `plugins/access_log`, `plugins/request_tracer`, `plugins/metrics`, `plugins/analytics`, `plugins/dashboard`.
- Storage: `plugins/duckdb_storage`.
- Permissions: `plugins/permissions` (MCP integration).

Each provider implements `adapter.py`, `routes.py`, transformers, health/tasks/hooks and has its own `pyproject.toml` (workspace package).

## Dependencies (root pyproject)

- Core deps: `fastapi[standard]`, `httpx[http2]`, `pydantic`, `pydantic-settings`, `typer`, `uvicorn`, `structlog`, `rich`, `prometheus-client`, `duckdb`, `duckdb-engine`, `sqlmodel`, `aiofiles`, `aiosqlite`, `aiohttp`, `h2`, `sortedcontainers`, `aioconsole`, `packaging`, `typing-extensions`.
- Dev deps: `ruff`, `mypy`, `pytest`, `pytest-asyncio`, `pytest-cov`, `pytest-httpx`, etc., plus provider SDKs (`anthropic`, `openai`, `claude-code-sdk`) for tests.

Plugin-local deps (examples):

- `plugins/codex`: `openai>=1.93.0`
- `plugins/claude_api`: `pyjwt>=2.10.1`
- `plugins/permissions`: `fastapi-mcp>=0.3.7`
- `plugins/request_tracer`: `httpx`, `aiofiles`, `structlog`
- `plugins/analytics`: `sqlmodel`, `structlog`

Initial removal candidates (dependency angle):

- Any root dependency used only by plugin code should migrate into that plugin; inventory required via import scan (planned in Phase 2).

## Configuration Map

Top-level `Settings` keys (env prefix uses `__` nesting):

- `server`, `logging`, `security`, `cors`, `http`, `binary`, `docker`, `scheduler`, `enable_plugins`, `plugins` (dict per plugin). reverse_proxy, observability, plugin_dir, and hooks removed; plugins own those concerns.

Behavioral notes:

- Config files: `.ccproxy.toml` (cwd), `ccproxy.toml` (repo root), or `XDG_CONFIG_HOME/ccproxy/config.toml` discovered via `find_toml_config_file()`.
- CLI overrides injected via `CCPROXY_CONFIG_OVERRIDES` JSON or parsed flags in CLI.
- Legacy logging envs are consolidated under `LOGGING__*` (see CLAUDE.md).

Potential legacy config keys to phase out:

- Compatibility stubs in `api/dependencies.py` suggest legacy metrics/log storage naming; ensure plugins own these settings.

## Tests Map

- Unit tests: `tests/unit/*` (CLI, config precedence, hooks, services, utils, API smoke), helpers, fixtures.
- Integration tests: `tests/integration/*` (access logging, metrics plugin, streaming, analytics pagination).
- Additional work-in-progress tests under `tests_new/` (contain type issues vs current API).

Counts (approx):

- Unit/integration files discovered: 60+ test files.

## Baseline Quality Gates

- Lint: ruff check → All checks passed.
- Types: mypy → 219 errors across 60 files (baseline captured). Representative failures in plugin formatters, oauth CLI integration, duckdb storage typing, and `tests_new/*` using outdated interfaces.
- Coverage: collected via pytest (excluding one legacy test import error) → line-rate ~65.9% for `ccproxy`. HTML in `htmlcov/`, XML in `coverage.xml`.

### Coverage Highlights (potential dead zones)

- `ccproxy/utils/models_provider.py` → 0% covered; no in-repo references found. Candidate for removal or plugin relocation.
- `ccproxy/adapters/openai/response_adapter.py` → ~13% covered (low; but used by plugins and tests).
- `ccproxy/adapters/openai/streaming.py` → ~14.6% covered (low; used by plugins).

Note: Many tests fail currently due to known migration gaps (scheduler/observability hooks, binary resolver expectations). Coverage snapshot is still useful for dead-code triage.

## Preliminary Redundancy/Dead Code Candidates

- `ccproxy/utils/models_provider.py`: Unreferenced and 0% covered. Action: remove or fold into relevant plugin route that lists models.
- `ccproxy/adapters/openai/*`: Core OpenAI adapter is still referenced by plugins (`claude_api`, `claude_sdk`, `codex`) and tests; not dead. Action: keep as shared format adapter for now; consider moving to a shared package if desired later.
- Core compatibility stubs removed: metrics/log storage stubs were deleted from `ccproxy/api/dependencies.py`. Plugins expose their own deps; analytics now imports storage from `plugins/duckdb_storage`.
- `ccproxy/api/middleware/hooks.py` updated to remove ProxyService-era references; centralized hook emission remains.
- Root-level dependency ownership: ensure provider-specific deps (e.g., `openai`, `anthropic`, `pyjwt`) are not required at root unless used by core.
- `tests_new/*`: Targets removed interfaces (e.g., `ProxyService`, `IPluginProvider` from old modules). Action: either update to current interfaces or remove.

## Dependency Ownership (initial scan)

Summary of root runtime dependencies and where they are used:

- Core-only or Core+Plugins: `fastapi`, `httpx[http2]`, `pydantic`, `pydantic-settings`, `typer`, `uvicorn`, `structlog`, `rich`, `rich-toolkit`, `typing-extensions`, `packaging`, `aiohttp`, `sortedcontainers`, `aiofiles`, `jsonschema`.
- Plugin-only: `prometheus-client` (metrics plugin), `sqlmodel` (analytics, duckdb_storage), `duckdb`, `duckdb-engine` (duckdb_storage), `textual` (permissions terminal handler).
- Unused in repo: `aiosqlite` (no imports found), `keyring` (mentioned in docs but not imported), `h2` (provided transitively by `httpx[http2]` if needed; no direct imports).

Notes and recommendations:

- `duckdb` / `duckdb-engine` / `sqlmodel` / `prometheus-client` / `textual`: keep for now since plugins ship inside the main wheel; mark as plugin-only and consider moving to plugin-specific distributions or optional extras in a future split.
- `keyring`: remove from core runtime deps and keep only under the optional `security` group (already declared) unless we add real usage.
- `aiosqlite`: remove from root runtime deps (no usage found).
- `h2`: remove explicit pin from root runtime deps; `httpx[http2]` will pull it transitively when needed.



## Open Questions / To Validate

- Are any core routes still dispatching through legacy adapters versus plugin adapters?
- Can the metrics and access logging be fully delegated to plugins without core stubs?
- Is `ccproxy/utils/models_provider.py` still needed vs plugin-supplied model listings?

## Next Actions (Phase 1 completion)

- Run `make test-coverage` to capture coverage and identify dead zones.
- Static usage scan (ripgrep) for `ccproxy/adapters/openai` and other suspects to confirm unreferenced state.
- Build a dependency usage report to identify root deps used only by plugin code.
- Draft removal/replace/deprecate decisions list for Phase 2.
