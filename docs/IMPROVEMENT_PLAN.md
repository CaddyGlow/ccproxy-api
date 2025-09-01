# CCProxy Refactoring & Improvement Plan

This document outlines a prioritized refactoring plan to strengthen profile lifecycle consistency, unify HTTP client usage, simplify observability, and improve extensibility, robustness, and developer experience.

## Goals

- Consolidate profile lifecycle so loading/updating reliably refreshes persisted fields.
- Unify HTTP client lifecycle and eliminate duplicate patterns.
- Simplify observability routes and config precedence with clear ownership.
- Strengthen plugin extensibility and storage robustness.
- Improve security (masking) and packaging of static assets.

## Guiding Principles

- Consistency: follow CONVENTIONS.md (types, imports, ruff/mypy, 88 cols).
- Separation: isolate side effects behind services and adapters; keep APIs small.
- Determinism: explicit config precedence; predictable plugin ownership of routes.
- Observability: structured JSON logs; no secrets; opt-in metrics via plugin.
- Developer UX: minimal diffs, clear PR scope, tests first where practical.

## Status Tracker (Live)

- Quick wins (target this week)
  - [x] Fix `asyncio.TimeoutError` catch in DuckDB storage worker.
  - [x] Replace `get_http_client` dependency with container-provided client.
  - [x] Ensure `/metrics` served by plugin; guard any core stub with `is_enabled()`.
  - [x] Add static assets to wheel/sdist includes.
- In flight
  - [ ] Phase 1 scaffolding (models/service/repository) created.
  - [ ] Phase 2 dependency injection updated and tested.
  - [ ] Phase 3 metrics plugin hooks wired and tested.

## Phase 1: Profile Lifecycle (P0)

- Model & Storage
  - Add a unified `Profile` model: provider, user_id, email, plan, organization, subscription dates, updated_at.
  - Create `ProfileRepository` backed by DuckDB (table `profiles`) with CRUD and indexed lookups.
- Service Abstraction
  - Add `ProfileService` with: `load_profile(provider)`, `refresh_profile(provider)`, `update_fields(provider, **fields)`, `get_cached(provider)` (with TTL cache).
  - Centralize writes through `ProfileService.update_fields(...)` to ensure fields update whenever profile is loaded/refreshed.
- Hooks Integration
  - Emit `PROFILE_UPDATED` whenever fields are written.
  - Subscribe to auth/token refresh/login hooks to auto-refresh profile.
- Plugin Integration
  - Replace ad-hoc JWT parsing in plugins (e.g., Codex) with `ProfileService` usage; keep provider-specific extraction in small adapters invoked by the service.
- API/CLI
  - Add `GET /api/profile` and `GET /api/profile/{provider}`.
  - Add `ccproxy auth profile` to display effective profile.
- Tests
  - Unit tests for repository, service, plugin adapters -> service integration, API responses, hook emissions.

Acceptance: updating tokens or calling refresh updates persisted profile fields; API returns latest; hooks fire; tests pass.

Phase 1 Checklist (per CLAUDE.md & Conventions)

- Dev server: run `make dev` with reload; confirm JSON logs and trace dirs.
- Style: `ruff format` (88 cols), `ruff check`; PEP 8; no mutable defaults.
- Naming/imports: modules `snake_case`, classes `CamelCase`, constants `UPPER_SNAKE_CASE`; absolute imports within `ccproxy`; stdlib→3rd→first-party order.
- Typing: full type hints on all functions and models; use `X | None` for optionals; avoid `Any`; add aliases if complex.
- Async: use async/await for all I/O; no blocking calls in async paths.
- Errors: raise `CCProxyError`-derived exceptions where appropriate; catch specific exceptions; use `raise ... from e`.
- Config: use Pydantic Settings; precedence CLI → Env → TOML → Defaults; env nesting with `__`.
- Logging/Security: use `LoggingSettings`; mask tokens/PII in logs; do not log sensitive bodies; no secrets in code or tests.
- Tests: pytest + pytest-asyncio; unit tests for repo, service, adapters→service, API, hooks; mark with `@pytest.mark.unit`.
- Docs: docstrings for new public APIs; concise rationale comments where non-obvious.
- Git/PR: Conventional Commits; focused changes; never `git add .`; run `make pre-commit` and `make test` before pushing.
- Patterns: follow plugin delegation pattern for integration points; avoid overengineering; keep functions cohesive.

Phase 1 PR Criteria

- Functionality: profile load/refresh/write path consistent; hooks emit `PROFILE_UPDATED`; API/CLI return current profile.
- Quality: ruff, mypy, and formatting pass; imports sorted; types strict.
- Tests: green unit suite with coverage over repo/service/API/hook paths.
- Docs: updated README/mkdocs as needed; docstrings present for new modules/classes.

## Phase 2: HTTP Client Unification (P0)

- Single Source of Truth
  - Route all HTTP through `ServiceContainer` + `HTTPPoolManager`.
  - Deprecate `ccproxy.core.http_client` singleton for request-time dependencies.
- Dependencies
  - Change `ccproxy/api/dependencies.get_http_client` to fetch client from the container.
- Shutdown
  - Rely on `ServiceContainer.close()`; remove the separate "Shared HTTP Client" shutdown component when no longer used.

Acceptance: no double-close; one client lifecycle; tests for shutdown and reuse are green.

Phase 2 Checklist (per CLAUDE.md & Conventions)

- Types/async: prefer `httpx.AsyncClient`; ensure graceful `aclose()` via container.
- Dependencies: route through `ServiceContainer`; update `get_http_client` dependency.
- Tests: unit tests for reuse and shutdown; markers `@pytest.mark.unit`.
- Quality gates: `make pre-commit` (ruff, mypy, format) passes; imports sorted.
- Logging: do not double-log; consistent levels via `LoggingSettings`.
- Commits: Conventional Commits; minimal, focused PR.

Phase 2 PR Criteria

- Exclusivity: no usage of legacy `ccproxy.core.http_client`; all codepaths use container-provided client.
- Lifecycle: one `AsyncClient` per container; `aclose()` invoked exactly once on shutdown.
- Tests: reuse/shutdown unit tests pass; no resource warnings.
- Quality: ruff, mypy, and formatting pass; imports sorted.
- Docs: dependency injection note updated (dependencies → container pattern).

## Phase 3: Observability Coherence (P1)

- Metrics Endpoint Ownership
  - Expose `/metrics` only via the metrics plugin; remove or hard-disable core `ObservabilityMetricsDep` expectations in routes.
  - If keeping a core stub, ensure it implements `is_enabled()` to prevent `None` attribute errors.
- Hooks Coverage
  - Metrics plugin subscribes to `REQUEST_STARTED`, `REQUEST_COMPLETED`, `REQUEST_FAILED` and optional chunk events.

Acceptance: `/metrics` works when plugin is enabled; no None-attribute errors; tests aligned.

Phase 3 Checklist (per CLAUDE.md & Conventions)

- Ownership: metrics exposed by plugin only; guard stubs with `is_enabled()`.
- Hooks: subscribe to request lifecycle events; avoid blocking operations.
- Tests: enable/disable plugin matrix; Prometheus client assertions.
- Logging: structured JSON; avoid sensitive data; use centralized settings.
- Docs: update README/mkdocs to reflect plugin ownership of `/metrics`.
- Quality gates: ruff/mypy green; keep functions under 88 columns.

Phase 3 PR Criteria

- Behavior: `/metrics` available only when metrics plugin enabled; disabled otherwise with clear response.
- Robustness: no `None` attribute errors; guarded stubs implement `is_enabled()`.
- Hooks: plugin subscribes to required lifecycle events and records metrics.
- Tests: enable/disable matrix and Prometheus assertions pass.
- Docs: README/mkdocs updated to reflect plugin ownership and usage.

## Phase 4: Config Precedence Simplification (P1)

- Precedence
  - Defaults → TOML → Env → CLI overrides (documented and enforced).
- Implementation
  - Use Pydantic-friendly update flow or a small deep-merge utility; simplify plugin dict merge with deterministic precedence.
- Tests
  - Matrix covering nested fields and plugin configs across precedence layers.

Acceptance: deterministic merges; fewer special cases; tests cover common permutations.

Phase 4 Checklist (per CLAUDE.md & Conventions)

- Implementation: enforce Defaults → TOML → Env → CLI; use Pydantic-safe deep-merge.
- Env vars: nested with `__` per conventions; document examples.
- Tests: precedence matrix incl. nested plugin dicts; mark unit.
- Types: `TypedDict`/models for config where helpful; no `Any`.
- Docs: document precedence and migration notes; update examples.
- Quality gates: `make pre-commit` must pass.

Phase 4 PR Criteria

- Determinism: precedence order enforced (Defaults < TOML < Env < CLI) with tests.
- Types: no `Any` in config merge paths; optional fields typed with `| None`.
- DX: examples and migration notes updated; env var nesting with `__` documented.
- Quality: ruff/mypy/format pass; small, focused changes.

## Phase 5: Plugin Extensibility (P2)

- Entry Points
  - Add `ccproxy.plugins` entry point group; discovery folds together entry points with local `plugins/` directory.
- Status Endpoint
  - Add `/api/plugins/status` summarizing manifests, initialization order, provides/requires services, and runtime presence.

Acceptance: third-party plugins load via entry points; status endpoint returns structured information.

Phase 5 Checklist (per CLAUDE.md & Conventions)

- Discovery: add `ccproxy.plugins` entry point group; merge with local `plugins/`.
- Structure: enforce delegation pattern (`adapter.py`, transformers, auth).
- Status API: include provides/requires, init order; no secrets in output.
- Tests: load external dummy plugin via entry points; status endpoint contract.
- Docs: authoring guide snippet; update plugin structure diagram.
- Security: validate manifests; handle import errors safely; log at debug.

Phase 5 PR Criteria

- Discovery: entry point loading works for a sample external plugin and merges with local `plugins/`.
- Status API: returns stable schema (manifests, provides/requires, init order) with tests.
- Resilience: missing/broken plugin import handled gracefully with debug logs.
- Docs: plugin authoring and discovery sections updated.

## Phase 6: Storage Robustness (P2)

- Background Worker Resilience
  - Catch `asyncio.TimeoutError` in `SimpleDuckDBStorage._background_worker` (instead of `TimeoutError`).
  - Optional: bounded queue size and warn logs when lag exceeds a threshold; make configurable via settings.
- Profiles Table
  - Add schema and SQLModel model for `profiles`; minimal migrations support if needed.

Acceptance: worker stable under idle and load; profile CRUD works; tests added.

Phase 6 Checklist (per CLAUDE.md & Conventions)

- Worker: catch `asyncio.TimeoutError`; consider bounded queue + warnings.
- Profiles: add schema/model and minimal migrations; index common lookups.
- Tests: idle/load scenarios; timeout handling; CRUD coverage.
- Logging: avoid noisy error logs under normal idle; structured fields.
- Types: strict typing for repository interfaces; no mutable defaults.
- Quality gates: ruff/mypy/tests all green.

Phase 6 PR Criteria

- Stability: worker idle without noisy errors; load with bounded lag and warnings as configured.
- Profiles: CRUD operations persist reliably; indexed lookups performant (basic assertions in tests).
- Tests: timeout handling and CRUD tests pass; no flaky behavior.
- Quality: structured logs; strict typing; pre-commit passes.

## Phase 7: Security & Masking (P3)

- Central Sanitizer
  - Shared helpers to mask tokens in headers/bodies; apply in adapters, transformers, hooks, middleware, and error handlers.
- Optional Rate Limiting
  - Lightweight, configurable limiter middleware (per IP and/or per token), pluggable or core.

Acceptance: no token leakage in logs; limiter opt-in works.

Phase 7 Checklist (per CLAUDE.md & Conventions)

- Sanitization: central helpers to mask tokens/credentials; reuse in adapters/middleware.
- Coverage: error handlers and hooks also sanitized; redact bodies where needed.
- Limiter: optional, configurable; avoid blocking async loop; document env vars.
- Tests: ensure masked logs; rate limit behavior basic unit tests.
- Docs: update security section; examples of env-based toggles.
- Quality gates: pre-commit green; no secrets in tests or fixtures.

Phase 7 PR Criteria

- Redaction: tokens/credentials masked in logs across adapters, middleware, and error handlers (verified in tests).
- Limiter: disabled by default; enables via settings; non-blocking and configurable.
- Security: no sensitive bodies logged; guidance in docs; examples for env toggles.
- Tests: masking and basic limiter behavior covered; suite green.

## Phase 8: Packaging & Assets (P2)

- Static Assets
  - Include `ccproxy/static/dashboard/**` in wheel and sdist `pyproject.toml` includes.
  - Keep feature flag for dashboard and preserve actionable 404 message when missing.

Acceptance: dashboard assets ship; mount works when enabled.

Phase 8 Checklist (per CLAUDE.md & Conventions)

- Packaging: include static assets in wheel/sdist via `pyproject.toml`.
- Feature flag: dashboard disabled by default with actionable 404.
- CI: add wheel content check for asset paths.
- Tests: import/resource discovery sanity; skip if flag disabled.
- Docs: note flag and asset packaging; upgrade notes.
- Quality gates: ruff/mypy unaffected; packaging builds cleanly.

Phase 8 PR Criteria

- Packaging: wheel and sdist include `ccproxy/static/dashboard/**` (verified by CI step or local check).
- Feature flag: dashboard route respects flag; actionable 404 when disabled/missing.
- Docs: packaging/flag behavior documented; upgrade notes added if needed.
- CI: wheel content check updated to guard regressions.

## Phase 9: Testing & CI (P3)

- Observability Tests
  - Update/reenable tests to match metrics plugin ownership and Prometheus client behavior.
- Additional Tests
  - HTTP lifecycle (single client), config precedence, plugin provides/requires resolution (positive and negative cases).
- CI Hardening
  - mypy stricter on plugin modules; wheel content check for dashboard assets.

Acceptance: green CI with added coverage on critical paths.

Phase 9 Checklist (per CLAUDE.md & Conventions)

- Observability: align tests with plugin-owned metrics; avoid None access.
- Lifecycle: tests for single HTTP client reuse and shutdown.
- Config: precedence tests; plugin dependency resolution positive/negative.
- CI: type-check plugin modules; enforce Conventional Commits (lint in CI if available).
- Coverage: ensure critical paths covered; mark tests appropriately.
- Docs: link to TESTING.md quick commands; keep examples current.

Phase 9 PR Criteria

- Coverage: tests added for observability ownership, HTTP lifecycle, config precedence, and plugin dependency resolution.
- CI: mypy stricter rules on plugin modules pass; commit style checks (if configured) pass.
- Stability: all test jobs green; no unnecessary flakiness introduced.
- Documentation: TESTING.md references validated; examples current.

## Quick Wins (Target First)

- Fix `asyncio.TimeoutError` catch in DuckDB storage worker.
- Adjust `/metrics` endpoint to avoid `ObservabilityMetricsDep` None usage or stub it safely; prefer plugin ownership.
- Add static assets to build includes.
- Replace `get_http_client` dependency with container-provided client.
- Add a basic integration test ensuring plugin-registered `/metrics` route works as expected.

## Sequencing

- P0: Phase 1 (Profile lifecycle), Phase 6 (worker fix), Phase 2 (HTTP unification).
- P1: Phase 3 (Observability), Phase 4 (Config precedence).
- P2: Phase 5 (Plugins via entry points), Phase 8 (Assets packaging).
- P3: Phase 7 (Security/masking), Phase 9 (Testing & CI).

## Deliverables

- New modules
  - `ccproxy/models/profile.py`
  - `ccproxy/storage/repositories/profile_repository.py`
  - `ccproxy/services/profile.py`
- Refactors
  - HTTP deps to container; remove legacy shared client usages.
  - Observability route ownership by metrics plugin.
  - Config loader simplification and documentation of precedence.
- Documentation
  - Profile lifecycle and hooks; config precedence; plugin authoring via entry points; observability model.
- Tests
  - Profile service/repo/API; HTTP lifecycle; config precedence; plugin dependency resolution.

## Risks & Mitigations

- Breaking route ownership: moving `/metrics` to a plugin can surprise users.
  - Mitigation: feature flag + actionable 404; clear release notes and docs.
- HTTP client lifecycle regressions: double-close or leaks during transition.
  - Mitigation: container-managed lifecycle, targeted reuse/shutdown tests.
- Config precedence changes: silent behavior differences across environments.
  - Mitigation: config precedence matrix tests; migration notes and examples.
- Plugin discovery via entry points: startup failures due to bad third-party plugins.
  - Mitigation: sandboxed import with error isolation and debug logs; health status API.

## Migration Notes (Operator-Facing)

- Observability: `/metrics` is provided by the metrics plugin. Enable via config; otherwise expect a clear 404.
- Config: precedence is Defaults < TOML < Env < CLI. Use `__` for nested env vars.
- HTTP: all outbound HTTP is unified via the container; legacy singleton is deprecated.
- Packaging: dashboard assets included in wheel/sdist when enabled by feature flag.

## References

- REFACTORING_PLAN.md – broader refactor context and rationale.
- HOOKS_INTEGRATION_PLAN.md – hook events and integration patterns.
- HOOK_ORDERING_IMPLEMENTATION_PLAN.md – deterministic hook ordering.
- DEPENDENCY_INJECTION_PLAN.md – DI patterns and container guidance.
- TESTING.md – test setup, markers, coverage guidance.
- CONVENTIONS.md – formatting, linting, typing, and naming conventions.

---

Owned by: Core maintainers (with plugin-specific contributions).
Tracking: Create a milestone with P0–P3 issues matching the phases above.
