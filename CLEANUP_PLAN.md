# CCProxy Codebase Cleanup Plan

Last updated: 2025-09-01

Owner: CCProxy maintainers
Scope: Remove redundancy/dead code from the migration to a plugin-based architecture; tighten boundaries, update tests/docs, and simplify configuration.

## Guiding Principles

- Keep changes incremental, small, and test-backed.
- Prefer deprecation windows for public APIs; batch removals into a major release.
- Enforce plugin boundaries in code, config, and docs.
- Maintain full type coverage and pre-commit hygiene.

## Progress Snapshot

- [x] Draft plan document and baseline approach
- [x] Phase 1: Inventory & Baseline
- [x] Phase 2: Dead Code & Redundancy Detection
- [x] Phase 3: Boundaries & Interfaces
- [ ] Phase 4: Redundant Code Removal (in progress)
- [ ] Phase 5: API/CLI/Config Surface
- [ ] Phase 6: Tests Refactor
- [ ] Phase 7: Docs & Communication
- [ ] Phase 8: CI, Linting, Types
- [ ] Phase 9: Incremental PR Plan & Release

## Phase 1: Inventory & Baseline

Goal: Establish a clear baseline and identify legacy vs plugin-owned code.

- [x] Code map: catalog `ccproxy/*`, `plugins/*`, and `pyproject` entry points
- [x] Dependency map: list deps used only by legacy code; mark for removal
- [x] Config map: collect config keys/env vars; note keys superseded by plugins
- [x] Tests map: identify tests bound to legacy code; map to plugin ownership
- [x] Baseline quality: run checks and capture coverage snapshot
- [x] Deliverable: Inventory report in `docs/cleanup/inventory.md`

## Phase 2: Dead Code & Redundancy Detection

Goal: Find code paths and settings made obsolete by the plugin migration.

- [x] Static usage scan: ripgrep for imports/references of suspected legacy modules
- [x] Coverage audit: locate files/branches with zero hits
- [x] API/CLI scan: endpoints/commands that still have legacy shims
- [x] Config scan: detect unused settings via Settings model and references
- [x] Deliverable: Candidate checklist with remove/replace/deprecate decisions (`docs/cleanup/phase2-checklist.md`)

## Phase 3: Boundaries & Interfaces

Goal: Make plugin contracts explicit and enforceable; centralize loading.

- [ ] Define `Protocol`/`ABC` for each plugin kind (provider, adapter, formatter)
- [x] Ensure discovery via entry points + local workspace; avoid ad-hoc shims
- [x] Import hygiene: add boundary check (no core → plugin concretes)
- [x] Centralize plugin loading; app and CLI use `load_plugin_system`
- [x] Deliverable: `ccproxy/plugins/loader.py` and interface modules finalized
- [x] Add import boundary check script (`scripts/check_import_boundaries.py`) and Makefile target `check-boundaries`

## Phase 4: Redundant Code Removal

Goal: Remove duplicated core implementations; keep interfaces + orchestration only.

- [ ] Delete core code duplicated by plugins (adapters/services)
- [ ] Remove transitional shims that conditionally prefer plugins
- [ ] Remove feature flags toggling legacy vs plugin paths
- [x] Drop dependencies used only by removed code (update `pyproject`) — removed `aiosqlite`, runtime `keyring`, and explicit `h2`
- [ ] Trim obsolete config keys and add migration-friendly validation errors
- [x] Remove dead modules — deleted `ccproxy/utils/models_provider.py`
- [x] Remove core compatibility stubs — removed metrics/log storage stubs from `ccproxy/api/dependencies.py`; analytics now depends on plugin storage directly

Detailed removal targets (check off as completed):

- [ ] Remove legacy metrics Pushgateway settings from config docs and code
  - Files: `config.example.toml` (docs note already added), any `Settings` model fields (ensure none remain)
  - Tests: update or delete any assertions referencing `pushgateway_*`
- [ ] Remove any core→plugin back-compat aliasing for DuckDB storage
  - Files: core dependencies/injection; ensure analytics imports storage from `plugins.duckdb_storage`
  - Config: ensure `plugins.duckdb_storage.register_app_state_alias` defaults false
- [ ] Eliminate conditional imports that select between core and plugin adapters
  - Files: app/CLI startup paths; centralize on `load_plugin_system()`
- [ ] Delete feature flags or env vars that toggle legacy behavior
  - Examples: old `OBSERVABILITY__*` push flags, legacy auth/storage toggles


## Phase 5: API, CLI, and Config Surface

Goal: Expose a single plugin-driven surface; remove legacy-only routes/commands.

- [x] Ensure API routes delegate via plugin adapters only
- [ ] Remove legacy endpoints and CLI subcommands
- [x] Namespace plugin settings (e.g., `plugins.<name>.*`) with defaults/examples
- [x] Update `config.example.toml` (plugin-first examples); review `.env.example` later

Actionable checklist:

- [ ] Audit API routers for any legacy shims
  - Expected public routes: health, plugins, and plugin-mounted routes only
- [ ] CLI: remove hidden/legacy commands if any remain; ensure `plugins` command sources loader
- [ ] Config docs: add a small “Plugin Config Quickstart” to README linking to `docs/user-guide/pool-configuration.md`
- [ ] Provide migration error messages for deprecated config keys (fail-fast with help text)

## Phase 6: Tests Refactor

Goal: Align tests to plugin-first architecture with strong coverage.

- [ ] Remove or rewrite tests for deleted core code
- [ ] Move provider-specific tests under plugin packages
- [x] Standardize plugin fixtures/deps: analytics/metrics tests use plugin deps
- [x] Hooks tests: updated assertions for structured logging
- [x] Binary resolver tests: guard env-specific paths; add TRACE fallback in caching
- [x] Scheduler tests: remove pushgateway expectations (moved to metrics plugin)
- [ ] Integration tests for discovery and end-to-end flows
- [ ] Maintain/raise coverage thresholds

Test work items:

- [ ] Move provider-specific tests into their plugin packages alongside code
- [ ] Exclude `tests_new/*` from CI until ported; track a task list for porting
- [ ] Add integration tests for: plugin discovery failure modes, analytics+duckdb end-to-end, metrics endpoint exposure
- [ ] Raise coverage gate gradually (e.g., +2% per PR touching tests)

## Phase 7: Docs & Communication

Goal: Clarify architecture, contracts, and migration path.

- [ ] Architecture docs: plugin anatomy, lifecycle, and discovery
- [ ] Migration guide: map core X → plugin Y, config changes, deprecations
- [ ] README/examples: plugin-first usage; update env vars and commands
- [ ] Changelog: deprecations/removals with versions/dates

Documentation tasks:

- [ ] Architecture overview: consolidate `docs/PLUGIN_SYSTEM_DOCUMENTATION.md` and `docs/PLUGIN_AUTHORING.md` into an “Architecture” section in `docs/index.md`
- [ ] Migration guide: plugin-first changes, config key removals, Pushgateway → metrics plugin
- [ ] Per-plugin pages: ensure each bundled plugin has config and examples page (link existing READMEs)
- [ ] Update quickstart to include enabling/disabling plugins via `plugins.*` keys

## Phase 8: CI, Linting, and Type Safety

Goal: Guard boundaries and keep quality gates strong.

- [ ] CI job to import only public plugin interfaces from core
- [ ] Tighten `ruff` rules for import layering and ordering
- [ ] Enforce `mypy` across core and bundled plugins
- [ ] Keep pre-commit hooks fast; remove legacy-specific checks

CI/type-safety tasks:

- [ ] Add CI step to run `scripts/check_import_boundaries.py` and fail on violations
- [ ] Configure `mypy` to check plugin packages; create/refresh baseline if needed
- [ ] Tighten `ruff` import rules (`I`/`TID`), forbid core→plugin concrete imports
- [ ] Ensure `make` targets exist: `check-boundaries`, `typecheck`, `lint`, `test`

## Phase 10: Plugin-by-Plugin Cleanup

Goal: Ensure each bundled plugin is free of legacy coupling and follows the standard layout.

- Access Log (`plugins/access_log`)
  - [ ] Confirm no direct writes to core loggers beyond structured events
  - [ ] Validate `client_*` options; deprecate unused keys
  - [ ] Tests: round-trip structured/common/combined formats
- Request Tracer (`plugins/request_tracer`)
  - [ ] Ensure raw HTTP logging is optional and documented
  - [ ] Verify trace directory handling and rotation
  - [ ] Tests: verbose API on/off matrices
- Claude API/SDK (`plugins/claude_api`, `plugins/claude_sdk`)
  - [ ] Ensure adapters depend only on shared interfaces; avoid core concretes
  - [ ] Stream handling: unify SSE parsing and error propagation
  - [ ] Tests: health endpoints, streaming happy-path and error-path
- Codex (`plugins/codex`)
  - [ ] Confirm OpenAI client only imported inside plugin; no root import
  - [ ] Response/Request transformers parity with Claude plugins
  - [ ] Tests: transform correctness + SSE chunking
- DuckDB Storage (`plugins/duckdb_storage`)
  - [ ] Remove any app-state alias unless explicitly enabled
  - [ ] Migrations: idempotent schema setup and vacuum policy
  - [ ] Tests: persistence across restarts; concurrent writers
- Analytics (`plugins/analytics`)
  - [ ] Depend on storage plugin via interface; no inline engines
  - [ ] Pagination and filtering correctness for large tables
  - [ ] Tests: backpressure and memory limits
- Dashboard (`plugins/dashboard`)
  - [ ] Static mounting paths; CSP-safe assets
  - [ ] Feature flag alignment with `observability.*`
  - [ ] Tests: assets served and 404s
- Permissions (`plugins/permissions`)
  - [ ] MCP integration boundaries and timeouts
  - [ ] Tests: policy evaluation and denial logging
- OAuth (`plugins/oauth_codex`, `plugins/oauth_claude`)
  - [ ] Token storage/refresh boundaries; remove unused models
  - [ ] Tests: provider login and callback flows

## Operational Playbook

- Small PRs only; keep removals scoped and reversible
- Add deprecation warnings one release ahead where feasible
- When removing config keys, add a validation error with a migration hint
- After each phase increment, update this document’s Progress Snapshot

## Open Questions

- Do we plan to split bundled plugins into separate distributions soon? If yes, move plugin-only deps to extras now and prepare `pyproject` changes.
- Any remaining legacy CLI commands to retire, or are they already removed?
- Should `tests_new/*` be deleted outright after porting, or kept as examples?


## Phase 9: Incremental PR Plan

Goal: Deliver changes safely in small, reviewable chunks.

- [ ] PR 1: Inventory report + contracts + central loader (no removals)
- [ ] PR 2..N: Remove one legacy area at a time; update tests/docs; drop deps
- [ ] Final PR: Remove remaining flags/shims; finalize config cleanup; update changelog
- [ ] Release: Cut a major version if breaking; publish migration notes

## Tooling & Safeguards

- Unused code detector: script to list modules not imported; verify with coverage
- Import boundary check: linter/AST check to prevent core importing plugin concretes
- Deprecation helpers: decorator/context emitting structured warnings with links
- Plugin health check: CLI to validate entry points, config, and loadability

## Success Criteria

- No legacy paths; single source via plugin contracts only
- Clean `pyproject` deps; no libraries used only by removed code
- Tests green with maintained/raised coverage; zero skipped legacy tests
- Docs/examples aligned; config templates free of deprecated keys
- Clear changelog and migration guide; semver-observant release

## Decision Log

- 2025-09-01: Created cleanup plan and progress tracker
- 2025-09-01: Completed Phase 1 inventory and Phase 2 checklist
- 2025-09-01: Removed dead module `ccproxy/utils/models_provider.py`
- 2025-09-01: Pruned root deps (`aiosqlite`, runtime `keyring`, explicit `h2`) and added plugin extras groups
- 2025-09-01: Added centralized plugin loader and import boundary checks; wired loader into app and CLI auth
- 2025-09-01: Removed core compatibility stubs; updated analytics to depend on plugin storage
- 2025-09-01: Phase 6 progress: updated hooks tests (structured logs), analytics/metrics tests (plugin deps), binary resolver tests (env guards + TRACE fallback), scheduler tests (no pushgateway)

## Next Actions

- Phase 4: Remove legacy pushgateway mentions; confirm no remaining core→plugin aliases; delete any conditional adapter selection
- Phase 5: Add README Plugin Config Quickstart and migration messages for deprecated keys
- Phase 6: Exclude `tests_new/*` in CI, add integration tests for discovery/metrics/analytics; raise coverage gate modestly
- Phase 8: Wire `check-boundaries`, ruff/mypy to CI with failing gates
