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
- [ ] Phase 1: Inventory & Baseline
- [ ] Phase 2: Dead Code & Redundancy Detection
- [ ] Phase 3: Boundaries & Interfaces
- [ ] Phase 4: Redundant Code Removal
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
- [ ] Ensure discovery via entry points only; remove ad-hoc registries/shims
- [ ] Import hygiene: prevent core → plugin concrete imports
- [ ] Centralize plugin loading; remove scattered loaders/feature flags
- [ ] Deliverable: `ccproxy/plugins/loader.py` and interface modules finalized

## Phase 4: Redundant Code Removal

Goal: Remove duplicated core implementations; keep interfaces + orchestration only.

- [ ] Delete core code duplicated by plugins (adapters/services)
- [ ] Remove transitional shims that conditionally prefer plugins
- [ ] Remove feature flags toggling legacy vs plugin paths
- [ ] Drop dependencies used only by removed code (update `pyproject`)
- [ ] Trim obsolete config keys and add migration-friendly validation errors

## Phase 5: API, CLI, and Config Surface

Goal: Expose a single plugin-driven surface; remove legacy-only routes/commands.

- [ ] Ensure API routes delegate via plugin adapters only
- [ ] Remove legacy endpoints and CLI subcommands
- [ ] Namespace plugin settings (e.g., `plugins.<name>.*`) with defaults/examples
- [ ] Update `config.example.toml` and `.env.example`

## Phase 6: Tests Refactor

Goal: Align tests to plugin-first architecture with strong coverage.

- [ ] Remove or rewrite tests for deleted core code
- [ ] Move provider-specific tests under plugin packages
- [ ] Standardize plugin fixtures/mocks; avoid concrete plugin imports in core tests
- [ ] Integration tests for discovery and end-to-end flows
- [ ] Maintain/raise coverage thresholds

## Phase 7: Docs & Communication

Goal: Clarify architecture, contracts, and migration path.

- [ ] Architecture docs: plugin anatomy, lifecycle, and discovery
- [ ] Migration guide: map core X → plugin Y, config changes, deprecations
- [ ] README/examples: plugin-first usage; update env vars and commands
- [ ] Changelog: deprecations/removals with versions/dates

## Phase 8: CI, Linting, and Type Safety

Goal: Guard boundaries and keep quality gates strong.

- [ ] CI job to import only public plugin interfaces from core
- [ ] Tighten `ruff` rules for import layering and ordering
- [ ] Enforce `mypy` across core and bundled plugins
- [ ] Keep pre-commit hooks fast; remove legacy-specific checks

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

## Next Actions

- Kick off Phase 1 inventory and produce `docs/cleanup/inventory.md`
- Run `make check` and `make test-coverage` to capture baseline
- Propose interface `Protocol`s and central loader skeleton for review
