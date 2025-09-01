# Phase 2 — Dead Code & Redundancy Detection

Last updated: 2025-09-01

Purpose: Identify dead/redundant code and config left from the plugin migration, and produce a remove/replace/deprecate checklist.

## Summary of Findings

- Dead module: `ccproxy/utils/models_provider.py` — 0% coverage, no references across repo.
- Root deps likely plugin-only: `duckdb`, `duckdb-engine`, `sqlmodel`, `prometheus-client`, `textual` (kept for now as plugins are bundled).
- Unused root deps: `aiosqlite`, `keyring` (docs-only), `h2` (no direct imports; http2 via `httpx[http2]`).
- Low-coverage but used: `ccproxy/adapters/openai/response_adapter.py`, `ccproxy/adapters/openai/streaming.py`, `ccproxy/core/transformers.py`, `ccproxy/core/http_client_hooks.py`.
- API/CLI duplication: none found beyond plugin-mounted routes; core exposes only health/plugins routes.
- Config: `SchedulerSettings` no longer has Pushgateway fields (moved to metrics plugin). Tests referring to `pushgateway_*` expect legacy config. Plan test updates or plugin-aware feature flags.
- Tests: `tests_new/*` reference removed/old interfaces; not part of active test suite and cause type errors.

## Evidence

- Coverage (coverage.xml): `utils/models_provider.py` → line-rate 0.0; `adapters/openai/response_adapter.py` ~0.13; `adapters/openai/streaming.py` ~0.146; `core/transformers.py` ~0.246; `core/http_client_hooks.py` ~0.176.
- Static scans: No in-repo references to `utils/models_provider.py`. OpenAI adapter is referenced across plugins and tests.
- Dependency scans: No imports of `aiosqlite`, `keyring`, or `h2` in core/plugins/tests.

## Checklist (Decision / Item / Rationale)

- [ ] REMOVE: `ccproxy/utils/models_provider.py`
  - Rationale: Unreferenced and uncovered. Model listing belongs in provider plugins.
  - Action: Delete file and remove any export (none present). Changelog entry.

- [ ] DEPRECATE → MOVE LATER: OpenAI adapter location
  - Rationale: Actively used by multiple plugins and tests. Consider moving to a shared internal package later, not now.
  - Action: Keep in core for now; revisit after plugin packages can depend on a shared library.

- [ ] KEEP (Refactor later): `core/http_client_hooks.py`, `core/transformers.py`
  - Rationale: Low coverage but in use through `http_client` and adapters. Hardening/typing to be addressed in later refactor.

- [ ] CONFIG CLEANUP: Scheduler Pushgateway (legacy)
  - Rationale: Functionality moved to metrics plugin; tests expect old settings.
  - Action: Update tests to target metrics plugin; ensure Settings does not carry legacy fields; add migration note.

- [ ] PYPROJECT PRUNE: Root runtime dependencies
  - [ ] REMOVE: `aiosqlite` (unused)
  - [ ] REMOVE: `keyring` from runtime; keep in optional `security` group only (already present)
  - [ ] REMOVE: explicit `h2` (no direct import; `httpx[http2]` pulls transitively)
  - [ ] NOTE: mark as plugin-only (keep for now): `duckdb`, `duckdb-engine`, `sqlmodel`, `prometheus-client`, `textual`

- [ ] TEST SUITE HYGIENE: `tests_new/*`
  - Rationale: Targets removed interfaces and causes type failures.
  - Action: Exclude from CI/type-check or delete after porting tests to v2 APIs.

## Proposed Migration Notes (Docs/Changelog)

- Scheduler pushgateway settings were removed; use Metrics plugin configuration instead.
- Model listing endpoints are provided by provider plugins; `utils/models_provider.py` removed.
- Root dependencies trimmed: `aiosqlite`, `keyring` (runtime), `h2`.

## Next Steps (Phase 3+4 tie-in)

- Confirm removals with maintainers.
- Implement removals in a small PR: delete `utils/models_provider.py`, prune pyproject deps, add changelog.
- Prepare test updates for scheduler/metrics configuration and plugin-centric paths.
- Add optional extras in packaging to prepare future split (done):
  - `plugins-storage` → duckdb, duckdb-engine, sqlmodel
  - `plugins-metrics` → prometheus-client
  - `plugins-ui` → textual
