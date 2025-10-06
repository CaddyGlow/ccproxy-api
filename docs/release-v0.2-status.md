# Release Readiness – v0.2.0

## Blocking Issues
- Version metadata still marked as development builds (`ccproxy/core/_version.py:31`; `CHANGELOG.md:8`). Tag `v0.2.0`, regenerate `_version.py`, and update the changelog date before packaging.
- Packaging depends on a local placeholder plugin (`pyproject.toml:22`; `uv.lock:561-575`). Remove the `dummy` dependency and refresh `uv.lock` or publish the plugin properly.
- MkDocs navigation references missing documents (`docs/migration/0.2-plugin-first.md`, `docs/OAUTH_PLUGIN_ARCHITECTURE.md`). Restore the sources or update navigation/README links so docs build succeeds.

## Major Work Remaining
- Installation docs still point to dev branches (`README.md:155-159`; `docs/getting-started/installation.md:18-24`). Update once the official 0.2.0 artefacts exist.
- Backups and temporary files linger (.bak files under `ccproxy/`, `.ccproxy.toml.bak`, `scripts/test_endpoint.sh.20250913-110546.bak`). Remove them so they do not ship in sdists.
- `uv.lock` pins unreleased versions (e.g., `duckdb 1.3.2` dated 2025-07-08). Re-sync dependencies against published wheels and re-run the lock.

## Outstanding TODOs – Detailed Triage
- **ccproxy/api/app.py:546** – The startup sequence still performs format-chain validation inline. Refactor this block into `MiddlewareManager` or a dedicated validator in `ccproxy/core/middleware.py`, then invoke it from startup. Add regression coverage via a unit test that exercises `MiddlewareManager.apply_to_app` with a mocked format registry to ensure validation happens before middleware application.
- **ccproxy/api/routes/health.py:9** – While the handlers manually set the `Content-Type`, FastAPI may still advertise `application/json` in the OpenAPI schema. Define `responses={200: {"content": {"application/health+json": {}}}}` or set `response_class=PlainTextResponse` with explicit serialization to guarantee the IETF media type. Add a contract test that hits `/health` and asserts `response.headers["content-type"] == "application/health+json"`.
- **ccproxy/services/adapters/http_adapter.py:577** – Streaming responses bypass the reverse format chain, so downstream clients may receive provider-native payloads. Implement a reversible SSE/JSON chunk transformer (possibly reusing `ccproxy.llms.streaming.accumulators`) and cover it with integration tests for both Anthropic→OpenAI and OpenAI→Anthropic streaming paths.
- **ccproxy/plugins/claude_api/detection_service.py:105** / **ccproxy/plugins/codex/detection_service.py:112** – Detection logic cannot be exercised without the respective CLIs. Introduce feature-flagged test fixtures that monkeypatch `_get_*_version` to raise `FileNotFoundError`, assert the fallback JSON is returned, and document in release notes that CLI-less environments rely on packaged defaults.
- **ccproxy/plugins/copilot/oauth/provider.py:319** – The `copilot_access` flag should reflect profile capabilities. Extend `CopilotProfile` parsing to surface access entitlements and populate the flag when fetching `get_user_profile()`. Include a unit test that loads a mocked profile with business access and confirms the flag is true.
- **ccproxy/plugins/docker/protocol.py:24** – The protocol omits discovery helpers (`get_version`, `image_info`, `list_images`). Either expand the protocol and the concrete adapter to implement them or prune the TODO and document the current surface so typed plugins do not rely on undefined methods.
- **ccproxy/plugins/credential_balancer/README.md:83** – The README still lists “cooldown TODO”. Either deliver dynamic cooldown handling (parsing `Retry-After` headers, plumbing through the hook) or rewrite the section to describe the current manual cooldown behaviour.
- **ccproxy/templates/plugin_scaffold.py:287** – The scaffold intentionally raises `503` with TODO messaging. Decide whether to ship a production-ready minimal adapter (e.g., echo passthrough) or update the README so new plugin authors understand they must replace the stub before shipping.

## Minor Follow-Ups
- Hook implementation docstring promises hooks that are not exported (`ccproxy/core/plugins/hooks/implementations/__init__.py:7`). Align the documentation with the available implementations or include the missing hooks.
- Confirm final release copy for the changelog and docs (e.g., target release date, OAuth migration material) before tagging.

## Suggested Next Steps
1. Clean up packaging (drop `dummy`, update lockfile, verify wheels build).
2. Repair documentation sources and navigation, then run `uv run mkdocs build` and `make ci`.
3. Finalize release metadata: set the changelog date, regenerate `_version.py`, tag `v0.2.0`, and rebuild artefacts.
