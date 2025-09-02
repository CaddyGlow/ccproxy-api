# CLI Plugin Loading: Problem and Proposed Solution

## Problem Summary

- The auth CLI needed request tracing (request_tracer) for HTTP flows, but importing the plugin directly in `ccproxy/cli/commands/auth.py` couples the CLI to a specific plugin implementation.
- We changed the CLI to initialize the full plugin system so request_tracer can self-register via its normal lifecycle. While this works, it also initializes many unrelated plugins (including provider and storage plugins), which leads to:
  - Errors from plugins that expect the async task manager to be started (e.g., scheduled tasks).
  - DuckDB lock conflicts from metrics/analytics storage initialization during CLI runs.
  - Unnecessary overhead and side effects for simple auth operations.

## Goals

- Enable request tracing automatically for CLI HTTP flows without direct imports or tight coupling.
- Avoid initializing the full plugin ecosystem when running CLI commands; only load what’s required.
- Provide a clean, configurable way to control which plugins the CLI loads.

## Proposed Solution

Introduce a “filtered plugin initialization” path used by CLI flows:

1. Discover all plugin factories using the existing loader.
2. Build a filtered `PluginRegistry` that includes only:
   - The specific auth provider plugin required by the CLI command (e.g., `oauth_claude`, `oauth_codex`).
   - A small allowlist of safe system plugins needed for CLI observability (default: `request_tracer`).
3. Do not include provider plugins (e.g., `claude_api`, `codex`) or plugins that can cause heavy side effects by default.
4. Initialize only this filtered registry for the CLI command, wiring in a `HookRegistry/HookManager` and a hook-enabled HTTP client.

This keeps behavior consistent with the API server’s plugin lifecycle while avoiding unwanted side effects.

## Key Behaviors

- Auth provider registration: The selected auth provider plugin registers its OAuth provider with the CLI’s `OAuthRegistry` as usual.
- Tracing: `request_tracer` initializes via hooks and attaches to the CLI’s HTTP client; no direct imports in the CLI code.
- No server-only assumptions: The filtered path avoids dependencies on FastAPI app state and does not require the task manager or DB storage.

## Configuration

- Add a CLI allowlist for system plugins:
  - Environment variable: `CLI_PLUGINS__ALLOWLIST` (comma-separated), e.g. `request_tracer,metrics`.
  - Optional TOML (nice-to-have):

    ```toml
    [cli.plugins]
    allowlist = ["request_tracer"]
    denylist = []
    ```

- Defaults:
  - Allowlist: `["request_tracer"]`.
  - Hard exclusions: All `ProviderPluginFactory`-based plugins (e.g., `claude_api`, `codex`), unless explicitly allowed in the future.

## Implementation Outline

1. Helper to construct a filtered `PluginRegistry` for CLI:
   - Use `load_plugin_system(settings)` to discover factories.
   - Map provider input to the corresponding auth plugin name (`claude-api` → `oauth_claude`, `codex` → `oauth_codex`).
   - Build a new `PluginRegistry` and register only:
     - The resolved auth provider factory (`AuthProviderPluginFactory`).
     - Allowlisted system plugin factories (`SystemPluginFactory`), defaulting to `request_tracer`.
   - Explicitly skip registering `ProviderPluginFactory` plugins.

2. In the auth CLI bootstrap:
   - Create a `HookRegistry` and `HookManager`, and register `HookManager` into the `ServiceContainer` so dependent services (e.g., `StreamingHandler`) resolve cleanly.
   - Initialize only the filtered registry (`initialize_all(core_services)`).
   - Fetch the OAuth provider from the `OAuthRegistry` after initialization.
   - Normalize provider keys for lookup: `claude_api|claude-api|claude` → `claude-api`, `openai|openai-api|codex` → `codex`.

## Validation Plan

- Run `ccproxy-api auth status claude_api` and `ccproxy-api auth status codex`:
  - Confirm that request_tracer hooks are registered and the HTTP client shows `has_hooks=True` in logs.
  - Ensure no “Task manager is not started” errors and no DuckDB lock errors.
  - Verify logs show only the selected auth provider plugin and `request_tracer` initialized; provider plugins like `claude_api`, `codex` do not initialize.
- Test `CLI_PLUGINS__ALLOWLIST` to include other system plugins if desired (e.g., `access_log`, `metrics`).

## Risks / Trade-offs

- Some users may want more observability in CLI flows (e.g., access logs, metrics). The allowlist addresses this without defaulting to broad loading.
- A few system plugins may assume FastAPI app state; we should keep the default allowlist conservative (`request_tracer`) and require opt-in for others.
- This improves CLI performance and reduces side effects while adding a small amount of code for filtered registry handling.

## Future Work

- Expose CLI flags for toggling tracing verbosity or raw HTTP logging without editing plugin configuration.
- Provide a shared utility (e.g., `core/plugins/cli_loader.py`) to centralize filtered registry creation for any CLI command that needs minimal plugin support.
- Document plugin expectations for CLI compatibility (e.g., tolerate missing app state and task manager; no required background tasks during init).

