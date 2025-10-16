# Credential Balancer (system plugin)

The credential balancer manages pools of upstream credentials (API keys, OAuth tokens, etc.) for a given provider and rotates between them based on health. It integrates as a system plugin and exposes a registry key (auth manager) that provider plugins can use to fetch a currently healthy credential at request time.

- Balances across multiple credential files per provider.
- Detects failures from HTTP responses and temporarily disables bad credentials with cooldowns.
- Supports manual refresh, proportional selection, sticky-on-success, and backoff.
- Exposes a named auth manager registry key (defaults to `<provider>_credential_balancer`).

## When to use

Use the balancer when you have multiple tokens for the same provider and want resilient failover and automatic rotation without changing application code or secrets storage.

## Quick Start (minimal)

The following minimal example configures CCproxy with the Codex provider to use a pool of Codex OAuth tokens.

```toml
[plugins]
# Enable the credential balancer system plugin
enabled_plugins = [
   "codex",
   "oauth_codex",
   "credential_balancer"
]

# Point the codex provider at the balancer-managed auth manager
[plugins.codex]
auth_manager = "codex_credential_balancer"

[[plugins.credential_balancer.providers]]
provider = "codex"
strategy = "round_robin" # or "failover"

manager_class = "ccproxy.plugins.oauth_codex.manager.CodexTokenManager"
storage_class = "ccproxy.plugins.oauth_codex.storage.CodexTokenStorage"

credentials = [
  { path = "~/.config/ccproxy/codex_plus.json" },
  { path = "~/.config/ccproxy/codex_pro.json" },
]

```
## Full Configuration Reference

Enable the system plugin and define one or more provider pools. Each pool declares where to read credentials from and optional tuning parameters. See `config.example.toml` for full, commented examples.

```toml
[[plugins.credential_balancer.providers]]
# Provider identifier, e.g. "claude-api", "openai", "codex".
provider = "claude-api"
strategy = "round_robin"             # or "failover"
max_failures_before_disable = 2
cooldown_seconds = 120.0
failure_status_codes = [401, 403]

# Pool defaults (example: Claude OAuth manager/storage)
manager_class = "ccproxy.plugins.oauth_claude.manager.ClaudeApiTokenManager"
storage_class = "ccproxy.plugins.oauth_claude.storage.ClaudeOAuthStorage"

credentials = [
  { type = "manager", file = "~/.config/ccproxy/claude_primary.json", label = "primary" },
  { type = "manager", file = "~/.config/ccproxy/claude_backup.json", label = "backup" },
]
```

After defining a pool, point the corresponding provider plugin at the balancer by overriding its auth manager to the registry key:

```toml
[plugins.claude-api]
# Use the balancer-provided registry entry instead of a static key file
auth_manager = "claude-api_credential_balancer"
```

If you set a custom `manager_name` in the balancer configuration, use that value for `auth_manager` instead.

## How it works

- Startup: for each entry in `[[plugins.credential_balancer.providers]]`, the plugin constructs a Manager that loads credentials from the declared files and registers it under `manager_name`.
- Request path: provider adapters ask the registry for a credential via the `auth_manager` key; the balancer selects a currently healthy token.
- Feedback loop: the `credential_balancer` hook observes provider HTTP responses and records failures/successes to update health, handle cooldowns, and trigger failover when necessary.

## TODO

- Extract cooldown period from provider error responses and apply dynamic per-credential cooldowns.
  - Collect and parse HTTP error payloads/headers in the hook (e.g., Retry-After or equivalent fields).
  - Pass an optional cooldown override with the failure event to the manager.
  - Ensure logs include the derived cooldown value for observability.

## Logs and observability

The plugin emits structured events to aid troubleshooting, including (non-exhaustive):
- `credential_balancer_manager_registered`
- `credential_balancer_token_selected`
- `credential_balancer_failure_detected`
- `credential_balancer_failover`
- `credential_balancer_manual_refresh_succeeded`

During development, server logs stream to `/tmp/ccproxy/ccproxy.log` when running `ccproxy serve`.

## Files and APIs

- Runtime code: `ccproxy/plugins/credential_balancer/`
  - `plugin.py`: plugin factory and lifecycle wiring
  - `manager.py`: rotation, health, selection, and feedback processing
  - `hook.py`: HTTP lifecycle hook that feeds response outcomes back to the manager
  - `config.py`: Pydantic models for pool configuration and defaults
- Enable via `pyproject.toml` entry point `credential_balancer` (already wired).

## Testing

- Unit tests: `tests/plugins/credential_balancer/unit/`
- Run fast tests: `./Taskfile test-unit`
- Full suite: `./Taskfile test`

Follow the project’s testing markers and async patterns as described in `TESTING.md`.

## Further reading

- Authentication overview: `docs/user-guide/authentication.md`
- Example configuration: `config.example.toml`

Commands
- `uv run ccproxy serve` (logs at `/tmp/ccproxy/ccproxy.log`)
