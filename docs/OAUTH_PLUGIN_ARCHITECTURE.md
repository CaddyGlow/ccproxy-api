# OAuth Plugin Architecture

This document describes how CCProxy structures OAuth-capable plugins and the shared conventions you should follow when adding new providers.

## Core Concepts
- **Token Storage** – Each provider ships a storage backend that persists OAuth credentials (usually JSON files under `~/.config/ccproxy`).
- **Token Manager** – A subclass of `BaseEnhancedTokenManager` responsible for refreshing tokens, exposing `TokenSnapshot` data, and integrating with the CLI.
- **Plugin Runtime** – The plugin factory wires the manager and storage into a `ProviderPluginRuntime` so adapters receive authenticated HTTP clients via dependency injection.
- **Hook Integration** – OAuth managers emit hook events (`HookEvent.OAUTH_TOKEN_REQUEST`, `HookEvent.OAUTH_TOKEN_RESPONSE`, etc.) for observability and analytics.

## File Layout
```
ccproxy/plugins/<provider>/
├── manager.py           # Token manager implementation
├── storage.py           # Credential persistence helpers
├── plugin.py            # Factory + manifest wiring
├── adapter.py           # HTTP adapter using the manager
├── routes.py            # FastAPI routers (if applicable)
├── schemas.py           # Pydantic models for OAuth payloads
└── README.md            # Provider-specific notes
```

## OAuth Flow
1. **CLI Initiation** – `ccproxy auth login <provider>` looks up the plugin manifest and resolves its `OAuthFlow` descriptor.
2. **Device / Browser Grant** – The CLI opens the provider authorization URL (PKCE, device-code, or standard authorization code) and polls for completion using `httpx`.
3. **Credential Storage** – The resulting tokens are serialized via the plugin storage class. Secrets are masked when echoed back to the user.
4. **Runtime Refresh** – On startup the token manager loads credentials, validates expiry, and schedules refresh jobs through the async task manager.
5. **Snapshot Reporting** – Managers expose `TokenSnapshot` for `/auth status`, enabling consistent diagnostics across providers.

## Implementing a New OAuth Plugin
1. **Define Settings** – Create a `Config` model with the provider-specific endpoints and scopes. Register defaults in the plugin manifest.
2. **Implement Storage** – Subclass `BaseJSONTokenStorage` (or equivalent) to read/write credential files. Handle migration of legacy formats in `migrate()`.
3. **Implement Manager** – Extend `BaseEnhancedTokenManager` and implement `_build_token_snapshot()` plus provider-specific refresh logic. Use `AsyncHTTPClientFactory` from the container for outbound requests.
4. **Expose Routes / CLI** – If the provider needs webhook callbacks or additional CLI commands, add them via `manifest.routes` and `manifest.cli_commands`.
5. **Register Plugin** – Export `factory = <YourFactory>()` at module scope so entry-point discovery works.

## Security Recommendations
- Store tokens with `0600` permissions and avoid embedding secrets in configuration files committed to source control.
- Use PKCE (code verifier + challenge) whenever the provider supports it.
- Rotate refresh tokens when the provider issues new ones; do not assume they are long lived.
- Emit structured log events without leaking raw tokens. The `TokenSnapshot` helper masks sensitive fields by default.

## Testing Checklist
- Unit-test storage migrations to ensure existing users keep working after upgrades.
- Mock OAuth endpoints with `pytest-httpx` and cover success, refresh, and error paths.
- Run `./Taskfile test-plugins` with the plugin enabled to verify end-to-end behaviour.
- Exercise CLI flows manually using sandbox credentials before releasing a new provider version.

## Troubleshooting
- Use `ccproxy auth status --provider <name>` to inspect stored credentials and expiry timestamps.
- Enable verbose logging: `LOGGING__LEVEL=debug uv run ccproxy serve`.
- If refresh attempts fail repeatedly, delete the credential file and redo `ccproxy auth login` to obtain a clean grant.
