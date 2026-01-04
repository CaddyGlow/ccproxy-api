# Auth Providers For Developers

This guide explains how CCProxy wires together OAuth providers, token managers,
and the developer-facing tooling that interacts with them. Use it as a reference
when building a new provider, extending credential flows, or debugging auth
issues locally.

## Architecture Overview

- **OAuth providers** implement `OAuthProviderProtocol` and are registered with
  the central `OAuthRegistry`. Each provider encapsulates the HTTP/OAuth dance
  (authorization URLs, device flows, refresh logic, revocation) and exposes a
  storage backend for credentials.
- **Token managers** implement `BaseTokenManager` (or
  `EnhancedTokenManager`) for provider-specific credential lifecycles.
  They perform storage I/O, cache data, and surface high-level helper methods
  used by adapters and the CLI.
- **Token snapshots** (`TokenSnapshot`) provide a safe, uniform view of the
  tokens and metadata the manager currently holds. Snapshots are intentionally
  lightweight so they can be shared with adapters, CLI commands, and future
  monitoring endpoints without leaking provider internals.
- **Adapters, plugins, and CLI commands** depend on the snapshot interface
  instead of probing credentials with `hasattr` checks. This keeps the call
  surface consistent across providers.

```mermaid
flowchart LR
    subgraph Registry & CLI
        CLI[CLI auth commands]
        Registry[OAuthRegistry]
    end

    subgraph Provider Plugin
        Provider[OAuth provider]
        Manager[Token manager]
        Storage[Token storage]
    end

    CLI -->|discover|get_oauth_provider| Registry --> Provider
    Provider -->|load/save| Storage
    Provider -->|create_token_manager| Manager
    Manager -->|snapshot|get_token_snapshot
    CLI -->|render status| Manager

    Adapter[HTTP adapter] -->|resolve token| Manager
```

## Provider Responsibilities

- Implement the `OAuthProviderProtocol` surface fully. In particular:
  - `get_authorization_url`, `handle_callback`, and `refresh_access_token`
    must return provider-specific credential models.
  - `get_storage()` should return a storage instance (subclassing
    `TokenStorage`) ready for read/write operations.
  - `get_credential_summary()` should leverage snapshots if available to avoid
    re-implementing masking logic.
- Expose factory helpers such as `create_token_manager()` when a provider wants
  to hand back a fully configured token manager to the CLI or services.
- Populate `cli` capabilities (device flow, browser flow, manual fallback) so
  the CLI automatically selects the right flow for the environment.

## Token Managers & Snapshots

- New managers should inherit from `BaseTokenManager` or `EnhancedTokenManager`.
- Override `_build_token_snapshot(...)` to map provider models into a
  `TokenSnapshot`. At minimum populate:
  - `provider`: short stable identifier (e.g. `"claude-api"`).
  - `access_token`: raw token string (will be masked by helper methods).
  - `refresh_token`: optional refresh value.
  - `expires_at`: `datetime` describing the current access token expiry.
  - `scopes` and `extras` for provider-specific metadata (plan tier, account
    type, etc.).
- Call `save_credentials()`/`load_credentials()` as usual; the base class now
  uses snapshots for `get_access_token()` and `get_expiration_time()` so your
  manager implementations stay focused on provider logic.
- Use `TokenSnapshot.access_token_preview()` when printing to logs or CLI to
  avoid exposing full secrets.

### Snapshot Helper For Non-Manager Contexts

The CLI includes `_token_snapshot_from_credentials()` to convert credential
models into snapshots even when a manager instance is unavailable (for example,
when a provider returns raw credentials but does not hand back a token manager).
If you add a new provider credential model, extend that helper so developers
keep consistent previews in CLI status output.

## CLI Usage Patterns

- `ccproxy auth login <provider>` discovers the provider via the registry,
  selects a flow based on the provider's CLI configuration, and stores
  credentials using the provider's storage implementation.
- `ccproxy auth status <provider>` now:
  1. Loads credentials through the provider.
  2. Attempts to obtain a token manager and request a snapshot.
  3. Falls back to converting raw credentials into a snapshot if a manager is
     unavailable.
  4. Renders profile information, masked token previews, and troubleshooting
     hints.
- `ccproxy auth logout <provider>` still uses the provider's storage but benefits
  from the shared snapshot logic when printing diagnostics along the way.

### Header Forwarding

- `collect_cli_forward_headers()` in `ccproxy.utils.headers` centralizes the
  allow/deny logic for CLI-detected headers. Adapters should delegate to this
  helper instead of duplicating filtering rules.
- Detection services expose `get_detected_headers()`, `get_ignored_headers()`,
  and `get_redacted_headers()`—the helper respects each list and falls back to
  the raw header snapshot if a custom implementation raises during filtering.
- When forwarding headers, adapters must still block overwriting
  authentication-related values (`Authorization`, `X-API-Key`) after merging the
  helper’s output.

## Testing Tips

- Unit-test snapshot builders with fixture data to ensure masking, scopes, and
  expiry calculations stay stable. See `tests/unit/auth/test_token_snapshots.py`
  for examples covering Claude, OpenAI, and Copilot credentials.
- When tests rely on `Path.home()` (Claude token wrapper reads `~/.claude`),
  mock it to avoid interacting with the developer's real credentials.
- For end-to-end CLI tests, patch provider factories so that login/status flows
  return deterministic snapshot data.

## Local Development Checklist

1. Register your provider in the service container so
   `OAuthRegistry` discovers it (usually in the plugin's `manifest`).
2. Provide a storage implementation (JSON, database, secrets manager) and wire
   it into both the provider and token manager constructors.
3. Implement `_build_token_snapshot` in the manager and update the CLI helper if
   raw credential conversion is needed.
4. Add documentation for any provider-specific CLI flags or environment
   configuration. Development docs should explain how to obtain client IDs,
   secrets, or mock servers for local testing.
5. Run the focused pytest suites under `tests/plugins/<provider>` and the
   shared snapshot tests to confirm behavior before opening a PR.

By following this structure, every auth provider offers a consistent and
type-safe surface area for both adapters and developer tooling, making future
refactors—such as centralized monitoring or stricter static typing—a drop-in
upgrade.
