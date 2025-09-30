# Credential Balancer Plugin

The `credential_balancer` system plugin manages pools of upstream credentials for a
provider. It composes multiple auth managers, exposes them through a single
registry name, and automatically rotates between them when a token starts
failing. Enable it when you maintain several Claude or OpenAI tokens and want
hands-off failover instead of manually rewriting configuration files.

## When to Use It

- Keep a hot spare token available if a primary credential is revoked or paused.
- Spread traffic across several rate-limited seats without custom routing code.
- Combine credentials produced by other auth managers with JSON exports.
- Prepare a staged credential, then cut over by moving it to the top of the pool.

## Runtime Architecture

Each configured provider builds one `CredentialBalancerTokenManager`. During
startup the plugin:

1. Instantiates a dedicated balancer for every entry in
   `[[plugins.credential_balancer.providers]]`.
2. Registers the manager with the global `AuthManagerRegistry` using
   `manager_name` (defaults to `<provider>_credential_balancer`).
3. Attaches the `credential_balancer` hook so provider HTTP responses feed back
   into the active manager.

The hook listens for `HTTP_RESPONSE` and `HTTP_ERROR` events. When a provider
request completes it forwards the `request_id` and status code to the manager.
Failing responses (default: `401` or `403`) increment the credential's failure
count and, once thresholds are met, trigger a cooldown before the token is used
again.

## Supported Credential Sources

### Manager-based credentials

Manager sources use provider-specific auth manager classes for proper type-safe
credential loading. You can specify managers either by:

1. **Direct class specification** (recommended): Specify fully qualified class
   names for maximum flexibility and no registry dependency.

2. **Registry lookup**: Reference a manager already registered in the runtime
   via `manager_key`.

Manager-based credentials support additional configuration options:

- **Storage options:**
  - `enable_backups` (bool): Create timestamped backups before overwriting
    credentials (default: `true`)

- **Manager options:**
  - `credentials_ttl` (float): Seconds to cache credentials before rechecking
    storage (default: `30.0`)
  - `refresh_grace_seconds` (float): Seconds before expiry to trigger proactive
    token refresh (default: `120.0`)

You may freely interleave file and manager entries. They are evaluated in the
order provided.

## Configuration Reference

### Basic Configuration (Recommended)

Use manager-based credentials with pool-level class defaults for clean configuration:

```toml
[plugins.credential_balancer]
enabled = true

[[plugins.credential_balancer.providers]]
provider = "codex"
strategy = "round_robin"             # or "failover"
max_failures_before_disable = 2
cooldown_seconds = 120.0
failure_status_codes = [401, 403]

# Pool-level defaults applied to all credentials
manager_class = "ccproxy.plugins.oauth_codex.manager.CodexTokenManager"
storage_class = "ccproxy.plugins.oauth_codex.storage.CodexTokenStorage"

credentials = [
  { type = "manager", file = "~/.config/ccproxy/codex_pro.json", label = "codex_pro" },
  { type = "manager", file = "~/.config/ccproxy/codex_plus.json", label = "codex_plus" },
]
```

#### Provider-Specific Manager and Storage Classes

Use these class names for different providers:

**OpenAI Codex:**
- Manager: `ccproxy.plugins.oauth_codex.manager.CodexTokenManager`
- Storage: `ccproxy.plugins.oauth_codex.storage.CodexTokenStorage`

**Claude API (OAuth):**
- Manager: `ccproxy.plugins.oauth_claude.manager.ClaudeApiTokenManager`
- Storage: `ccproxy.plugins.oauth_claude.storage.ClaudeOAuthStorage`

**GitHub Copilot:**
- Manager: `ccproxy.plugins.copilot.manager.CopilotTokenManager`
- Storage: `ccproxy.plugins.copilot.oauth.storage.CopilotOAuthStorage`

### Advanced Configuration Options

```toml
[[plugins.credential_balancer.providers]]
provider = "codex"
manager_class = "ccproxy.plugins.oauth_codex.manager.CodexTokenManager"
storage_class = "ccproxy.plugins.oauth_codex.storage.CodexTokenStorage"

credentials = [
  # Advanced config with storage and manager tuning
  { type = "manager",
    file = "~/.config/ccproxy/codex_pro.json",
    config = {
      enable_backups = true,            # Create timestamped backups (default: true)
      credentials_ttl = 60.0,           # Cache credentials for 60s (default: 30.0)
      refresh_grace_seconds = 300.0,    # Refresh 5min before expiry (default: 120.0)
    },
    label = "codex_pro"
  },

  # Override pool-level classes for specific credential
  { type = "manager",
    file = "~/.config/ccproxy/codex_plus.json",
    manager_class = "custom.manager.CustomManager",
    storage_class = "custom.storage.CustomStorage",
    label = "codex_plus"
  },
]
```

### Registry-Based Manager Lookup

```toml
[[plugins.credential_balancer.providers]]
provider = "claude-api"
strategy = "failover"
credentials = [
  # Use manager_key to reference registered managers
  { type = "manager", manager_key = "claude-api", file = "~/.config/ccproxy/claude_primary.json" },
  { type = "manager", manager_key = "claude-api", file = "~/.config/ccproxy/claude_backup.json" },
]
```


### Configuration Options

#### Pool-Level Options

- `enabled`: Disable the entire plugin without deleting provider entries.
- `provider`: Identifier expected by downstream adapters (for example
  `claude-api`, `claude_sdk`, or `codex`).
- `manager_name`: Registry name for the composed manager. Defaults to
  `<provider>_credential_balancer`.
- `manager_class`: Default manager class for all credentials in the pool (can be
  overridden per credential).
- `storage_class`: Default storage class for all credentials in the pool (can be
  overridden per credential).
- `strategy`: Selection policy. `failover` sticks with the first healthy entry,
  while `round_robin` rotates on every request.
- `max_failures_before_disable`: Consecutively failing responses required to
  mark a credential unavailable.
- `cooldown_seconds`: Time to keep a failed credential paused. Use `0` to retry
  immediately or a large value to require manual intervention.
- `failure_status_codes`: Sorted list of HTTP codes treated as credential
  failures. Add `429` if you wish to fail over when providers enforce soft
  throttles.

#### Credential Source Options

**Common fields:**
- `type`: Must be `"manager"` (default, can be omitted).
- `label`: Optional friendly name used in logs and metrics.

**Manager credential fields:**
- `file`: Path to custom credential file (supports `~` and environment variables).
- `manager_class`: Fully qualified manager class name (overrides pool default).
- `storage_class`: Fully qualified storage class name (overrides pool default).
- `manager_key`: Alternative to `manager_class` for registry-based lookup.
- `config`: Optional dict with advanced settings:
  - `enable_backups` (bool): Create timestamped backups (default: `true`)
  - `credentials_ttl` (float): Cache credentials for N seconds (default: `30.0`)
  - `refresh_grace_seconds` (float): Refresh N seconds before expiry (default:
    `120.0`)

## Rotation Strategies

### Failover

`failover` prioritises the first healthy credential. The manager only advances
when a response returns one of the configured failure status codes the required
number of times. After the cooldown expires the credential is eligible again.
This is ideal for keeping a warm backup without splitting traffic.

### Round robin

`round_robin` cycles through the credential list for every new request. Entries
that enter cooldown are skipped until their timer resets. Pick this strategy
when you need to divide load evenly across several accounts.

## Integrating with Provider Plugins

After defining a pool, point the relevant provider at the balancer's registry
name:

```toml
[plugins.claude_api]
auth_manager = "claude-api_credential_balancer"
```

If you override `manager_name`, reference that value instead. No further changes
are required for built-in adapters—they automatically include the `request_id`
field consumed by the balancer hook. Custom adapters must forward the request
context so the hook can correlate provider responses with the credential in use.

## Monitoring and Maintenance

- Logs include events such as `credential_balancer_failure_detected`,
  `credential_balancer_failover`, and
  `credential_balancer_manual_refresh_succeeded` to aid troubleshooting.
- Call `ccproxy auth status <provider> --file <path>` to inspect a snapshot
  before adding it to the pool.
- Use `cooldown_seconds = 0` during development to retry the same file
  immediately while you iterate on credentials.
- Periodically run `CredentialBalancerTokenManager.cleanup_expired_requests()`
  in custom tooling if you build long-lived background jobs that never touch the
  HTTP hook system.

For a quick-start walkthrough covering exports and pool wiring see the
[Authentication guide](../user-guide/authentication.md#rotating-multiple-credential-files).
