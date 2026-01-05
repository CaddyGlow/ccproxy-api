# CCProxy API Server

CCProxy is a local, plugin-based reverse proxy that unifies access to
multiple AI providers (e.g., Claude SDK/API and OpenAI Codex) behind a
consistent API. It ships with bundled plugins for providers, logging,
tracing, metrics, analytics, and more.

## Supported Providers

- Anthropic Claude API/SDK (OAuth2 flow or Claude CLI/SDK token files)
- OpenAI Codex (ChatGPT backend Responses API using OAuth for paid/pro accounts)
- GitHub Copilot (chat and completions for free, paid, or business accounts)

Each provider adapter exposes the same surface area: OpenAI Chat
Completions, OpenAI Responses, and Anthropic Messages. The proxy maintains a
shared model-mapping layer so you can reuse the same `model` identifier
across providers without rewriting client code.

Authentication can reuse existing provider files (e.g., Claude CLI SDK
tokens and the Codex CLI credential store), or you can run
`ccproxy auth login <provider>` to complete the OAuth flow from the CLI;
stored secrets are picked up automatically by the proxy.

## Extensibility

CCProxy's plugin system lets you add instrumentation and storage layers
without patching the core server. Bundled plugins currently include:

- [`access_log`](ccproxy/plugins/access_log/README.md): structured access
  logging for client and provider traffic
- [`analytics`](ccproxy/plugins/analytics/README.md): DuckDB-backed analytics
  APIs for captured request logs
- [`claude_api`](ccproxy/plugins/claude_api/README.md): Anthropic Claude HTTP
  API adapter with health and metrics
- [`claude_sdk`](ccproxy/plugins/claude_sdk/README.md): local Claude CLI/SDK
  adapter with session pooling
- [`codex`](ccproxy/plugins/codex/README.md): OpenAI Codex provider adapter
  with OAuth support
- [`command_replay`](ccproxy/plugins/command_replay/README.md): generates
  `curl`/`xh` commands for captured requests
- [`copilot`](ccproxy/plugins/copilot/README.md): GitHub Copilot provider
  adapter with OAuth token management
- [`credential_balancer`](ccproxy/plugins/credential_balancer/README.md):
  rotates upstream credentials based on health
- [`dashboard`](ccproxy/plugins/dashboard/README.md): serves the CCProxy
  dashboard SPA and APIs
- [`docker`](ccproxy/plugins/docker/README.md): runs providers inside Docker via
  CLI extensions
- [`duckdb_storage`](ccproxy/plugins/duckdb_storage/README.md): exposes
  DuckDB-backed storage for logs and analytics
- [`max_tokens`](ccproxy/plugins/max_tokens/README.md): normalizes
  `max_tokens` fields to provider limits
- [`metrics`](ccproxy/plugins/metrics/README.md): Prometheus-compatible metrics
  with optional Pushgateway
- [`oauth_claude`](ccproxy/plugins/oauth_claude/README.md): standalone OAuth
  provider for Claude integrations
- [`oauth_codex`](ccproxy/plugins/oauth_codex/README.md): standalone OAuth
  provider for Codex integrations
- [`permissions`](ccproxy/plugins/permissions/README.md): interactive approval
  flow for privileged tool actions
- [`pricing`](ccproxy/plugins/pricing/README.md): caches model pricing data for
  cost-aware features
- [`request_tracer`](ccproxy/plugins/request_tracer/README.md): detailed
  request/response tracing for debugging

Shared helpers such as
[`claude_shared`](ccproxy/plugins/claude_shared/README.md) provide metadata
consumed by the Claude plugins. Each plugin directory contains its own README
with configuration examples.

## Quick Links

- Docs site entry: `docs/index.md`
- Getting started: `docs/getting-started/quickstart.md`
- Configuration reference: `docs/getting-started/configuration.md`
- Examples: `docs/examples.md`
- Migration (0.2): `docs/migration/0.2-plugin-first.md`

## Plugin Config Quickstart

The plugin system is enabled by default (`enable_plugins = true`), and all
discovered plugins load automatically when no additional filters are set. Use
these knobs to adjust what runs:

- `enabled_plugins`: optional allow list; when set, only the listed plugins run.
- `disabled_plugins`: optional block list applied when `enabled_plugins` is not
  set.
- `plugins.<name>.enabled`: per-plugin flag (defaults to `true`) that you can
  override in TOML or environment variables. Any plugin set to `false` is added
  to the deny list alongside `disabled_plugins` during startup.

During startup we merge `disabled_plugins` and any `plugins.<name>.enabled = false`
entries into a single deny list. At runtime the loader checks the allow list
first and then confirms the plugin is not deny listed. Configure plugins under
`plugins.<name>` in TOML or via nested environment variables.

Use `ccproxy plugins list` to inspect discovered plugins and
`ccproxy plugins settings <name>` to review configuration fields.

### TOML example (`.ccproxy.toml`)

```toml
enable_plugins = true
# enabled_plugins = ["metrics", "analytics"]  # Optional allow list
disabled_plugins = ["duckdb_storage"]          # Optional block list

[plugins.access_log]
client_enabled = true
client_format = "structured"
client_log_file = "/tmp/ccproxy/access.log"

[plugins.request_tracer]
json_logs_enabled = true
raw_http_enabled = true
log_dir = "/tmp/ccproxy/traces"

[plugins.duckdb_storage]
enabled = false

[plugins.analytics]
enabled = true

# Metrics plugin
[plugins.metrics]
enabled = true
# pushgateway_enabled = true
# pushgateway_url = "http://localhost:9091"
# pushgateway_job = "ccproxy"
# pushgateway_push_interval = 60
```

### Environment variables (nested with `__`)

```bash
export DISABLED_PLUGINS="duckdb_storage"      # Optional block list
export PLUGINS__ACCESS_LOG__ENABLED=true
export PLUGINS__ACCESS_LOG__CLIENT_ENABLED=true
export PLUGINS__ACCESS_LOG__CLIENT_FORMAT=structured
export PLUGINS__ACCESS_LOG__CLIENT_LOG_FILE=/tmp/ccproxy/access.log

export PLUGINS__REQUEST_TRACER__ENABLED=true
export PLUGINS__REQUEST_TRACER__JSON_LOGS_ENABLED=true
export PLUGINS__REQUEST_TRACER__RAW_HTTP_ENABLED=true
export PLUGINS__REQUEST_TRACER__LOG_DIR=/tmp/ccproxy/traces

export PLUGINS__DUCKDB_STORAGE__ENABLED=true
export PLUGINS__ANALYTICS__ENABLED=true
export PLUGINS__METRICS__ENABLED=true
# export PLUGINS__METRICS__PUSHGATEWAY_ENABLED=true
# export PLUGINS__METRICS__PUSHGATEWAY_URL=http://localhost:9091
```

## Running

To install the latest stable release without cloning the repository, use `uvx`
to grab the published wheel and launch the CLI:

```bash
uvx --with "ccproxy-api[all]" ccproxy serve --port 8000
```

If you prefer `pipx`, install the package (optionally with extras) and use the
local shim:

```bash
pipx install "ccproxy-api[all]"
ccproxy serve  # default on localhost:8000
```

## License

See `LICENSE`.
