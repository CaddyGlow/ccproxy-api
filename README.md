CCProxy API Server

CCProxy is a local, plugin‑based reverse proxy that unifies access to multiple AI providers (e.g., Claude SDK/API and OpenAI Codex) behind a consistent API. It ships with bundled plugins for providers, logging, tracing, metrics, analytics, and more.

Quick Links
- Docs site entry: `docs/index.md`
- Getting started: `docs/getting-started/quickstart.md`
- Configuration reference: `docs/getting-started/configuration.md`
- Examples: `docs/examples.md`
- Migration (0.2): `docs/migration/0.2-plugin-first.md`

Plugin Config Quickstart

Enable plugins and configure them under `plugins.<name>` in TOML or via nested environment variables.

TOML example (`.ccproxy.toml`):

```toml
enable_plugins = true

[plugins.access_log]
enabled = true
client_enabled = true
client_format = "structured"
client_log_file = "/tmp/ccproxy/access.log"

[plugins.request_tracer]
enabled = true
json_logs_enabled = true
raw_http_enabled = true
log_dir = "/tmp/ccproxy/traces"

[plugins.duckdb_storage]
enabled = true

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

Environment variables (nested with `__`):

```bash
export ENABLE_PLUGINS=true
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

Running

```bash
ccproxy serve  # default on localhost:8000
```

License

See `LICENSE`.
