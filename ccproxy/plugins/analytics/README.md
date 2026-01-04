# Analytics Plugin

Persists structured access logs and serves query APIs for observability data.

## Highlights
- Ensures DuckDB schemas exist and registers the `access_logs` SQLModel table
- Publishes an ingest service consumed by the access log hook
- Adds `/logs` routes for querying, streaming, and inspecting request history

## Configuration
- `AnalyticsPluginConfig` toggles collection, retention, and debug logging
- Requires the `duckdb_storage` plugin to supply the underlying engine
- Generate defaults with `python3 scripts/generate_config_from_model.py \
  --format toml --plugin analytics --config-class AnalyticsPluginConfig`

```toml
[plugins.analytics]
# enabled = true
```

## Related Components
- `plugin.py`: runtime initialization and service registration
- `ingest.py`: writes events into DuckDB using SQLModel
- `routes.py`: FastAPI router for analytics and log queries
