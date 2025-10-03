# DuckDB Storage Plugin

Provides DuckDB-backed storage for analytics and request logging data.

## Highlights
- Initializes a DuckDB database and exposes it via the plugin registry
- Creates directories automatically and reuses the configured database path
- Optionally runs VACUUM/OPTIMIZE on shutdown for compactness

## Configuration
- `DuckDBStorageConfig` toggles enablement, database path, and optimizations
- Other plugins reference the exposed `log_storage` service by name
- Generate defaults with `python3 scripts/generate_config_from_model.py \
  --format toml --plugin duckdb_storage --config-class DuckDBStorageConfig`

```toml
[plugins.duckdb_storage]
# enabled = true
# database_path = "~/.local/share/ccproxy/metrics.duckdb"
# optimize_on_shutdown = false
```

## Related Components
- `plugin.py`: runtime lifecycle and service registration
- `storage.py`: `SimpleDuckDBStorage` helper for connections
- `routes.py`: FastAPI router under `/duckdb` for simple diagnostics
