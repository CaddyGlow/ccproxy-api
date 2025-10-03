# Dashboard Plugin

Serves the CCProxy dashboard SPA and supporting APIs.

## Highlights
- Mounts static assets for the dashboard when available on disk
- Registers dashboard routes for health, session, and telemetry views
- Integrates with FastAPI app mounting during plugin initialization

## Configuration
- `DashboardPluginConfig` toggles static asset mounting and route exposure
- Defaults to auto-mounting assets under `/dashboard/assets` when present
- Generate defaults with `python3 scripts/generate_config_from_model.py \
  --format toml --plugin dashboard --config-class DashboardPluginConfig`

```toml
[plugins.dashboard]
# enabled = true
# mount_static = true
```

## Related Components
- `plugin.py`: runtime for mounting static files
- `routes.py`: FastAPI router for dashboard APIs
- `config.py`: settings model for plugin toggles
