# Pricing Plugin

Caches model pricing data and exposes it to other plugins for cost awareness.

## Highlights
- Loads pricing catalogs and keeps them fresh via the update task
- Publishes a `pricing` service in the plugin registry for dependents
- Tracks cache health, age, and failures for health reporting

## Configuration
- `PricingConfig` toggles enablement, refresh cadence, and startup behavior
- Auto-update schedules can force refresh on launch or run periodically
- Generate defaults with `python3 scripts/generate_config_from_model.py \
  --format toml --plugin pricing --config-class PricingConfig`

```toml
[plugins.pricing]
# enabled = true
# cache_dir = "~/.cache/ccproxy"
# cache_ttl_hours = 24
# source_url = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
# download_timeout = 30
# auto_update = true
# memory_cache_ttl = 300
# update_interval_hours = 6.0
# force_refresh_on_startup = false
# fallback_to_embedded = false
# pricing_provider = "all"
```

## Related Components
- `service.py`: pricing lookup and cache management
- `tasks.py`: asynchronous cache refresh task
- `plugin.py`: runtime lifecycle and service registration
