# Metrics Plugin

Collects Prometheus-style metrics for CCProxy events and optionally pushes them.

## Highlights
- Registers a metrics hook to observe request, token, error, and pool events
- Exposes a `/metrics` endpoint for scraping when enabled
- Can schedule a Pushgateway task for off-box metrics delivery

## Configuration
- `MetricsConfig` toggles collection, namespace, endpoint, and pushgateway options
- Scheduler integration is automatic when push mode is enabled
- Generate defaults with `python3 scripts/generate_config_from_model.py \
  --format toml --plugin metrics --config-class MetricsConfig`

```toml
[plugins.metrics]
# enabled = true
# namespace = "ccproxy"
# metrics_endpoint_enabled = true
# pushgateway_enabled = false
# pushgateway_job = "ccproxy"
# pushgateway_push_interval = 60
# collect_request_metrics = true
# collect_token_metrics = true
# collect_cost_metrics = true
# collect_error_metrics = true
# collect_pool_metrics = true
# histogram_buckets = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0]
```

## Related Components
- `hook.py`: implements the metrics collector and event handling
- `routes.py`: builds the FastAPI router for Prometheus scrape format
- `tasks.py`: Pushgateway task invoked by the scheduler
