# Metrics Plugin

A Prometheus metrics collection and export plugin for CCProxy that provides operational monitoring through an event-driven hook system.

## Features

- **Event-Driven Collection**: Uses the hook system to collect metrics from request/response lifecycle events
- **Prometheus Export**: Exposes metrics in Prometheus format at `/metrics` endpoint
- **Pushgateway Support**: Optional integration with Prometheus Pushgateway for batch metrics
- **Comprehensive Metrics**: Tracks requests, response times, tokens, costs, errors, and connection pools
- **Grafana Dashboards**: Pre-configured dashboards for visualization

## Configuration

The plugin is configured through the metrics configuration section:

```python
# Enable/disable the plugin
enabled: bool = True

# Prometheus namespace prefix
namespace: str = "ccproxy"

# Enable /metrics endpoint
metrics_endpoint_enabled: bool = True

# Pushgateway configuration
pushgateway_enabled: bool = False
pushgateway_url: str | None = None
pushgateway_job: str = "ccproxy"
pushgateway_push_interval: int = 60  # seconds

# Collection settings
collect_request_metrics: bool = True
collect_token_metrics: bool = True
collect_cost_metrics: bool = True
collect_error_metrics: bool = True
collect_pool_metrics: bool = True

# Response time histogram buckets (seconds)
histogram_buckets: list[float] = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0]
```

## Metrics Collected

### Request Metrics
- `ccproxy_requests_total`: Total number of requests (labels: method, endpoint, model, status, service_type)
- `ccproxy_response_duration_seconds`: Response time histogram (labels: model, endpoint, service_type)
- `ccproxy_active_requests`: Current number of active requests

### Token Metrics
- `ccproxy_tokens_total`: Total tokens processed (labels: type, model, service_type)
  - Types: input, output, cache_read, cache_write

### Cost Metrics
- `ccproxy_cost_usd_total`: Total cost in USD (labels: model, cost_type, service_type)

### Error Metrics
- `ccproxy_errors_total`: Total errors (labels: error_type, endpoint, model, service_type)

### Pool Metrics
- `ccproxy_pool_clients_total`: Total clients in pool
- `ccproxy_pool_clients_available`: Available clients
- `ccproxy_pool_clients_active`: Active clients
- `ccproxy_pool_connections_created_total`: Total connections created
- `ccproxy_pool_connections_closed_total`: Total connections closed
- `ccproxy_pool_acquisitions_total`: Total client acquisitions
- `ccproxy_pool_releases_total`: Total client releases
- `ccproxy_pool_health_check_failures_total`: Health check failures
- `ccproxy_pool_acquisition_duration_seconds`: Time to acquire client

### System Metrics
- `ccproxy_info`: System information (version, metrics_enabled)
- `up`: Service health indicator

## Hook Events

The plugin subscribes to these hook events:
- `REQUEST_STARTED`: Begin tracking request, increment active requests
- `REQUEST_COMPLETED`: Record metrics, decrement active requests
- `REQUEST_FAILED`: Record error metrics
- `PROVIDER_REQUEST_SENT`: Track provider requests
- `PROVIDER_RESPONSE_RECEIVED`: Record provider response metrics
- `PROVIDER_ERROR`: Record provider errors
- `PROVIDER_STREAM_START/CHUNK/END`: Track streaming metrics

## Grafana Integration

Pre-configured dashboards are included in the `grafana/` directory:
- `dashboards/ccproxy-dashboard.json`: Main operational dashboard
- `provisioning/`: Grafana provisioning configuration

To use the dashboards:
1. Copy the provisioning files to your Grafana provisioning directory
2. Restart Grafana to load the dashboards
3. Configure VictoriaMetrics or Prometheus as a data source

## Pushgateway Integration

For environments where Prometheus scraping isn't available, enable Pushgateway:

```python
pushgateway_enabled: True
pushgateway_url: "http://localhost:9091"
pushgateway_job: "ccproxy"
pushgateway_push_interval: 60
```

The plugin will automatically push metrics at the configured interval.

## Architecture

The plugin follows a clean, event-driven architecture:

1. **MetricsHook**: Subscribes to lifecycle events and updates metrics
2. **PrometheusMetrics**: Thread-safe metric collection using prometheus_client
3. **PushgatewayClient**: Optional batch metric export with circuit breaker
4. **Routes**: FastAPI endpoints for metric export
5. **Plugin Runtime**: Manages lifecycle, hook registration, and background tasks

## Testing

```bash
# Run plugin tests
pytest tests/unit/plugins/test_metrics.py -v

# Test metrics endpoint
curl http://localhost:8000/metrics

# Test health endpoint
curl http://localhost:8000/metrics/health
```

## Migration from Built-in Metrics

This plugin replaces the built-in metrics system in `ccproxy/observability/metrics.py`. The migration is transparent:
- Same metric names and labels are maintained
- Compatible with existing Grafana dashboards
- No changes required to Prometheus scraping configuration
