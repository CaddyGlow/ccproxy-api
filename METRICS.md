# Metrics and Observability System

This document describes the modern observability system in CCProxy, which provides operational monitoring via Prometheus metrics and business event tracking through structured logging.

## Overview

CCProxy uses a hybrid observability architecture that separates operational metrics from business events:

- **Operational Metrics**: Real-time system performance via Prometheus (request counts, response times, error rates)
- **Business Events**: Structured logging for request flows, cost tracking, and historical analysis
- **Request Context**: Correlation and timing across all observability data

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Request Context │───→│ Prometheus       │───→│ /metrics/       │
│ Manager         │    │ Metrics          │    │ prometheus      │
│                 │    │                  │    │                 │
│                 │    └──────────────────┘    └─────────────────┘
│                 │  
│                 │    ┌──────────────────┐    ┌─────────────────┐
│                 │───→│ Structured       │───→│ Storage         │
│                 │    │ Logging          │    │ Pipeline        │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Components

### PrometheusMetrics

The `PrometheusMetrics` class provides operational monitoring with native prometheus_client integration:

```python
from ccproxy.observability import get_metrics

metrics = get_metrics()

# Record operational metrics
metrics.record_request("POST", "/v1/messages", "claude-3-sonnet", "200")
metrics.record_response_time(1.5, "claude-3-sonnet", "/v1/messages")
metrics.record_tokens(150, "input", "claude-3-sonnet")
metrics.record_cost(0.0023, "claude-3-sonnet")
metrics.record_error("timeout_error", "/v1/messages")

# Active request tracking
metrics.inc_active_requests()
metrics.dec_active_requests()
```

**Available Metrics:**
- `ccproxy_requests_total` - Total requests by method, endpoint, model, status
- `ccproxy_response_duration_seconds` - Response time histogram
- `ccproxy_tokens_total` - Token counts by type (input/output) and model
- `ccproxy_cost_total` - Cost tracking by model and type
- `ccproxy_errors_total` - Error counts by type and endpoint
- `ccproxy_active_requests` - Current active request count

### Request Context Management

The request context system provides correlation and timing across all observability data:

```python
from ccproxy.observability import request_context, timed_operation

# Basic request context with timing
async with request_context(method="POST", model="claude-3-sonnet") as ctx:
    # Automatic timing and correlation
    assert ctx.request_id  # Unique correlation ID
    assert ctx.duration_ms  # Real-time duration

    # Add metadata during request
    ctx.add_metadata(tokens_input=150, status_code=200)

    # Timed operations within request
    async with timed_operation("api_call", ctx.request_id):
        # API call timing automatically logged
        pass
```

**Features:**
- Unique correlation IDs for request tracking
- High-precision timing with `time.perf_counter()`
- Automatic structured logging (request_start, request_success, request_error)
- Metadata accumulation throughout request lifecycle
- Context tracking for active request monitoring

### Structured Logging Pipeline

Business events are captured through structured logging with automatic storage pipeline:

```python
import structlog

logger = structlog.get_logger()

# Events automatically processed by pipeline
logger.info("api_request_complete",
    request_id="req_123",
    duration_ms=1500,
    tokens_input=150,
    tokens_output=75,
    cost_total=0.0023,
    model="claude-3-sonnet"
)
```

The pipeline automatically converts structured log events to storage metrics for historical analysis.

## Usage Patterns

### Basic Request Handling

```python
from ccproxy.observability import get_metrics, request_context

async def handle_api_request(request_data):
    metrics = get_metrics()

    async with request_context(
        method="POST",
        endpoint="messages",
        model=request_data.model
    ) as ctx:
        # Record start metrics
        metrics.inc_active_requests()
        metrics.record_request("POST", "messages", request_data.model, "pending")

        try:
            # Process request
            response = await process_request(request_data)

            # Record success metrics
            metrics.record_response_time(ctx.duration_seconds, request_data.model, "messages")
            metrics.record_tokens(response.input_tokens, "input", request_data.model)
            metrics.record_tokens(response.output_tokens, "output", request_data.model)
            metrics.record_cost(response.cost, request_data.model)

            return response

        except Exception as e:
            # Record error metrics
            metrics.record_error(type(e).__name__, "messages", request_data.model)
            raise

        finally:
            metrics.dec_active_requests()
```

### Advanced Timing Operations

```python
from ccproxy.observability import request_context, timed_operation

async with request_context(model="claude-3-sonnet") as ctx:
    # Multiple timed operations within request
    async with timed_operation("input_validation", ctx.request_id):
        validate_input(request_data)

    async with timed_operation("claude_api_call", ctx.request_id):
        response = await call_claude_api(request_data)

    async with timed_operation("response_processing", ctx.request_id):
        formatted_response = format_response(response)

    # All operations automatically logged with timing
    return formatted_response
```

## Configuration

### Environment Variables

```bash
# Optional: Enable verbose observability logging
CCPROXY_VERBOSE_STREAMING=true
CCPROXY_VERBOSE_API=true
```

### Prometheus Client

The system gracefully handles missing prometheus_client dependency:

```python
# Installation (optional)
pip install prometheus-client

# Without prometheus_client, metrics operations are no-ops
# System continues to function with structured logging only
```

### Storage Pipeline

The pipeline processor can be configured for different storage backends:

```python
from ccproxy.observability.pipeline import StoragePipelineProcessor

# Custom storage configuration
processor = StoragePipelineProcessor(
    batch_size=100,
    flush_interval=30.0,  # seconds
    storage_backend=custom_storage
)
```

## Endpoints

### Prometheus Metrics

**GET /metrics/prometheus**

Returns metrics in Prometheus format for scraping:

```
# HELP ccproxy_requests_total Total number of requests
# TYPE ccproxy_requests_total counter
ccproxy_requests_total{method="POST",endpoint="messages",model="claude-3-sonnet",status="200"} 1542

# HELP ccproxy_response_duration_seconds Response time in seconds
# TYPE ccproxy_response_duration_seconds histogram
ccproxy_response_duration_seconds_bucket{model="claude-3-sonnet",endpoint="messages",le="0.1"} 12
ccproxy_response_duration_seconds_bucket{model="claude-3-sonnet",endpoint="messages",le="0.5"} 45
ccproxy_response_duration_seconds_bucket{model="claude-3-sonnet",endpoint="messages",le="1.0"} 123
```

Headers:
- `Content-Type: text/plain; version=0.0.4; charset=utf-8`
- `Cache-Control: no-cache, no-store, must-revalidate`

**GET /metrics/status**

Returns observability system status:

```json
{
  "status": "healthy",
  "prometheus_enabled": true,
  "active_requests": 3,
  "total_requests": 1542
}
```

## Integration

### ProxyService Integration

The ProxyService automatically uses the new observability system:

```python
from ccproxy.api.dependencies import get_proxy_service

proxy_service = get_proxy_service(settings, credentials_manager)

# ProxyService.metrics is now PrometheusMetrics instance
assert hasattr(proxy_service, "metrics")
assert isinstance(proxy_service.metrics, PrometheusMetrics)
```

### Dependency Injection

Observability components are available through FastAPI dependencies:

```python
from ccproxy.api.dependencies import get_observability_metrics
from ccproxy.observability import PrometheusMetrics

@app.get("/custom-endpoint")
async def custom_endpoint(metrics: PrometheusMetrics = Depends(get_observability_metrics)):
    metrics.record_request("GET", "custom", "system", "200")
    return {"status": "ok"}
```

## Migration from Legacy System

The new observability system replaces the previous `ccproxy.metrics` module:

### Removed Components
- `MetricsCollector` class
- Complex storage abstraction layer
- Custom ORM and correlation system
- SQLite storage implementation (~1200 lines)
- Memory storage implementation
- Base storage classes

### Replaced With
- Direct prometheus_client integration
- Structured logging with structlog
- Simple storage pipeline
- Request context management
- ~90% code reduction (from 2000+ to ~200 lines)

### Breaking Changes
- `metrics_collector` parameter removed from ProxyService
- `/metrics/collector/*` endpoints removed
- Custom metrics storage format no longer supported
- MetricsCollector dependency injection removed

## Performance

The new system provides significant performance improvements:

- **Direct Prometheus Integration**: No intermediate abstraction layers
- **Minimal Overhead**: Graceful degradation when prometheus_client unavailable  
- **Efficient Context Tracking**: Thread-safe operations with minimal locks
- **Precision Timing**: Uses `time.perf_counter()` for accurate measurements
- **Reduced Memory**: No complex in-memory correlation tracking

## Monitoring and Alerting

### Prometheus Queries

Common queries for monitoring:

```promql
# Request rate
rate(ccproxy_requests_total[5m])

# Error rate
rate(ccproxy_errors_total[5m]) / rate(ccproxy_requests_total[5m])

# 95th percentile response time
histogram_quantile(0.95, rate(ccproxy_response_duration_seconds_bucket[5m]))

# Active requests
ccproxy_active_requests

# Token consumption rate
rate(ccproxy_tokens_total[5m])

# Cost per minute
rate(ccproxy_cost_total[1m]) * 60
```

### Recommended Alerts

```yaml
# High error rate
- alert: HighErrorRate
  expr: rate(ccproxy_errors_total[5m]) / rate(ccproxy_requests_total[5m]) > 0.1
  for: 2m

# High response time
- alert: HighResponseTime
  expr: histogram_quantile(0.95, rate(ccproxy_response_duration_seconds_bucket[5m])) > 5
  for: 1m

# High active requests
- alert: HighActiveRequests
  expr: ccproxy_active_requests > 100
  for: 1m
```

## Development

### Testing

The observability system includes comprehensive tests:

```bash
# Run observability tests
pytest tests/test_observability.py -v

# Test specific components
pytest tests/test_observability.py::TestPrometheusMetrics -v
pytest tests/test_observability.py::TestRequestContext -v
```

### Adding New Metrics

To add new operational metrics:

```python
# In PrometheusMetrics class
def record_custom_metric(self, value: float, labels: dict[str, str]) -> None:
    """Record custom operational metric."""
    if not self._enabled:
        return

    if not hasattr(self, '_custom_metric'):
        self._custom_metric = Counter(
            f"{self.namespace}_custom_total",
            "Custom metric description",
            list(labels.keys()),
            registry=self._registry
        )

    self._custom_metric.labels(**labels).inc(value)
```

### Adding Business Events

To add new business event logging:

```python
import structlog

logger = structlog.get_logger()

async def business_operation():
    logger.info("business_event",
        event_type="custom_operation",
        user_id="user_123",
        operation_result="success",
        processing_time_ms=250
    )
```

## Troubleshooting

### Prometheus Client Not Available

If prometheus_client is not installed:

```python
# System continues with structured logging only
# Install to enable Prometheus metrics:
pip install prometheus-client
```

### Missing Metrics Data

Check observability system status:

```bash
curl http://localhost:8000/metrics/status
```

### High Memory Usage

The new system uses minimal memory. If issues persist:

1. Check for metric label cardinality
2. Verify storage pipeline configuration
3. Monitor active request count

### Performance Issues

The observability system adds minimal overhead:

1. Prometheus operations are atomic
2. Context tracking is thread-safe
3. Structured logging is asynchronous
4. Storage pipeline batches operations

For performance-critical scenarios, prometheus_client can be disabled while maintaining structured logging.
