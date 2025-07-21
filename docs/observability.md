# Observability

`ccproxy` includes a comprehensive observability system to provide insights into the proxy's performance, usage, and health. The system is built on a hybrid architecture that combines real-time Prometheus metrics, structured logging, and an optional DuckDB-based data store for historical analytics.

## Features

-   **Prometheus Metrics:** Exposes a `/metrics` endpoint for real-time operational monitoring.
-   **Access Logs:** Captures detailed logs for every request, including token counts, costs, and timing.
-   **Log Storage:** Persists access logs to a local DuckDB database for historical querying and analysis.
-   **Query API:** Provides endpoints to query and analyze stored access logs.
-   **Real-time Dashboard:** A web-based dashboard to visualize metrics and logs in real-time.
-   **Pushgateway Support:** Can push metrics to a Prometheus Pushgateway for environments where scraping is not feasible.

## Configuration

Observability features are configured under the `observability` section in your configuration file or through environment variables with the `OBSERVABILITY__` prefix.

| Setting                     | Environment Variable                | Default                               | Description                                                                                             |
| --------------------------- | ----------------------------------- | ------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `metrics_endpoint_enabled`  | `OBSERVABILITY__METRICS_ENDPOINT_ENABLED` | `False`                               | Enable the `/metrics` endpoint for Prometheus scraping.                                                 |
| `logs_endpoints_enabled`    | `OBSERVABILITY__LOGS_ENDPOINTS_ENABLED`   | `False`                               | Enable the `/logs/*` endpoints for querying and analytics.                                              |
| `dashboard_enabled`         | `OBSERVABILITY__DASHBOARD_ENABLED`        | `False`                               | Enable the `/dashboard` endpoint.                                                                       |
| `logs_collection_enabled`   | `OBSERVABILITY__LOGS_COLLECTION_ENABLED`  | `False`                               | Enable storing access logs to the backend.                                                              |
| `log_storage_backend`       | `OBSERVABILITY__LOG_STORAGE_BACKEND`      | `duckdb`                              | The storage backend for logs (`duckdb` or `none`).                                                      |
| `duckdb_path`               | `OBSERVABILITY__DUCKDB_PATH`              | `~/.local/share/ccproxy/metrics.duckdb` | The path to the DuckDB database file.                                                                   |
| `pushgateway_url`           | `OBSERVABILITY__PUSHGATEWAY_URL`          | `None`                                | The URL for the Prometheus Pushgateway.                                                                 |

### Enabling Features

To enable all observability features, you can set the following in your `.env` file:

```
OBSERVABILITY__METRICS_ENDPOINT_ENABLED=true
OBSERVABILITY__LOGS_ENDPOINTS_ENABLED=true
OBSERVABILITY__DASHBOARD_ENABLED=true
OBSERVABILITY__LOGS_COLLECTION_ENABLED=true
```

## Prometheus Metrics

When enabled, the `/metrics` endpoint exposes a wide range of metrics in Prometheus format.

### Key Metrics

-   `ccproxy_requests_total`: Total number of requests (labels: `method`, `endpoint`, `model`, `status`, `service_type`).
-   `ccproxy_response_duration_seconds`: Histogram of response times (labels: `model`, `endpoint`, `service_type`).
-   `ccproxy_tokens_total`: Total number of tokens processed (labels: `type`, `model`, `service_type`).
-   `ccproxy_cost_usd_total`: Total estimated cost in USD (labels: `model`, `cost_type`, `service_type`).
-   `ccproxy_errors_total`: Total number of errors (labels: `error_type`, `endpoint`, `model`, `service_type`).
-   `ccproxy_active_requests`: Gauge of currently active requests.

## Access Logs & Storage

When `logs_collection_enabled` is `true`, the proxy captures detailed information for each request and stores it in a DuckDB database. This allows for historical analysis of usage patterns, costs, and performance.

### Log Schema

The `access_logs` table stores the following columns:

-   `request_id`
-   `timestamp`
-   `method`, `endpoint`, `path`, `query`
-   `client_ip`, `user_agent`
-   `service_type`, `model`, `streaming`
-   `status_code`, `duration_ms`, `duration_seconds`
-   `tokens_input`, `tokens_output`, `cache_read_tokens`, `cache_write_tokens`
-   `cost_usd`, `cost_sdk_usd`

## Logs API Endpoints

When `logs_endpoints_enabled` is `true`, the following endpoints become available under the `/logs` prefix:

-   `GET /logs/status`: Get the status of the observability system.
-   `GET /logs/query`: Query access logs with filters.
-   `GET /logs/analytics`: Get aggregated analytics from the logs.
-   `GET /logs/stream`: Stream logs in real-time via Server-Sent Events (SSE).
-   `GET /logs/entries`: Get raw log entries from the database.
-   `POST /logs/reset`: Clear all stored log data.

## Dashboard

When `dashboard_enabled` is `true`, a real-time web dashboard is available at the `/dashboard` endpoint. The dashboard provides a live view of requests, token usage, costs, and errors.
