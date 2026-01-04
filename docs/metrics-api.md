# Metrics API Documentation

This document describes metrics endpoints provided by the metrics plugin. The metrics surface is plugin-owned and mounted when the plugin is enabled.

## Overview

The metrics system provides comprehensive monitoring and analytics capabilities for the CCProxy API. It tracks request/response patterns, performance metrics, cost analytics, error monitoring, and usage statistics.

## Base URL

Core metrics endpoints are available under the `/metrics` prefix:

```
/metrics/*
```

## Authentication

Metrics endpoints follow the same authentication requirements as the main API.

## Endpoints

### 1. Metrics Status

Get the status of the metrics system.

**Endpoint:** `GET /metrics`

**Response:**
```json
{
  "status": "metrics endpoint available"
}
```

### 2. Get Metrics Data

Retrieve metrics data with filtering and pagination support.

**Endpoint:** `GET /metrics/data`

**Query Parameters:**
- `start_time` (datetime, optional): Start time for metrics query
- `end_time` (datetime, optional): End time for metrics query  
- `metric_type` (MetricType, optional): Filter by metric type
- `user_id` (string, optional): Filter by user ID
- `session_id` (string, optional): Filter by session ID
- `limit` (integer, optional): Maximum number of records (1-10000, default: 1000)
- `offset` (integer, optional): Number of records to skip (default: 0)

**Response:**
```json
{
  "data": [
    {
      "id": "uuid",
      "timestamp": "2025-07-12T10:30:00Z",
      "metric_type": "request",
      "request_id": "req-123",
      "user_id": "user-456",
      "session_id": "session-789",
      "metadata": {},
      // ... additional fields based on metric type
    }
  ],
  "pagination": {
    "total_count": 5000,
    "returned_count": 100,
    "offset": 0,
    "limit": 100,
    "has_next": true,
    "has_previous": false,
    "next_offset": 100,
    "previous_offset": null
  },
  "filters": {
    "start_time": "2025-07-12T00:00:00Z",
    "end_time": "2025-07-12T23:59:59Z",
    "metric_type": "request",
    "user_id": "user-456",
    "session_id": "session-789"
  }
}
```

### 3. Get Metrics Summary

Get aggregated metrics summary for analysis.

**Endpoint:** `GET /metrics/summary`

**Query Parameters:**
- `start_time` (datetime, optional): Start time for summary
- `end_time` (datetime, optional): End time for summary
- `user_id` (string, optional): Filter by user ID
- `session_id` (string, optional): Filter by session ID

**Response:**
```json
{
  "time_period": {
    "start_time": "2025-07-12T00:00:00Z",
    "end_time": "2025-07-12T23:59:59Z"
  },
  "request_metrics": {
    "total_requests": 1000,
    "successful_requests": 950,
    "failed_requests": 50,
    "error_rate": 0.05
  },
  "response_metrics": {
    "avg_response_time_ms": 750.5,
    "p95_response_time_ms": 1200.0,
    "p99_response_time_ms": 2500.0
  },
  "token_metrics": {
    "total_input_tokens": 50000,
    "total_output_tokens": 75000,
    "total_tokens": 125000
  },
  "cost_metrics": {
    "total_cost": 25.75,
    "avg_cost_per_request": 0.02575
  },
  "usage_patterns": {
    "unique_users": 25,
    "peak_requests_per_minute": 45
  },
  "model_usage": {
    "claude-3-5-sonnet-20241022": 800,
    "claude-3-haiku-20240307": 200
  },
  "error_types": {
    "rate_limit_exceeded": 30,
    "authentication_failed": 15,
    "internal_server_error": 5
  }
}
```

### 4. Stream Metrics (SSE)

Stream real-time metrics via Server-Sent Events.

**Endpoint:** `GET /metrics/stream`

**Query Parameters:**
- `metric_types` (array of MetricType, optional): Metric types to subscribe to
- `user_id` (string, optional): Filter by user ID
- `session_id` (string, optional): Filter by session ID
- `subscription_types` (array of string, optional): Subscription types (default: ["live"])
  - `live`: Real-time individual metrics
  - `summary`: Periodic aggregated summaries
  - `time_series`: Time-series data points

**Response:** Server-Sent Events stream
```
Content-Type: text/event-stream

data: {"event": "metric", "data": {"metric_type": "request", ...}}

data: {"event": "summary", "data": {"total_requests": 100, ...}}

data: {"event": "heartbeat", "timestamp": "2025-07-12T10:30:00Z"}
```

### 5. Get SSE Connections Info

Get information about active SSE connections.

**Endpoint:** `GET /metrics/sse/connections`

**Response:**
```json
{
  "total_connections": 5,
  "max_connections": 100,
  "connections": [
    {
      "id": "conn-123",
      "created_at": "2025-07-12T10:25:00Z",
      "filters": {
        "metric_types": ["request", "response"],
        "user_id": "user-456"
      },
      "subscription_types": ["live", "summary"]
    }
  ]
}
```

### 6. Metrics Health

Get health status of the metrics system.

**Endpoint:** `GET /metrics/health`

**Response:**
```json
{
  "healthy": true,
  "storage": {
    "healthy": true,
    "total_metrics": 10000,
    "last_update": "2025-07-12T10:29:30Z"
  },
  "collector": {
    "healthy": true,
    "metrics_collected": 10000,
    "buffer_size": 50,
    "last_flush": "2025-07-12T10:29:00Z"
  },
  "sse": {
    "healthy": true,
    "connections": 5,
    "max_connections": 100
  }
}
```

## Data Models

### MetricType Enum

Available metric types:

```python
class MetricType(str, Enum):
    REQUEST = "request"      # Incoming request metrics
    RESPONSE = "response"    # Outgoing response metrics  
    ERROR = "error"          # Error and exception metrics
    COST = "cost"           # Cost calculation metrics
    LATENCY = "latency"     # Latency and timing metrics
    USAGE = "usage"         # Usage aggregation metrics
```

### Base MetricRecord

All metrics inherit from this base model:

```python
class MetricRecord(BaseModel):
    id: UUID                           # Unique metric identifier
    timestamp: datetime                # When the metric was recorded
    metric_type: MetricType           # Type of metric
    request_id: str | None = None     # Associated request ID
    user_id: str | None = None        # User identifier
    session_id: str | None = None     # Session identifier
    metadata: dict[str, Any] = {}     # Additional metadata
```

### RequestMetric

Tracks incoming requests:

```python
class RequestMetric(MetricRecord):
    metric_type: MetricType = MetricType.REQUEST

    # Request details
    method: str                       # HTTP method (GET, POST, etc.)
    path: str                        # Request path
    endpoint: str                    # Matched endpoint pattern
    api_version: str                 # API version used

    # Client information
    client_ip: str | None = None     # Client IP address
    user_agent: str | None = None    # User agent string

    # Request characteristics
    content_length: int | None = None # Request body size
    content_type: str | None = None   # Content type

    # Model and provider information
    model: str | None = None          # AI model requested
    provider: str | None = None       # 'anthropic' or 'openai'

    # Request parameters
    max_tokens: int | None = None     # Maximum tokens requested
    temperature: float | None = None  # Temperature parameter
    streaming: bool = False           # Whether streaming was requested
```

### ResponseMetric

Tracks outgoing responses:

```python
class ResponseMetric(MetricRecord):
    metric_type: MetricType = MetricType.RESPONSE

    # Response details
    status_code: int                  # HTTP status code
    response_time_ms: float          # Total response time
    content_length: int | None = None # Response body size
    content_type: str | None = None   # Response content type

    # Token usage
    input_tokens: int | None = None   # Input tokens consumed
    output_tokens: int | None = None  # Output tokens generated
    cache_read_tokens: int | None = None    # Cache read tokens
    cache_write_tokens: int | None = None   # Cache write tokens

    # Streaming information
    streaming: bool = False           # Whether response was streamed
    first_token_time_ms: float | None = None      # Time to first token
    stream_completion_time_ms: float | None = None # Stream completion time

    # Quality metrics
    completion_reason: str | None = None    # Why completion stopped
    safety_filtered: bool = False           # Whether content was filtered
```

### ErrorMetric

Tracks errors and exceptions:

```python
class ErrorMetric(MetricRecord):
    metric_type: MetricType = MetricType.ERROR

    # Error details
    error_type: str                   # Type of error
    error_code: str | None = None     # Error code if available
    error_message: str | None = None  # Error message
    stack_trace: str | None = None    # Stack trace for debugging

    # Context
    endpoint: str | None = None       # Endpoint where error occurred
    method: str | None = None         # HTTP method
    status_code: int | None = None    # HTTP status code

    # Recovery information
    retry_count: int = 0              # Number of retries attempted
    recoverable: bool = False         # Whether error is recoverable
```

### CostMetric

Tracks cost calculations and comparisons:

```python
class CostMetric(MetricRecord):
    metric_type: MetricType = MetricType.COST

    # Cost breakdown (calculated by cost calculator)
    input_cost: float = 0.0           # Input token cost
    output_cost: float = 0.0          # Output token cost
    cache_read_cost: float = 0.0      # Cache read cost
    cache_write_cost: float = 0.0     # Cache write cost
    total_cost: float = 0.0           # Total calculated cost

    # SDK-provided cost information (for comparison)
    sdk_total_cost: float | None = None       # SDK reported total cost
    sdk_input_cost: float | None = None       # SDK reported input cost
    sdk_output_cost: float | None = None      # SDK reported output cost
    sdk_cache_read_cost: float | None = None  # SDK reported cache read cost
    sdk_cache_write_cost: float | None = None # SDK reported cache write cost

    # Cost comparison
    cost_difference: float | None = None      # calculated_cost - sdk_cost
    cost_accuracy_percentage: float | None = None # Calculation accuracy

    # Pricing model
    model: str                        # AI model used
    pricing_tier: str | None = None   # Pricing tier if applicable
    currency: str = "USD"             # Currency code

    # Token counts (for validation)
    input_tokens: int = 0             # Input tokens used
    output_tokens: int = 0            # Output tokens generated
    cache_read_tokens: int = 0        # Cache read tokens
    cache_write_tokens: int = 0       # Cache write tokens
```

### LatencyMetric

Tracks detailed latency and timing information:

```python
class LatencyMetric(MetricRecord):
    metric_type: MetricType = MetricType.LATENCY

    # Timing breakdown
    request_processing_ms: float = 0.0    # Request processing time
    claude_api_call_ms: float = 0.0       # Claude API call time
    response_processing_ms: float = 0.0   # Response processing time
    total_latency_ms: float = 0.0         # Total end-to-end latency

    # Queue and waiting times
    queue_time_ms: float = 0.0            # Time spent in queue
    wait_time_ms: float = 0.0             # Additional wait time

    # Streaming metrics
    first_token_latency_ms: float | None = None   # Time to first token
    token_generation_rate: float | None = None    # Tokens per second
```

### UsageMetric

Tracks usage patterns and aggregations:

```python
class UsageMetric(MetricRecord):
    metric_type: MetricType = MetricType.USAGE

    # Usage counts
    request_count: int = 1            # Number of requests in window
    token_count: int = 0              # Total tokens in window

    # Time window
    window_start: datetime            # Window start time
    window_end: datetime              # Window end time
    window_duration_seconds: float    # Window duration

    # Aggregation level
    aggregation_level: str = "hourly" # hourly, daily, weekly, monthly
```

### MetricsSummary

Aggregated summary of metrics over a time period:

```python
class MetricsSummary(BaseModel):
    # Time period
    start_time: datetime              # Summary period start
    end_time: datetime                # Summary period end

    # Request metrics
    total_requests: int = 0           # Total requests
    successful_requests: int = 0      # Successful requests
    failed_requests: int = 0          # Failed requests
    error_rate: float = 0.0           # Error rate (0.0-1.0)

    # Response metrics
    avg_response_time_ms: float = 0.0     # Average response time
    p95_response_time_ms: float = 0.0     # 95th percentile response time
    p99_response_time_ms: float = 0.0     # 99th percentile response time

    # Token metrics
    total_input_tokens: int = 0       # Total input tokens
    total_output_tokens: int = 0      # Total output tokens
    total_tokens: int = 0             # Total tokens

    # Cost metrics
    total_cost: float = 0.0           # Total cost
    avg_cost_per_request: float = 0.0 # Average cost per request

    # Usage patterns
    unique_users: int = 0             # Number of unique users
    peak_requests_per_minute: int = 0 # Peak requests per minute

    # Model distribution
    model_usage: dict[str, int] = {}  # Usage count by model

    # Error breakdown
    error_types: dict[str, int] = {}  # Error count by type
```

## Error Handling

All endpoints return standard HTTP status codes:

- `200 OK`: Successful request
- `400 Bad Request`: Invalid parameters
- `401 Unauthorized`: Authentication required
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

Error responses follow this format:
```json
{
  "detail": "Error description"
}
```

## Rate Limiting

Metrics endpoints are subject to the same rate limiting as other API endpoints. For high-frequency monitoring, consider using the SSE streaming endpoint instead of polling.

## Best Practices

1. **Filtering**: Use time-based filtering to reduce response sizes and improve performance
2. **Pagination**: Use appropriate page sizes for large datasets
3. **Streaming**: Use SSE for real-time monitoring instead of frequent polling
4. **Caching**: Implement client-side caching for summary data that doesn't change frequently
5. **Monitoring**: Monitor the metrics system health regularly using the `/metrics/health` endpoint

## Examples

### Basic Usage

```bash
# Get recent request metrics
curl "https://api.example.com/metrics/data?metric_type=request&limit=100"

# Get metrics summary for the last hour
curl "https://api.example.com/metrics/summary?start_time=2025-07-12T09:00:00Z&end_time=2025-07-12T10:00:00Z"

# Stream live metrics
curl -N "https://api.example.com/metrics/stream?metric_types=request,response"
```

### Advanced Filtering

```bash
# Get error metrics for a specific user
curl "https://api.example.com/metrics/data?metric_type=error&user_id=user-123&start_time=2025-07-12T00:00:00Z"

# Get cost metrics with pagination
curl "https://api.example.com/metrics/data?metric_type=cost&limit=50&offset=100"
```

### SSE Streaming

```javascript
const eventSource = new EventSource('/metrics/stream?metric_types=request,response');

eventSource.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Received metric:', data);
};

eventSource.onerror = function(event) {
    console.error('SSE error:', event);
};
```
