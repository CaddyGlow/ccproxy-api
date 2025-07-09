## 1. **Metrics Module Structure**

```
ccproxy/
├── metrics/                       # Metrics and statistics module
│   ├── __init__.py
│   ├── collector.py              # Main metrics collector
│   ├── models.py                 # Metric data models
│   ├── storage/                  # Storage backends
│   │   ├── __init__.py
│   │   ├── base.py              # Abstract storage interface
│   │   ├── memory.py            # In-memory storage (dev/testing)
│   │   ├── sqlite.py            # SQLite storage (lightweight)
│   │   ├── postgres.py          # PostgreSQL storage (production)
│   │   └── timeseries.py        # Time-series DB (Prometheus/InfluxDB)
│   ├── exporters/                # Metric exporters
│   │   ├── __init__.py
│   │   ├── prometheus.py        # Prometheus exporter
│   │   ├── json.py              # JSON API exporter
│   │   └── opentelemetry.py     # OpenTelemetry exporter
│   ├── middleware.py             # FastAPI middleware for metrics
│   ├── decorators.py             # Function decorators for metrics
│   └── dashboard/                # Built-in dashboard
│       ├── __init__.py
│       ├── routes.py             # Dashboard API routes
│       └── templates/            # Dashboard HTML templates
```

## 2. **Core Metrics Models**

```python
# metrics/models.py
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum

class MetricType(Enum):
    REQUEST = "request"
    RESPONSE = "response"
    ERROR = "error"
    TOKEN_USAGE = "token_usage"
    COST = "cost"

class RequestMetric(BaseModel):
    """Tracks individual API requests"""
    request_id: str
    timestamp: datetime
    user_id: Optional[str] = None

    # Request details
    method: str
    path: str
    endpoint_type: str  # "anthropic", "openai", "claude_sdk"

    # Client info
    user_agent: Optional[str] = None
    client_ip: Optional[str] = None
    api_key_hash: Optional[str] = None  # Hashed for security

    # Model info
    model: Optional[str] = None
    model_version: Optional[str] = None

    # Performance
    response_time_ms: Optional[float] = None
    status_code: Optional[int] = None

    # Token usage
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    # Cost tracking
    input_cost: Optional[float] = None
    output_cost: Optional[float] = None
    total_cost: Optional[float] = None

    # Error info
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AggregatedMetrics(BaseModel):
    """Aggregated metrics over a time period"""
    period_start: datetime
    period_end: datetime

    # Request stats
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0

    # Performance stats
    avg_response_time_ms: float = 0.0
    p50_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0

    # Token usage stats
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0

    # Cost stats
    total_cost: float = 0.0
    avg_cost_per_request: float = 0.0

    # Breakdown by model
    model_usage: Dict[str, int] = Field(default_factory=dict)
    model_costs: Dict[str, float] = Field(default_factory=dict)

    # Error stats
    error_rate: float = 0.0
    errors_by_type: Dict[str, int] = Field(default_factory=dict)
```

## 3. **Metrics Collector**

```python
# metrics/collector.py
import asyncio
from typing import Optional, Dict, Any
from datetime import datetime
import hashlib

class MetricsCollector:
    """Central metrics collection service"""

    def __init__(
        self,
        storage: MetricStorage,
        cost_calculator: CostCalculator,
        exporters: List[MetricExporter] = None
    ):
        self.storage = storage
        self.cost_calculator = cost_calculator
        self.exporters = exporters or []
        self._buffer = []
        self._flush_interval = 10  # seconds
        self._start_background_tasks()

    async def track_request(
        self,
        request_id: str,
        request: Request,
        endpoint_type: str,
        user: Optional[User] = None
    ) -> RequestMetric:
        """Start tracking a request"""
        metric = RequestMetric(
            request_id=request_id,
            timestamp=datetime.utcnow(),
            user_id=user.id if user else None,
            method=request.method,
            path=request.url.path,
            endpoint_type=endpoint_type,
            user_agent=request.headers.get("user-agent"),
            client_ip=request.client.host if request.client else None,
            api_key_hash=self._hash_api_key(request.headers.get("authorization"))
        )

        # Store in buffer for batch processing
        self._buffer.append(metric)
        return metric

    async def track_response(
        self,
        request_id: str,
        response_time_ms: float,
        status_code: int,
        model: Optional[str] = None,
        tokens: Optional[Dict[str, int]] = None,
        error: Optional[Exception] = None
    ):
        """Update metrics with response data"""
        metric = await self._get_metric(request_id)
        if not metric:
            return

        metric.response_time_ms = response_time_ms
        metric.status_code = status_code
        metric.model = model

        if tokens:
            metric.input_tokens = tokens.get("input_tokens", 0)
            metric.output_tokens = tokens.get("output_tokens", 0)
            metric.total_tokens = metric.input_tokens + metric.output_tokens

            # Calculate costs
            if model:
                costs = self.cost_calculator.calculate(
                    model=model,
                    input_tokens=metric.input_tokens,
                    output_tokens=metric.output_tokens
                )
                metric.input_cost = costs["input_cost"]
                metric.output_cost = costs["output_cost"]
                metric.total_cost = costs["total_cost"]

        if error:
            metric.error_type = type(error).__name__
            metric.error_message = str(error)

        # Export to various backends
        await self._export_metric(metric)

    async def get_stats(
        self,
        start_time: datetime,
        end_time: datetime,
        group_by: Optional[str] = None,  # "hour", "day", "model", "user"
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get aggregated statistics"""
        metrics = await self.storage.query(start_time, end_time, filters)
        return self._aggregate_metrics(metrics, group_by)
```

## 4. **Cost Calculator**

```python
# metrics/cost_calculator.py
class CostCalculator:
    """Calculate costs based on model and token usage"""

    # Pricing per 1M tokens (example rates)
    PRICING = {
        "claude-3-opus": {"input": 15.0, "output": 75.0},
        "claude-3-sonnet": {"input": 3.0, "output": 15.0},
        "claude-3-haiku": {"input": 0.25, "output": 1.25},
        "gpt-4": {"input": 30.0, "output": 60.0},
        "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
    }

    def calculate(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> Dict[str, float]:
        """Calculate costs for token usage"""
        pricing = self._get_pricing(model)

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost
        }
```

## 5. **Metrics Middleware**

```python
# metrics/middleware.py
import time
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

class MetricsMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for automatic metrics collection"""

    def __init__(self, app, metrics_collector: MetricsCollector):
        super().__init__(app)
        self.metrics = metrics_collector

    async def dispatch(self, request: Request, call_next):
        # Generate request ID
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        request.state.request_id = request_id

        # Start tracking
        start_time = time.time()

        # Determine endpoint type
        endpoint_type = self._determine_endpoint_type(request.url.path)

        # Track request
        await self.metrics.track_request(
            request_id=request_id,
            request=request,
            endpoint_type=endpoint_type,
            user=getattr(request.state, "user", None)
        )

        # Process request
        try:
            response = await call_next(request)
            response_time_ms = (time.time() - start_time) * 1000

            # Extract metrics from response
            tokens = self._extract_token_usage(response)
            model = self._extract_model(response)

            # Track response
            await self.metrics.track_response(
                request_id=request_id,
                response_time_ms=response_time_ms,
                status_code=response.status_code,
                model=model,
                tokens=tokens
            )

            # Add metrics headers
            response.headers["x-request-id"] = request_id
            response.headers["x-response-time-ms"] = str(response_time_ms)

            return response

        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000

            # Track error
            await self.metrics.track_response(
                request_id=request_id,
                response_time_ms=response_time_ms,
                status_code=500,
                error=e
            )
            raise
```

## 6. **Storage Backends**

```python
# metrics/storage/sqlite.py
import aiosqlite
from datetime import datetime
from typing import List, Optional, Dict, Any

class SQLiteMetricStorage(MetricStorage):
    """SQLite storage for metrics"""

    def __init__(self, db_path: str = "metrics.db"):
        self.db_path = db_path
        self._init_db()

    async def _init_db(self):
        """Initialize database schema"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    request_id TEXT PRIMARY KEY,
                    timestamp DATETIME NOT NULL,
                    user_id TEXT,
                    method TEXT NOT NULL,
                    path TEXT NOT NULL,
                    endpoint_type TEXT NOT NULL,
                    user_agent TEXT,
                    model TEXT,
                    response_time_ms REAL,
                    status_code INTEGER,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    total_tokens INTEGER,
                    total_cost REAL,
                    error_type TEXT,
                    error_message TEXT,
                    metadata JSON
                )
            """)

            # Create indexes for common queries
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON metrics(timestamp)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_id
                ON metrics(user_id)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_model
                ON metrics(model)
            """)

            await db.commit()

    async def store(self, metric: RequestMetric):
        """Store a metric"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO metrics
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metric.request_id,
                metric.timestamp,
                metric.user_id,
                metric.method,
                metric.path,
                metric.endpoint_type,
                metric.user_agent,
                metric.model,
                metric.response_time_ms,
                metric.status_code,
                metric.input_tokens,
                metric.output_tokens,
                metric.total_tokens,
                metric.total_cost,
                metric.error_type,
                metric.error_message,
                json.dumps(metric.metadata)
            ))
            await db.commit()
```

## 7. **Prometheus Exporter**

```python
# metrics/exporters/prometheus.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest

class PrometheusExporter(MetricExporter):
    """Export metrics to Prometheus"""

    def __init__(self):
        # Request metrics
        self.request_count = Counter(
            'ccproxy_requests_total',
            'Total number of requests',
            ['method', 'endpoint_type', 'status_code', 'model']
        )

        self.response_time = Histogram(
            'ccproxy_response_time_seconds',
            'Response time in seconds',
            ['method', 'endpoint_type', 'model'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
        )

        # Token metrics
        self.token_usage = Counter(
            'ccproxy_tokens_total',
            'Total token usage',
            ['model', 'token_type']  # token_type: input/output
        )

        # Cost metrics
        self.total_cost = Counter(
            'ccproxy_cost_dollars_total',
            'Total cost in dollars',
            ['model', 'user_id']
        )

        # Error metrics
        self.error_count = Counter(
            'ccproxy_errors_total',
            'Total number of errors',
            ['error_type', 'endpoint_type']
        )

    async def export(self, metric: RequestMetric):
        """Export a metric to Prometheus"""
        labels = {
            'method': metric.method,
            'endpoint_type': metric.endpoint_type,
            'status_code': str(metric.status_code or 'unknown'),
            'model': metric.model or 'unknown'
        }

        # Update counters
        self.request_count.labels(**labels).inc()

        if metric.response_time_ms:
            self.response_time.labels(
                method=metric.method,
                endpoint_type=metric.endpoint_type,
                model=metric.model or 'unknown'
            ).observe(metric.response_time_ms / 1000)

        if metric.input_tokens:
            self.token_usage.labels(
                model=metric.model or 'unknown',
                token_type='input'
            ).inc(metric.input_tokens)

        if metric.output_tokens:
            self.token_usage.labels(
                model=metric.model or 'unknown',
                token_type='output'
            ).inc(metric.output_tokens)

        if metric.total_cost:
            self.total_cost.labels(
                model=metric.model or 'unknown',
                user_id=metric.user_id or 'anonymous'
            ).inc(metric.total_cost)

        if metric.error_type:
            self.error_count.labels(
                error_type=metric.error_type,
                endpoint_type=metric.endpoint_type
            ).inc()

    def get_metrics(self) -> bytes:
        """Get metrics in Prometheus format"""
        return generate_latest()
```

## 8. **Dashboard Routes**

```python
# metrics/dashboard/routes.py
from fastapi import APIRouter, Request, Query
from fastapi.responses import HTMLResponse
from datetime import datetime, timedelta

router = APIRouter()

@router.get("/metrics/dashboard")
async def dashboard(request: Request):
    """Render metrics dashboard"""
    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request}
    )

@router.get("/metrics/api/stats")
async def get_stats(
    metrics: MetricsCollector = Depends(get_metrics_collector),
    period: str = Query("1h", regex="^(1h|6h|24h|7d|30d)$"),
    group_by: Optional[str] = Query(None, regex="^(hour|day|model|user)$")
):
    """Get aggregated statistics"""
    end_time = datetime.utcnow()
    start_time = end_time - _parse_period(period)

    stats = await metrics.get_stats(
        start_time=start_time,
        end_time=end_time,
        group_by=group_by
    )

    return {
        "period": period,
        "start_time": start_time,
        "end_time": end_time,
        "stats": stats
    }

@router.get("/metrics/api/requests/{request_id}")
async def get_request_details(
    request_id: str,
    metrics: MetricsCollector = Depends(get_metrics_collector)
):
    """Get details for a specific request"""
    metric = await metrics.get_request(request_id)
    if not metric:
        raise HTTPException(404, "Request not found")
    return metric

@router.get("/metrics/prometheus")
async def prometheus_metrics(
    exporter: PrometheusExporter = Depends(get_prometheus_exporter)
):
    """Prometheus metrics endpoint"""
    return Response(
        content=exporter.get_metrics(),
        media_type="text/plain"
    )
```

## 9. **Integration with Services**

```python
# services/proxy_service.py
class ProxyService:
    def __init__(
        self,
        proxy_client: ProxyClient,
        metrics: MetricsCollector,
        # ... other dependencies
    ):
        self.proxy_client = proxy_client
        self.metrics = metrics

    async def handle_request(
        self,
        request: Request,
        path: str,
        user: Optional[User] = None
    ) -> Response:
        """Process and forward request with metrics"""
        request_id = request.state.request_id

        # Forward request
        response = await self.proxy_client.forward(request)

        # Extract metrics from response
        if hasattr(response, "model"):
            await self.metrics.update_metric(
                request_id,
                model=response.model
            )

        if hasattr(response, "usage"):
            await self.metrics.update_metric(
                request_id,
                tokens={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                }
            )

        return response
```

## 10. **Configuration**

```python
# config/settings.py
class MetricsSettings(BaseSettings):
    """Metrics configuration"""

    # Enable/disable metrics
    metrics_enabled: bool = True

    # Storage backend
    metrics_storage: str = "sqlite"  # sqlite, postgres, memory
    metrics_db_url: str = "sqlite:///metrics.db"

    # Retention
    metrics_retention_days: int = 30

    # Exporters
    prometheus_enabled: bool = True
    opentelemetry_enabled: bool = False

    # Dashboard
    dashboard_enabled: bool = True
    dashboard_auth_required: bool = True

    # Performance
    metrics_batch_size: int = 100
    metrics_flush_interval: int = 10
```

## Key Features:

1. **Comprehensive Tracking**: Request/response times, token usage, costs, errors
2. **Multiple Storage Options**: In-memory, SQLite, PostgreSQL, time-series DBs
3. **Real-time Exports**: Prometheus, OpenTelemetry, custom webhooks
4. **Built-in Dashboard**: Web UI for viewing metrics
5. **Low Overhead**: Async processing, batching, efficient storage
6. **Privacy-Focused**: API keys are hashed, PII can be excluded
7. **Flexible Aggregation**: By time period, model, user, endpoint
8. **Cost Tracking**: Automatic cost calculation based on model and usage

This metrics system provides complete observability while maintaining performance and privacy.
