"""
Metric models for the metrics domain.

This module defines the data structures used to capture and store
various types of metrics throughout the application.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class MetricType(str, Enum):
    """Types of metrics that can be collected."""
    
    REQUEST = "request"
    RESPONSE = "response"
    ERROR = "error"
    COST = "cost"
    LATENCY = "latency"
    USAGE = "usage"


class MetricRecord(BaseModel):
    """Base metric record."""
    
    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metric_type: MetricType
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RequestMetric(MetricRecord):
    """Metrics for incoming requests."""
    
    metric_type: MetricType = MetricType.REQUEST
    
    # Request details
    method: str
    path: str
    endpoint: str
    api_version: str
    
    # Client information
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None
    
    # Request characteristics
    content_length: Optional[int] = None
    content_type: Optional[str] = None
    
    # Model and provider information
    model: Optional[str] = None
    provider: Optional[str] = None  # 'anthropic' or 'openai'
    
    # Request parameters
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    streaming: bool = False


class ResponseMetric(MetricRecord):
    """Metrics for outgoing responses."""
    
    metric_type: MetricType = MetricType.RESPONSE
    
    # Response details
    status_code: int
    response_time_ms: float
    content_length: Optional[int] = None
    content_type: Optional[str] = None
    
    # Token usage
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cache_read_tokens: Optional[int] = None
    cache_write_tokens: Optional[int] = None
    
    # Streaming information
    streaming: bool = False
    first_token_time_ms: Optional[float] = None
    stream_completion_time_ms: Optional[float] = None
    
    # Quality metrics
    completion_reason: Optional[str] = None
    safety_filtered: bool = False


class ErrorMetric(MetricRecord):
    """Metrics for errors and exceptions."""
    
    metric_type: MetricType = MetricType.ERROR
    
    # Error details
    error_type: str
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    
    # Context
    endpoint: Optional[str] = None
    method: Optional[str] = None
    status_code: Optional[int] = None
    
    # Recovery information
    retry_count: int = 0
    recoverable: bool = False


class CostMetric(MetricRecord):
    """Metrics for cost calculations."""
    
    metric_type: MetricType = MetricType.COST
    
    # Cost breakdown
    input_cost: float = 0.0
    output_cost: float = 0.0
    cache_read_cost: float = 0.0
    cache_write_cost: float = 0.0
    total_cost: float = 0.0
    
    # Pricing model
    model: str
    pricing_tier: Optional[str] = None
    currency: str = "USD"
    
    # Token counts (for validation)
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


class LatencyMetric(MetricRecord):
    """Metrics for latency tracking."""
    
    metric_type: MetricType = MetricType.LATENCY
    
    # Timing breakdown
    request_processing_ms: float = 0.0
    claude_api_call_ms: float = 0.0
    response_processing_ms: float = 0.0
    total_latency_ms: float = 0.0
    
    # Queue and waiting times
    queue_time_ms: float = 0.0
    wait_time_ms: float = 0.0
    
    # Streaming metrics
    first_token_latency_ms: Optional[float] = None
    token_generation_rate: Optional[float] = None  # tokens per second


class UsageMetric(MetricRecord):
    """Metrics for usage tracking."""
    
    metric_type: MetricType = MetricType.USAGE
    
    # Usage counts
    request_count: int = 1
    token_count: int = 0
    
    # Time window
    window_start: datetime
    window_end: datetime
    window_duration_seconds: float
    
    # Aggregation level
    aggregation_level: str = "hourly"  # hourly, daily, weekly, monthly


class MetricsSummary(BaseModel):
    """Summary of metrics over a time period."""
    
    # Time period
    start_time: datetime
    end_time: datetime
    
    # Request metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    error_rate: float = 0.0
    
    # Response metrics
    avg_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0
    
    # Token metrics
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    
    # Cost metrics
    total_cost: float = 0.0
    avg_cost_per_request: float = 0.0
    
    # Usage patterns
    unique_users: int = 0
    peak_requests_per_minute: int = 0
    
    # Model distribution
    model_usage: Dict[str, int] = Field(default_factory=dict)
    
    # Error breakdown
    error_types: Dict[str, int] = Field(default_factory=dict)


class AggregatedMetrics(BaseModel):
    """Aggregated metrics over a time period (alias for MetricsSummary for compatibility)."""
    
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


# Type aliases for convenience
AnyMetric = Union[RequestMetric, ResponseMetric, ErrorMetric, CostMetric, LatencyMetric, UsageMetric]
MetricData = Dict[str, Any]