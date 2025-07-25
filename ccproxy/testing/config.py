"""Configuration models for testing utilities."""

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel


# Type aliases for traffic patterns and response types
TrafficPattern = Literal["constant", "burst", "ramping", "realistic"]
ResponseType = Literal["success", "error", "mixed", "unavailable"]
AuthType = Literal["none", "bearer", "configured", "credentials"]


class MockResponseConfig(BaseModel):
    """Configuration for realistic mock responses."""

    # Token range configurations
    input_token_range: tuple[int, int] = (10, 500)  # Min/max input tokens
    output_token_range: tuple[int, int] = (5, 1000)  # Min/max output tokens
    cache_token_probability: float = 0.3  # Chance of cache tokens
    cache_read_range: tuple[int, int] = (50, 200)  # Cache read token range
    cache_write_range: tuple[int, int] = (20, 100)  # Cache write token range

    # Latency simulation
    base_latency_ms: tuple[int, int] = (100, 2000)  # Base response latency
    streaming_chunk_delay_ms: tuple[int, int] = (10, 100)  # Per-chunk delay

    # Content variation
    response_length_variety: bool = True  # Vary response length
    short_response_range: tuple[int, int] = (1, 3)  # Short response sentences
    long_response_range: tuple[int, int] = (5, 15)  # Long response sentences

    # Error simulation
    simulate_errors: bool = True  # Include error scenarios
    error_probability: float = 0.05  # Chance of error response

    # Realistic timing
    token_generation_rate: float = 50.0  # Tokens per second for streaming


class TrafficConfig(BaseModel):
    """Configuration for traffic generation scenarios."""

    # Basic settings
    duration_seconds: int = 60
    requests_per_second: float = 1.0
    pattern: TrafficPattern = "constant"

    # Target Configuration
    target_url: str = "http://localhost:8000"  # Proxy server URL
    api_formats: list[str] = ["anthropic", "openai"]  # Which formats to test
    format_distribution: dict[str, float] = {  # % distribution of formats
        "anthropic": 0.7,
        "openai": 0.3,
    }

    # Request configuration
    models: list[str] = ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"]
    message_types: list[str] = ["short", "long", "tool_use"]
    streaming_probability: float = 0.3

    # Advanced Request Types
    advanced_scenarios: bool = False  # Enable complex scenarios
    tool_use_probability: float = 0.2  # Specific probability for tool use

    # Response configuration
    response_type: ResponseType = "mixed"
    error_probability: float = 0.1
    latency_ms_min: int = 100
    latency_ms_max: int = 2000

    # Authentication and Testing
    bypass_mode: bool = True  # Use bypass headers (test mode)
    real_api_keys: dict[str, str] = {}  # Real API keys when bypass_mode=False

    # Timeframe simulation
    simulate_historical: bool = False
    start_timestamp: datetime | None = None
    end_timestamp: datetime | None = None

    # Output configuration
    output_file: Path | None = None
    log_requests: bool = True
    log_responses: bool = False
    log_format_conversions: bool = True  # Log API format transformations


class RequestScenario(BaseModel):
    """Individual request scenario configuration."""

    model: str
    message_type: str
    streaming: bool
    response_type: ResponseType
    timestamp: datetime

    # API Format and Endpoint Control
    api_format: Literal["anthropic", "openai"] = "anthropic"
    endpoint_path: str = (
        "/api/v1/messages"  # "/api/v1/messages" or "/api/v1/chat/completions"
    )

    # Request Control
    bypass_upstream: bool = True  # Add bypass header to prevent real API calls
    use_real_auth: bool = False  # Use real API keys vs test mode

    # Enhanced Headers
    headers: dict[str, str] = {}  # All request headers

    # Target URL
    target_url: str = "http://localhost:8000"  # Full base URL for request

    # Payload Customization
    custom_payload: dict[str, Any] | None = None  # Override default payload generation


class TrafficMetrics(BaseModel):
    """Enhanced metrics for dual-format testing."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    error_requests: int = 0
    average_latency_ms: float = 0.0
    requests_per_second: float = 0.0
    start_time: datetime
    end_time: datetime | None = None

    # Format-specific metrics
    anthropic_requests: int = 0
    openai_requests: int = 0

    # Streaming vs non-streaming
    streaming_requests: int = 0
    standard_requests: int = 0

    # Format validation
    format_validation_errors: int = 0

    # Response time by format
    anthropic_avg_latency_ms: float = 0.0
    openai_avg_latency_ms: float = 0.0

    # Token usage
    total_input_tokens: int = 0
    total_output_tokens: int = 0
