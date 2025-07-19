"""Scenario generation and traffic pattern utilities."""

import random
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from ccproxy.testing.config import RequestScenario, ResponseType, TrafficConfig


class ScenarioGenerator:
    """Generate request scenarios based on traffic configuration."""

    def __init__(self, config: TrafficConfig):
        self.config = config

    def generate_scenarios(self) -> list[RequestScenario]:
        """Generate request scenarios based on configuration."""
        total_requests = int(
            self.config.duration_seconds * self.config.requests_per_second
        )
        scenarios = []

        # Calculate timeframe
        start_time = self.config.start_timestamp or datetime.now(UTC)
        time_span = self.config.duration_seconds

        for i in range(total_requests):
            # Determine timing based on pattern
            time_offset = self._calculate_time_offset(i, total_requests, time_span)
            request_time = start_time + time_offset

            # Select random parameters
            model = random.choice(self.config.models)
            message_type = random.choice(self.config.message_types)
            streaming = random.random() < self.config.streaming_probability

            # Determine response type
            response_type = self._determine_response_type()

            # Determine API format based on distribution
            api_format = self._determine_api_format()

            # Set endpoint path based on format
            endpoint_path = (
                "/api/v1/chat/completions"
                if api_format == "openai"
                else "/api/v1/messages"
            )

            # Generate headers with bypass and format-specific headers
            headers = self._generate_headers(api_format, streaming)

            scenarios.append(
                RequestScenario(
                    model=model,
                    message_type=message_type,
                    streaming=streaming,
                    response_type=response_type,
                    timestamp=request_time,
                    api_format=api_format,
                    endpoint_path=endpoint_path,
                    bypass_upstream=self.config.bypass_mode,
                    use_real_auth=not self.config.bypass_mode,
                    headers=headers,
                    target_url=self.config.target_url,
                )
            )

        return scenarios

    def _calculate_time_offset(
        self, request_index: int, total_requests: int, time_span: int
    ) -> timedelta:
        """Calculate time offset for request based on traffic pattern."""
        if self.config.pattern == "constant":
            return timedelta(seconds=request_index / self.config.requests_per_second)
        elif self.config.pattern == "burst":
            # Front-load requests in bursts
            burst_size = max(1, int(total_requests * 0.1))
            if request_index < burst_size:
                return timedelta(seconds=request_index * 0.1)
            else:
                remaining_time = time_span - (burst_size * 0.1)
                remaining_requests = total_requests - burst_size
                return timedelta(
                    seconds=(burst_size * 0.1)
                    + ((request_index - burst_size) / remaining_requests)
                    * remaining_time
                )
        elif self.config.pattern == "ramping":
            # Gradually increase request rate
            normalized_time = request_index / total_requests
            accelerated_time = normalized_time**2
            return timedelta(seconds=accelerated_time * time_span)
        else:  # realistic
            # Add some randomness to simulate real user behavior
            base_time = request_index / self.config.requests_per_second
            jitter = random.uniform(-0.5, 0.5)
            return timedelta(seconds=max(0, base_time + jitter))

    def _determine_response_type(self) -> ResponseType:
        """Determine response type based on configuration."""
        if self.config.response_type == "mixed":
            rand = random.random()
            if rand < self.config.error_probability:
                return "error"
            elif rand < self.config.error_probability * 1.2:
                return "unavailable"
            else:
                return "success"
        else:
            return self.config.response_type

    def _determine_api_format(self) -> Literal["anthropic", "openai"]:
        """Determine API format based on distribution configuration."""
        if len(self.config.api_formats) == 1:
            format_name = self.config.api_formats[0]
            if format_name == "anthropic":
                return "anthropic"
            elif format_name == "openai":
                return "openai"
            return "anthropic"  # Default fallback

        # Use weighted random selection based on format_distribution
        rand = random.random()
        cumulative = 0.0

        for format_name in self.config.api_formats:
            weight = self.config.format_distribution.get(format_name, 0.0)
            cumulative += weight
            if rand <= cumulative:
                if format_name == "anthropic":
                    return "anthropic"
                elif format_name == "openai":
                    return "openai"

        # Fallback to first format if distribution doesn't add up
        format_name = self.config.api_formats[0]
        if format_name == "anthropic":
            return "anthropic"
        elif format_name == "openai":
            return "openai"
        return "anthropic"  # Default fallback

    def _generate_headers(self, api_format: str, streaming: bool) -> dict[str, str]:
        """Generate headers with bypass and format-specific headers."""
        headers = {}

        # Add bypass header if in bypass mode
        if self.config.bypass_mode:
            headers["X-CCProxy-Bypass-Upstream"] = "true"

        # Add real API authentication if not in bypass mode
        if not self.config.bypass_mode and self.config.real_api_keys:
            if api_format == "openai" and "openai" in self.config.real_api_keys:
                headers["Authorization"] = (
                    f"Bearer {self.config.real_api_keys['openai']}"
                )
            elif api_format == "anthropic" and "anthropic" in self.config.real_api_keys:
                headers["Authorization"] = (
                    f"Bearer {self.config.real_api_keys['anthropic']}"
                )

        # Format-specific headers
        if api_format == "openai":
            headers["Content-Type"] = "application/json"
            headers["Accept"] = "application/json"
        else:  # anthropic
            headers["Content-Type"] = "application/json"
            headers["Accept"] = "application/json"
            headers["anthropic-version"] = "2023-06-01"

        # Streaming-specific headers
        if streaming:
            headers["Accept"] = "text/event-stream"
            headers["Cache-Control"] = "no-cache"

        return headers


class TrafficPatternAnalyzer:
    """Analyze and validate traffic patterns."""

    @staticmethod
    def analyze_distribution(scenarios: list[RequestScenario]) -> dict[str, Any]:
        """Analyze the distribution of scenarios."""
        analysis = {
            "total_scenarios": len(scenarios),
            "api_format_distribution": {},
            "model_distribution": {},
            "message_type_distribution": {},
            "streaming_percentage": 0.0,
            "time_span_seconds": 0.0,
        }

        if not scenarios:
            return analysis

        # Count distributions
        api_formats: dict[str, int] = {}
        models: dict[str, int] = {}
        message_types: dict[str, int] = {}
        streaming_count = 0

        for scenario in scenarios:
            # API format distribution
            api_formats[scenario.api_format] = (
                api_formats.get(scenario.api_format, 0) + 1
            )

            # Model distribution
            models[scenario.model] = models.get(scenario.model, 0) + 1

            # Message type distribution
            message_types[scenario.message_type] = (
                message_types.get(scenario.message_type, 0) + 1
            )

            # Streaming count
            if scenario.streaming:
                streaming_count += 1

        # Calculate percentages
        total = len(scenarios)
        analysis["api_format_distribution"] = {
            k: v / total for k, v in api_formats.items()
        }
        analysis["model_distribution"] = {k: v / total for k, v in models.items()}
        analysis["message_type_distribution"] = {
            k: v / total for k, v in message_types.items()
        }
        analysis["streaming_percentage"] = streaming_count / total

        # Calculate time span
        timestamps = [scenario.timestamp for scenario in scenarios]
        if timestamps:
            analysis["time_span_seconds"] = (
                max(timestamps) - min(timestamps)
            ).total_seconds()

        return analysis
