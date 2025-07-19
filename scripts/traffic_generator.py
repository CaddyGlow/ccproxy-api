#!/usr/bin/env python3
"""Traffic Generator for CCProxy API Server.

This script generates realistic API traffic patterns for testing and development.
It reuses the test infrastructure's factories and fixtures to provide consistent
mocking and authentication patterns.
"""

import asyncio
import contextlib
import json
import random
import time
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import httpx
import structlog
import typer

from ccproxy.testing import (
    PayloadBuilder,
    RequestScenario,
    ResponseHandler,
    ScenarioGenerator,
    TrafficConfig,
    TrafficMetrics,
)


logger = structlog.get_logger(__name__)


# Enums for CLI compatibility
class TrafficPatternEnum(str, Enum):
    constant = "constant"
    burst = "burst"
    ramping = "ramping"
    realistic = "realistic"


class ResponseTypeEnum(str, Enum):
    success = "success"
    error = "error"
    mixed = "mixed"
    unavailable = "unavailable"


class AuthTypeEnum(str, Enum):
    none = "none"
    bearer = "bearer"
    configured = "configured"
    credentials = "credentials"


class TrafficGenerator:
    """Main traffic generator class."""

    def __init__(self, config: TrafficConfig):
        """Initialize traffic generator with configuration."""
        self.config = config
        self.metrics = TrafficMetrics(start_time=datetime.now(UTC))

        # Create utilities for generating scenarios and processing responses
        self.scenario_generator = ScenarioGenerator(config)
        self.response_handler = ResponseHandler()
        self.payload_builder: PayloadBuilder = PayloadBuilder()

        # Setup logging
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure structured logging for the traffic generator."""
        # Structlog configuration is typically handled at the application level
        # For the traffic generator, we'll use the existing structlog configuration
        pass

    def generate_scenarios(self) -> list[RequestScenario]:
        """Generate request scenarios based on configuration."""
        return self.scenario_generator.generate_scenarios()

    async def run_scenario(
        self, scenario: RequestScenario, client: httpx.AsyncClient
    ) -> dict[str, Any]:
        """Execute a single request scenario."""
        # Build request payload
        payload = self._build_request_payload(scenario)

        # Add simulated latency
        if not self.config.simulate_historical:
            latency = random.randint(
                self.config.latency_ms_min, self.config.latency_ms_max
            )
            await asyncio.sleep(latency / 1000.0)

        # Execute request
        start_time = time.time()

        try:
            if scenario.streaming:
                response_data = await self._execute_streaming_request(
                    client, payload, scenario
                )
            else:
                response_data = await self._execute_standard_request(
                    client, payload, scenario
                )

            end_time = time.time()

            # Update metrics with format-specific tracking
            self.metrics.total_requests += 1
            latency_ms = (end_time - start_time) * 1000

            # Update format-specific metrics
            if scenario.api_format == "anthropic":
                self.metrics.anthropic_requests += 1
            else:
                self.metrics.openai_requests += 1

            # Update streaming metrics
            if scenario.streaming:
                self.metrics.streaming_requests += 1
            else:
                self.metrics.standard_requests += 1

            # Update average latency (running average)
            if self.metrics.total_requests == 1:
                self.metrics.average_latency_ms = latency_ms
            else:
                self.metrics.average_latency_ms = (
                    self.metrics.average_latency_ms * (self.metrics.total_requests - 1)
                    + latency_ms
                ) / self.metrics.total_requests

            # Update format-specific latency
            if scenario.api_format == "anthropic":
                if self.metrics.anthropic_requests == 1:
                    self.metrics.anthropic_avg_latency_ms = latency_ms
                else:
                    self.metrics.anthropic_avg_latency_ms = (
                        self.metrics.anthropic_avg_latency_ms
                        * (self.metrics.anthropic_requests - 1)
                        + latency_ms
                    ) / self.metrics.anthropic_requests
            else:
                if self.metrics.openai_requests == 1:
                    self.metrics.openai_avg_latency_ms = latency_ms
                else:
                    self.metrics.openai_avg_latency_ms = (
                        self.metrics.openai_avg_latency_ms
                        * (self.metrics.openai_requests - 1)
                        + latency_ms
                    ) / self.metrics.openai_requests

            # Update token metrics if available
            if "tokens_input" in response_data and response_data["tokens_input"]:
                self.metrics.total_input_tokens += response_data["tokens_input"]
            if "tokens_output" in response_data and response_data["tokens_output"]:
                self.metrics.total_output_tokens += response_data["tokens_output"]

            if response_data.get("status_code", 200) < 400:
                self.metrics.successful_requests += 1
            else:
                self.metrics.failed_requests += 1

            # Log request if enabled
            if self.config.log_requests:
                logger.info(
                    "Request completed",
                    model=scenario.model,
                    message_type=scenario.message_type,
                    api_format=scenario.api_format,
                    status_code=response_data.get("status_code", 200),
                    latency_ms=latency_ms,
                )

            if self.config.log_responses:
                logger.debug("Response received", response=response_data)

            if self.config.log_format_conversions and scenario.api_format == "openai":
                logger.info(
                    "Format conversion applied",
                    from_format="openai",
                    to_format="anthropic",
                    model=scenario.model,
                )

            return {
                "scenario": scenario.model_dump(),
                "response": response_data,
                "latency_ms": latency_ms,
                "timestamp": scenario.timestamp.isoformat(),
            }

        except Exception as e:
            self.metrics.error_requests += 1
            logger.error(
                "Request failed",
                error=str(e),
                model=scenario.model,
                api_format=scenario.api_format,
                endpoint=scenario.endpoint_path,
            )
            return {
                "scenario": scenario.model_dump(),
                "error": str(e),
                "timestamp": scenario.timestamp.isoformat(),
            }

    def _build_request_payload(self, scenario: RequestScenario) -> dict[str, Any]:
        """Build request payload based on scenario."""
        return self.payload_builder.build_payload(scenario)

    async def _execute_standard_request(
        self,
        client: httpx.AsyncClient,
        payload: dict[str, Any],
        scenario: RequestScenario,
    ) -> dict[str, Any]:
        """Execute a standard (non-streaming) request."""
        url = f"{scenario.target_url}{scenario.endpoint_path}"

        response = await client.post(
            url,
            json=payload,
            headers=scenario.headers,
            timeout=30.0,
        )

        # Process response using ResponseHandler
        return self.response_handler.process_response(response, scenario)

    async def _execute_streaming_request(
        self,
        client: httpx.AsyncClient,
        payload: dict[str, Any],
        scenario: RequestScenario,
    ) -> dict[str, Any]:
        """Execute a streaming request."""
        url = f"{scenario.target_url}{scenario.endpoint_path}"

        async with client.stream(
            "POST",
            url,
            json=payload,
            headers=scenario.headers,
            timeout=30.0,
        ) as response:
            # Process streaming response using ResponseHandler
            return self.response_handler.process_response(response, scenario)


async def run_traffic_generation(config: TrafficConfig) -> dict[str, Any]:
    """Run traffic generation with real HTTP client and proper concurrency."""
    generator = TrafficGenerator(config)
    scenarios = generator.generate_scenarios()

    # Create real HTTP client with appropriate connection limits
    limits = httpx.Limits(max_keepalive_connections=50, max_connections=100)
    timeout = httpx.Timeout(30.0, connect=10.0)

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        results = []

        # Create semaphore to limit concurrent requests (avoid overwhelming the server)
        max_concurrent = min(20, len(scenarios))  # Limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)

        # Real-time RPS tracking
        request_timestamps = []
        rps_display_interval = 5  # Show RPS every 5 seconds
        last_rps_display = datetime.now(UTC)

        async def execute_scenario_with_timing(
            scenario: RequestScenario, index: int
        ) -> dict[str, Any] | None:
            """Execute a scenario with proper timing and concurrency control."""
            async with semaphore:
                # Wait until the appropriate time for this scenario
                if not config.simulate_historical:
                    current_time = datetime.now(UTC)
                    time_until_execution = (
                        scenario.timestamp - current_time
                    ).total_seconds()
                    if time_until_execution > 0:
                        await asyncio.sleep(time_until_execution)

                try:
                    result = await generator.run_scenario(scenario, client)

                    # Track request completion for RPS calculation
                    completion_time = datetime.now(UTC)
                    request_timestamps.append(completion_time)

                    # Clean up old timestamps (keep only last 60 seconds for RPS calculation)
                    cutoff_time = completion_time - timedelta(seconds=60)
                    while request_timestamps and request_timestamps[0] < cutoff_time:
                        request_timestamps.pop(0)

                    # Show progress for longer runs
                    if (
                        len(scenarios) > 10
                        and index % max(1, len(scenarios) // 10) == 0
                    ):
                        progress = (index / len(scenarios)) * 100
                        print(f"Progress: {progress:.1f}%")

                    return result
                except Exception as e:
                    print(f"Error executing scenario {index}: {e}")
                    return None

        async def display_rps_periodically() -> None:
            """Display current RPS every few seconds."""
            nonlocal last_rps_display
            while True:
                await asyncio.sleep(rps_display_interval)
                current_time = datetime.now(UTC)

                # Calculate current RPS based on requests in last 60 seconds
                cutoff_time = current_time - timedelta(seconds=60)
                recent_requests = [ts for ts in request_timestamps if ts >= cutoff_time]
                current_rps = len(recent_requests) / 60.0 if recent_requests else 0.0

                # Calculate RPS for last interval
                interval_cutoff = current_time - timedelta(seconds=rps_display_interval)
                interval_requests = [
                    ts for ts in request_timestamps if ts >= interval_cutoff
                ]
                interval_rps = (
                    len(interval_requests) / rps_display_interval
                    if interval_requests
                    else 0.0
                )

                total_completed = len(request_timestamps)
                print(
                    f"[RPS Monitor] Current: {interval_rps:.1f} RPS (last {rps_display_interval}s) | Avg: {current_rps:.1f} RPS (last 60s) | Total: {total_completed} requests"
                )

                last_rps_display = current_time

        # Start RPS monitoring task for long-running tests
        rps_monitor_task: asyncio.Task[None] | None = None
        if (
            config.duration_seconds > 10
        ):  # Only show RPS for tests longer than 10 seconds
            rps_monitor_task = asyncio.create_task(display_rps_periodically())

        try:
            # Execute all scenarios concurrently with proper timing
            if config.simulate_historical:
                # For historical simulation, run all scenarios concurrently
                historical_tasks = [
                    execute_scenario_with_timing(scenario, i)
                    for i, scenario in enumerate(scenarios)
                ]
                results = await asyncio.gather(
                    *historical_tasks, return_exceptions=True
                )
                # Filter out None results and exceptions
                results = [
                    r for r in results if r is not None and not isinstance(r, Exception)
                ]
            else:
                # For real-time simulation, use a scheduler approach
                real_time_tasks: list[asyncio.Task[dict[str, Any] | None]] = []
                for i, scenario in enumerate(scenarios):
                    task: asyncio.Task[dict[str, Any] | None] = asyncio.create_task(
                        execute_scenario_with_timing(scenario, i)
                    )
                    real_time_tasks.append(task)

                # Wait for all tasks to complete
                completed_results = await asyncio.gather(
                    *real_time_tasks, return_exceptions=True
                )
                results = [
                    r
                    for r in completed_results
                    if r is not None and not isinstance(r, Exception)
                ]
        finally:
            # Stop RPS monitoring
            if rps_monitor_task:
                rps_monitor_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await rps_monitor_task

    # Update final metrics
    generator.metrics.end_time = datetime.now(UTC)
    total_time = (
        generator.metrics.end_time - generator.metrics.start_time
    ).total_seconds()
    generator.metrics.requests_per_second = (
        generator.metrics.total_requests / total_time if total_time > 0 else 0
    )

    return {
        "config": config.model_dump(mode="json"),
        "metrics": generator.metrics.model_dump(mode="json"),
        "results": results,
    }


# CLI Interface
app = typer.Typer(help="Traffic Generator for CCProxy API Server")


@app.command()
def generate(
    duration: int = typer.Option(60, "--duration", "-d", help="Duration in seconds"),
    rps: float = typer.Option(1.0, "--rps", "-r", help="Requests per second"),
    pattern: TrafficPatternEnum = typer.Option(
        TrafficPatternEnum.constant, "--pattern", "-p", help="Traffic pattern"
    ),
    response_type: ResponseTypeEnum = typer.Option(
        ResponseTypeEnum.mixed, "--response-type", help="Response type simulation"
    ),
    target_url: str = typer.Option(
        "http://localhost:8000", "--target-url", help="Target proxy server URL"
    ),
    api_formats: str = typer.Option(
        "anthropic,openai",
        "--api-formats",
        help="API formats to test (comma-separated)",
    ),
    bypass_mode: bool = typer.Option(
        True,
        "--bypass/--no-bypass",
        help="Use bypass headers to prevent real API calls",
    ),
    anthropic_weight: float = typer.Option(
        0.7, "--anthropic-weight", help="Weight for Anthropic API format (0.0-1.0)"
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Output file for logs"
    ),
    log_responses: bool = typer.Option(
        False, "--log-responses", help="Log full responses"
    ),
    log_format_conversions: bool = typer.Option(
        True,
        "--log-conversions/--no-log-conversions",
        help="Log API format conversions",
    ),
    historical: bool = typer.Option(
        False, "--historical", help="Simulate historical timeframe"
    ),
    start_time: str | None = typer.Option(
        None,
        "--start-time",
        help="Start timestamp (ISO format, e.g., 2024-01-15T10:30:00Z)",
    ),
    end_time: str | None = typer.Option(
        None,
        "--end-time",
        help="End timestamp (ISO format, e.g., 2024-01-15T11:30:00Z)",
    ),
) -> None:
    """Generate API traffic based on specified parameters."""

    # Parse timestamps if provided
    start_timestamp = None
    end_timestamp = None
    if start_time:
        start_timestamp = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
    if end_time:
        end_timestamp = datetime.fromisoformat(end_time.replace("Z", "+00:00"))

    # Parse API formats
    format_list = [fmt.strip() for fmt in api_formats.split(",")]

    # Calculate format distribution
    openai_weight = 1.0 - anthropic_weight
    format_distribution = {}
    if "anthropic" in format_list:
        format_distribution["anthropic"] = anthropic_weight
    if "openai" in format_list:
        format_distribution["openai"] = openai_weight

    config = TrafficConfig(
        duration_seconds=duration,
        requests_per_second=rps,
        pattern=pattern.value,
        response_type=response_type.value,
        target_url=target_url,
        api_formats=format_list,
        format_distribution=format_distribution,
        bypass_mode=bypass_mode,
        output_file=output,
        log_responses=log_responses,
        log_format_conversions=log_format_conversions,
        simulate_historical=historical,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
    )

    typer.echo(f"Generating traffic to {target_url}")
    typer.echo(
        f"API Formats: {', '.join(format_list)} (Anthropic: {anthropic_weight:.1%}, OpenAI: {openai_weight:.1%})"
    )
    typer.echo(f"Pattern: {pattern.value}, RPS: {rps}, Duration: {duration}s")
    typer.echo(f"Bypass Mode: {'Enabled' if bypass_mode else 'Disabled'}")

    # Run traffic generation
    try:
        results_data = asyncio.run(run_traffic_generation(config))

        # Display results
        metrics = results_data["metrics"]
        typer.echo("\n--- Traffic Generation Complete ---")
        typer.echo(f"Total Requests: {metrics['total_requests']}")
        typer.echo(f"Successful: {metrics['successful_requests']}")
        typer.echo(f"Failed: {metrics['failed_requests']}")
        typer.echo(f"Errors: {metrics['error_requests']}")
        typer.echo(f"Actual RPS: {metrics['requests_per_second']:.2f}")
        typer.echo(f"Average Latency: {metrics['average_latency_ms']:.2f}ms")

        # Format-specific metrics
        if metrics.get("anthropic_requests", 0) > 0:
            typer.echo(
                f"Anthropic Requests: {metrics['anthropic_requests']} (avg: {metrics['anthropic_avg_latency_ms']:.2f}ms)"
            )
        if metrics.get("openai_requests", 0) > 0:
            typer.echo(
                f"OpenAI Requests: {metrics['openai_requests']} (avg: {metrics['openai_avg_latency_ms']:.2f}ms)"
            )

        # Token metrics
        if metrics.get("total_input_tokens", 0) > 0:
            typer.echo(
                f"Total Tokens: {metrics['total_input_tokens']} input, {metrics['total_output_tokens']} output"
            )

        if output:
            # Save detailed results
            with output.open("w") as f:
                json.dump(results_data, f, indent=2, default=str)
            typer.echo(f"Detailed results saved to: {output}")

    except Exception as e:
        typer.echo(f"Error running traffic generation: {e}", err=True)
        raise typer.Exit(1) from e


@app.command()
def presets() -> None:
    """Show available preset configurations."""
    presets = {
        "light-load": "1 RPS constant for 60s with mixed responses",
        "moderate-load": "5 RPS constant for 120s with 10% errors",
        "heavy-load": "20 RPS burst pattern for 300s",
        "stress-test": "50 RPS ramping for 600s",
        "historical-sim": "10 RPS realistic pattern with historical timestamps",
    }

    typer.echo("Available preset configurations:")
    for name, description in presets.items():
        typer.echo(f"  {name}: {description}")

    typer.echo("\nTo run a preset:")
    typer.echo("  python tools/traffic_generator.py preset <preset-name>")


@app.command()
def preset(
    preset_name: str = typer.Argument(..., help="Name of the preset to run"),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Output file for logs"
    ),
) -> None:
    """Run a predefined preset configuration."""

    preset_configs = {
        "light-load": TrafficConfig(
            duration_seconds=60,
            requests_per_second=1.0,
            pattern="constant",
            response_type="mixed",
            error_probability=0.05,
        ),
        "moderate-load": TrafficConfig(
            duration_seconds=120,
            requests_per_second=5.0,
            pattern="constant",
            response_type="mixed",
            error_probability=0.1,
        ),
        "heavy-load": TrafficConfig(
            duration_seconds=300,
            requests_per_second=20.0,
            pattern="burst",
            response_type="mixed",
            error_probability=0.15,
        ),
        "stress-test": TrafficConfig(
            duration_seconds=600,
            requests_per_second=50.0,
            pattern="ramping",
            response_type="mixed",
            error_probability=0.2,
        ),
        "historical-sim": TrafficConfig(
            duration_seconds=300,
            requests_per_second=10.0,
            pattern="realistic",
            response_type="mixed",
            simulate_historical=True,
            start_timestamp=datetime.now(UTC) - timedelta(hours=1),
            error_probability=0.08,
        ),
    }

    if preset_name not in preset_configs:
        typer.echo(f"Error: Unknown preset '{preset_name}'", err=True)
        typer.echo("Available presets:")
        for name in preset_configs:
            typer.echo(f"  {name}")
        raise typer.Exit(1)

    config = preset_configs[preset_name]
    if output:
        config.output_file = output

    typer.echo(f"Running preset: {preset_name}")
    typer.echo("Configuration:")
    typer.echo(f"  Duration: {config.duration_seconds}s")
    typer.echo(f"  RPS: {config.requests_per_second}")
    typer.echo(f"  Pattern: {config.pattern}")
    typer.echo(f"  Response Type: {config.response_type}")
    typer.echo(f"  Target URL: {config.target_url}")
    typer.echo(f"  API Formats: {', '.join(config.api_formats)}")

    # Run traffic generation
    try:
        results_data = asyncio.run(run_traffic_generation(config))

        # Display results
        metrics = results_data["metrics"]
        typer.echo("\n--- Traffic Generation Complete ---")
        typer.echo(f"Total Requests: {metrics['total_requests']}")
        typer.echo(f"Successful: {metrics['successful_requests']}")
        typer.echo(f"Failed: {metrics['failed_requests']}")
        typer.echo(f"Errors: {metrics['error_requests']}")
        typer.echo(f"Actual RPS: {metrics['requests_per_second']:.2f}")
        typer.echo(f"Average Latency: {metrics['average_latency_ms']:.2f}ms")

        if output:
            # Save detailed results
            with output.open("w") as f:
                json.dump(
                    {
                        "preset": preset_name,
                        "config": config.model_dump(mode="json"),
                        "metrics": metrics,
                        "results": results_data["results"],
                    },
                    f,
                    indent=2,
                    default=str,
                )
            typer.echo(f"Detailed results saved to: {output}")

    except Exception as e:
        typer.echo(f"Error running traffic generation: {e}", err=True)
        raise typer.Exit(1) from e


if __name__ == "__main__":
    app()
