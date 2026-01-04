"""Data models shared by the endpoint testing helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EndpointTest:
    """Configuration for a single endpoint test."""

    name: str
    endpoint: str
    stream: bool
    request: str  # Key in request_data
    model: str
    description: str = ""

    def __post_init__(self) -> None:
        if not self.description:
            stream_str = "streaming" if self.stream else "non-streaming"
            self.description = f"{self.name} ({stream_str})"


@dataclass
class EndpointRequestResult:
    """Outcome of a single HTTP request made while executing a test."""

    phase: str
    method: str
    status_code: int | None
    stream: bool
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class EndpointTestResult:
    """Result of running a single endpoint test."""

    test: EndpointTest
    index: int
    success: bool
    error: str | None = None
    exception: Exception | None = None
    request_results: list[EndpointRequestResult] = field(default_factory=list)

    @property
    def name(self) -> str:
        """Convenience access to the test name."""
        return self.test.name


@dataclass
class EndpointTestRunSummary:
    """Summary of executing a batch of endpoint tests."""

    base_url: str
    results: list[EndpointTestResult]
    successful_count: int
    failure_count: int
    errors: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        """Total number of executed tests."""
        return self.successful_count + self.failure_count

    @property
    def failed_results(self) -> list[EndpointTestResult]:
        """Return the list of failed test results."""
        return [result for result in self.results if not result.success]

    def all_passed(self) -> bool:
        """Return True when every executed test succeeded."""
        return self.failure_count == 0 and not self.errors

    def assert_success(self) -> None:
        """Raise AssertionError if any test failed (useful for pytest)."""
        if self.all_passed():
            return

        failed_names = ", ".join(result.name for result in self.failed_results)
        parts = []
        if self.failure_count:
            parts.append(
                f"{self.failure_count} endpoint test(s) failed: {failed_names}"
            )
        if self.errors:
            parts.append("; additional errors: " + "; ".join(self.errors))

        raise AssertionError(" ".join(parts) if parts else "Endpoint test run failed")


__all__ = [
    "EndpointTest",
    "EndpointRequestResult",
    "EndpointTestResult",
    "EndpointTestRunSummary",
]
