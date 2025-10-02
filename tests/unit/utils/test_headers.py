"""Tests for header utility helpers."""

from typing import Any

from ccproxy.utils.headers import collect_cli_forward_headers


class DummyHeaders:
    """Stub implementation mimicking DetectedHeaders."""

    def __init__(self, values: dict[str, str], *, fail_filter: bool = False) -> None:
        self._values = values
        self._fail_filter = fail_filter

    def filtered(self, *, ignores: Any = None, redacted: Any = None) -> dict[str, str]:
        if self._fail_filter:
            raise RuntimeError("filter failed")
        ignores = {*(ignores or [])}
        redacted = {*(redacted or [])}
        return {
            key: value
            for key, value in self._values.items()
            if key not in ignores and key not in redacted
        }

    def as_dict(self) -> dict[str, str]:
        return dict(self._values)

    def __bool__(self) -> bool:  # pragma: no cover - mirror DetectedHeaders
        return bool(self._values)


class DummyDetectionService:
    """Stub detection service exposing header helper methods."""

    def __init__(
        self,
        headers: DummyHeaders,
        ignores: list[str] | None = None,
        redacted: list[str] | None = None,
    ) -> None:
        self._headers = headers
        self._ignores = ignores or []
        self._redacted = redacted or []

    def get_detected_headers(self) -> DummyHeaders:
        return self._headers

    def get_ignored_headers(self) -> list[str]:
        return list(self._ignores)

    def get_redacted_headers(self) -> list[str]:
        return list(self._redacted)


def test_collect_cli_headers_none_service_returns_empty() -> None:
    assert collect_cli_forward_headers(None) == {}


def test_collect_cli_headers_filters_ignored_and_redacted() -> None:
    headers = DummyHeaders(
        {
            "authorization": "token",
            "session_id": "session-123",
            "editor-version": "vscode",
        }
    )
    service: Any = DummyDetectionService(
        headers,
        ignores=["authorization"],
        redacted=["session_id"],
    )

    result = collect_cli_forward_headers(service)

    assert result == {"editor-version": "vscode"}


def test_collect_cli_headers_falls_back_when_filtered_raises() -> None:
    headers = DummyHeaders(
        {
            "authorization": "token",
            "session_id": "session-123",
        },
        fail_filter=True,
    )
    service: Any = DummyDetectionService(headers)

    result = collect_cli_forward_headers(service)

    assert result == {
        "authorization": "token",
        "session_id": "session-123",
    }
