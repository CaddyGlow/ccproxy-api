"""Utilities for loading recorded endpoint samples for integration tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "endpoint_samples"


def load_sample(name: str) -> dict[str, Any]:
    """Load a single recorded sample by name."""

    path = SAMPLE_DIR / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Sample '{name}' not found at {path}")
    with path.open("r", encoding="utf-8") as handle:
        data: dict[str, Any] = json.load(handle)
        return data


def load_sample_registry() -> dict[str, dict[str, Any]]:
    """Load every recorded sample from the endpoint_samples directory."""

    registry: dict[str, dict[str, Any]] = {}
    for sample_path in sorted(SAMPLE_DIR.glob("*.json")):
        with sample_path.open("r", encoding="utf-8") as handle:
            sample = json.load(handle)

        name = sample.get("name")
        if not name:
            # Fallback to filename without suffix when metadata missing
            name = sample_path.stem
            sample["name"] = name
        registry[name] = sample

    return registry


def serialize_sse_events(events: list[dict[str, Any]]) -> bytes:
    """Serialize recorded SSE events to bytes suitable for httpx responses."""

    chunks: list[str] = []
    for event in events:
        lines: list[str] = []
        event_name = event.get("event")
        if event_name:
            lines.append(f"event: {event_name}")

        if "data" in event and event["data"] is not None:
            data_payload = event["data"]
        elif "json" in event:
            data_payload = json.dumps(event["json"])
        elif event.get("done"):
            data_payload = "[DONE]"
        else:
            data_payload = ""

        lines.append(f"data: {data_payload}")
        chunks.append("\n".join(lines))

    # Ensure SSE stream terminates with blank line
    return ("\n\n".join(chunks) + "\n\n").encode()


def response_content_from_sample(
    sample: dict[str, Any],
) -> tuple[int, dict[str, str], bytes]:
    """Build status, headers, and content tuple from a recorded sample."""

    response = sample.get("response", {})
    status_code = int(response.get("status_code", 200))
    headers = dict((response.get("headers") or {}).items())

    mode = response.get("mode", "json")
    if mode == "stream":
        events = response.get("events", [])
        content = serialize_sse_events(events)
        headers.setdefault("content-type", "text/event-stream")
    else:
        body = response.get("body", {})
        content = json.dumps(body).encode()
        headers.setdefault("content-type", "application/json")

    return status_code, headers, content
