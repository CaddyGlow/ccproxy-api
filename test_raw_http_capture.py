#!/usr/bin/env python3
"""Test that raw HTTP captures have readable response bodies when raw_http_enabled is true."""

import json
import os
import time
from pathlib import Path

import httpx


def test_raw_http_capture():
    """Test that raw HTTP captures are readable."""

    # Ensure raw HTTP tracing is enabled
    os.environ["PLUGINS__REQUEST_TRACER__RAW_HTTP_ENABLED"] = "true"

    print("\n=== Testing Raw HTTP Capture with Compression Disabled ===\n")

    # Make a test request to the API
    base_url = "http://127.0.0.1:8000"

    # Test the health endpoint
    print("Making test request to /health...")
    response = httpx.get(f"{base_url}/health")
    print(f"Response status: {response.status_code}")

    # Give the tracer time to write files
    time.sleep(0.5)

    # Check for raw HTTP capture files
    raw_log_dir = Path("/tmp/ccproxy/traces/raw")
    if not raw_log_dir.exists():
        print(f"Error: Raw log directory does not exist: {raw_log_dir}")
        return 1

    # Find the most recent trace file
    trace_files = sorted(raw_log_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not trace_files:
        print(f"Error: No trace files found in {raw_log_dir}")
        return 1

    latest_trace = trace_files[-1]
    print(f"\nExamining latest trace file: {latest_trace.name}")

    # Read and parse the trace file
    with open(latest_trace) as f:
        trace_data = json.load(f)

    # Check if response body is readable (not compressed)
    if "provider_response" in trace_data:
        response_body = trace_data["provider_response"].get("body", "")
        response_headers = trace_data["provider_response"].get("headers", {})

        print("\nProvider Response Headers:")
        for key, value in response_headers.items():
            if key.lower() in ["content-encoding", "accept-encoding"]:
                print(f"  {key}: {value}")

        print("\nProvider Response Body (first 200 chars):")
        print(f"  {response_body[:200]}")

        # Check if the body appears to be compressed (binary data)
        is_likely_compressed = any(
            ord(c) < 32 or ord(c) > 126
            for c in response_body[:100]
            if c not in "\n\r\t"
        )

        if is_likely_compressed:
            print(
                "\n✗ Response body appears to be compressed (contains non-printable characters)"
            )
            return 1
        else:
            print("\n✓ Response body is readable (not compressed)")
            return 0

    # Check client response as fallback
    if "client_response" in trace_data:
        response_body = trace_data["client_response"].get("body", "")
        print("\nClient Response Body (first 200 chars):")
        print(f"  {response_body[:200]}")

        # For health endpoint, we expect JSON
        try:
            json.loads(response_body)
            print("\n✓ Response body is valid JSON (not compressed)")
            return 0
        except json.JSONDecodeError:
            print("\n✗ Response body is not valid JSON (may be compressed)")
            return 1

    print("\nWarning: No response body found in trace")
    return 1


if __name__ == "__main__":
    # Check if server is running
    try:
        response = httpx.get("http://127.0.0.1:8000/health", timeout=1)
        exit_code = test_raw_http_capture()
    except httpx.ConnectError:
        print("Error: Server is not running. Please start the server with 'make dev'")
        exit_code = 1
    except Exception as e:
        print(f"Error: {e}")
        exit_code = 1

    exit(exit_code)
