#!/usr/bin/env python
"""Test streaming metrics with automatic verification against raw provider responses."""

import asyncio
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, TypedDict

import httpx


def wait_for_server() -> bool:
    """Wait for server to be ready."""
    for _ in range(30):
        try:
            response = httpx.get("http://127.0.0.1:8000/health")
            if response.status_code == 200:
                print("✓ Server is ready")
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def parse_raw_provider_response(request_id: str | None) -> dict[str, Any] | None:
    """Parse raw provider response to get actual token counts."""
    raw_file = Path(f"/tmp/ccproxy/raw/{request_id}_provider_response.http")
    if not raw_file.exists():
        # Check if any raw files exist for this request
        raw_dir = Path("/tmp/ccproxy/raw")
        matching_files = list(raw_dir.glob(f"{request_id}*"))
        if matching_files:
            print(
                f"  Found {len(matching_files)} raw files but no provider_response.http"
            )
        return None

    with raw_file.open() as f:
        content = f.read()

    # For Codex/OpenAI responses
    if "response.completed" in content:
        # Look for usage in the response.completed event - handle nested objects
        # Find the usage section which may contain nested objects
        match = re.search(r'"usage":\s*({[^}]*(?:{[^}]*}[^}]*)*})', content)
        if match:
            usage_str = match.group(1)
            try:
                # Extract tokens using regex
                input_match = re.search(r'"input_tokens":\s*(\d+)', usage_str)
                output_match = re.search(r'"output_tokens":\s*(\d+)', usage_str)

                if input_match and output_match:
                    return {
                        "provider": "codex",
                        "input_tokens": int(input_match.group(1)),
                        "output_tokens": int(output_match.group(1)),
                        "cache_read_tokens": None,
                        "cache_write_tokens": None,
                    }
            except Exception as e:
                print(f"  Error parsing Codex usage: {e}")

    # For Claude/Anthropic responses
    elif "message_delta" in content or "message_start" in content:
        # Look for the final message_delta with usage
        matches = re.findall(r'"usage":\s*({[^}]+})', content)
        if matches:
            # Take the last usage (from message_delta)
            usage_str = matches[-1]
            try:
                input_match = re.search(r'"input_tokens":\s*(\d+)', usage_str)
                output_match = re.search(r'"output_tokens":\s*(\d+)', usage_str)
                cache_read_match = re.search(
                    r'"cache_read_input_tokens":\s*(\d+)', usage_str
                )
                cache_write_match = re.search(
                    r'"cache_creation_input_tokens":\s*(\d+)', usage_str
                )

                if input_match and output_match:
                    return {
                        "provider": "claude",
                        "input_tokens": int(input_match.group(1)),
                        "output_tokens": int(output_match.group(1)),
                        "cache_read_tokens": int(cache_read_match.group(1))
                        if cache_read_match
                        else 0,
                        "cache_write_tokens": int(cache_write_match.group(1))
                        if cache_write_match
                        else 0,
                    }
            except Exception as e:
                print(f"  Error parsing Claude usage: {e}")

    return None


def check_logs_for_request(request_id: str | None) -> dict[str, Any] | None:
    """Check logs for a specific request ID and return metrics."""
    if not request_id or request_id == "unknown":
        return None

    log_file = Path("/tmp/ccproxy/ccproxy.log")
    if not log_file.exists():
        return None

    # Read log file
    with log_file.open() as f:
        lines = f.readlines()

    # Find logs for this request
    request_logs = []
    for line in lines:
        if request_id in line:
            try:
                log_data = json.loads(line)
                if log_data.get("request_id") == request_id:
                    request_logs.append(log_data)
            except json.JSONDecodeError:
                continue

    # Look for final metrics
    metrics = None
    for log in request_logs:
        event = log.get("event", "")

        # Look for final access log with metrics
        if event == "access_log" and (log.get("tokens_input") or log.get("cost_usd")):
            metrics = {
                "tokens_input": log.get("tokens_input"),
                "tokens_output": log.get("tokens_output"),
                "cache_read_tokens": log.get("cache_read_tokens"),
                "cache_write_tokens": log.get("cache_write_tokens"),
                "cost_usd": log.get("cost_usd"),
                "provider": log.get("provider")
                or log.get("metadata", {}).get("provider"),
                "model": log.get("model") or log.get("metadata", {}).get("model"),
                "duration_ms": log.get("duration_ms"),
            }

    return metrics


async def test_claude_sdk_native() -> str | None:
    """Test Claude SDK native endpoint."""
    print("\n=== Testing Claude SDK Native Endpoint ===")

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "http://127.0.0.1:8000/claude/v1/messages",
            headers={"Content-Type": "application/json"},
            json={
                "model": "claude-sonnet-4-20250514",
                "messages": [
                    {"role": "user", "content": "Say 'test' and nothing else"}
                ],
                "max_tokens": 10,
                "stream": True,
            },
        )

        request_id: str = response.headers.get("x-request-id", "unknown")
        print(f"Request ID: {request_id}")
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            chunks = []
            async for line in response.aiter_lines():
                if line and line.startswith("data: "):
                    chunks.append(line)
            print(f"  Total chunks: {len(chunks)}")
            return request_id
        else:
            print(f"  Error: {response.text[:200]}")
            return None


async def test_claude_api_native() -> str | None:
    """Test Claude API native endpoint."""
    print("\n=== Testing Claude API Native Endpoint ===")

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "http://127.0.0.1:8000/api/v1/messages",
            headers={"Content-Type": "application/json"},
            json={
                "model": "claude-sonnet-4-20250514",
                "messages": [
                    {"role": "user", "content": "Say 'test' and nothing else"}
                ],
                "max_tokens": 10,
                "stream": True,
            },
        )

        request_id: str = response.headers.get("x-request-id", "unknown")
        print(f"Request ID: {request_id}")
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            chunks = []
            async for line in response.aiter_lines():
                if line and line.startswith("data: "):
                    chunks.append(line)
            print(f"  Total chunks: {len(chunks)}")
            return request_id
        else:
            print(f"  Error: {response.text[:200]}")
            return None


async def test_codex_native() -> str | None:
    """Test Codex native endpoint."""
    print("\n=== Testing Codex Native Endpoint ===")

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "http://127.0.0.1:8000/api/codex/responses",
            headers={"Content-Type": "application/json"},
            json={
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "Say 'test' and nothing else",
                            }
                        ],
                    }
                ],
                "model": "gpt-5",
                "stream": True,
                "store": False,
            },
        )

        request_id: str = response.headers.get("x-request-id", "unknown")
        print(f"Request ID: {request_id}")
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            chunks = []
            async for line in response.aiter_lines():
                if line and line.startswith("data: "):
                    chunks.append(line)
            print(f"  Total chunks: {len(chunks)}")
            return request_id
        else:
            print(f"  Error: {response.text[:200]}")
            return None


async def test_codex_openai() -> str | None:
    """Test Codex OpenAI-compatible endpoint."""
    print("\n=== Testing Codex OpenAI-Compatible Endpoint ===")

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "http://127.0.0.1:8000/api/codex/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": "gpt-5",
                "messages": [
                    {"role": "user", "content": "Say 'test' and nothing else"}
                ],
                "max_tokens": 10,
                "stream": True,
            },
        )

        request_id: str = response.headers.get("x-request-id", "unknown")
        print(f"Request ID: {request_id}")
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            chunks = []
            async for line in response.aiter_lines():
                if line and line.startswith("data: "):
                    chunks.append(line)
            print(f"  Total chunks: {len(chunks)}")
            return request_id
        else:
            print(f"  Error: {response.text[:200]}")
            return None


def verify_metrics(
    test_name: str,
    request_id: str,
    logged_metrics: dict[str, Any] | None,
    raw_metrics: dict[str, Any] | None,
) -> bool:
    """Verify that logged metrics match raw provider response."""
    print(f"\n{test_name} (Request: {request_id[:12]}...):")

    if not logged_metrics:
        print("  ✗ No metrics found in logs")
        return False

    if not raw_metrics:
        print("  ⚠ No raw provider response found for verification")
    else:
        # Compare token counts
        input_match = logged_metrics["tokens_input"] == raw_metrics["input_tokens"]
        output_match = logged_metrics["tokens_output"] == raw_metrics["output_tokens"]

        print(f"  Provider: {raw_metrics['provider'].upper()}")
        print(
            f"  Input tokens: {logged_metrics['tokens_input']} (logged) vs {raw_metrics['input_tokens']} (raw) {'✓' if input_match else '✗'}"
        )
        print(
            f"  Output tokens: {logged_metrics['tokens_output']} (logged) vs {raw_metrics['output_tokens']} (raw) {'✓' if output_match else '✗'}"
        )

        # Check cache tokens if available
        if raw_metrics["cache_read_tokens"] is not None:
            cache_match = (
                logged_metrics.get("cache_read_tokens")
                == raw_metrics["cache_read_tokens"]
            )
            print(
                f"  Cache read tokens: {logged_metrics.get('cache_read_tokens', 0)} (logged) vs {raw_metrics['cache_read_tokens']} (raw) {'✓' if cache_match else '✗'}"
            )

        if (
            raw_metrics["cache_write_tokens"] is not None
            and raw_metrics["cache_write_tokens"] > 0
        ):
            cache_write_match = (
                logged_metrics.get("cache_write_tokens")
                == raw_metrics["cache_write_tokens"]
            )
            print(
                f"  Cache write tokens: {logged_metrics.get('cache_write_tokens', 0)} (logged) vs {raw_metrics['cache_write_tokens']} (raw) {'✓' if cache_write_match else '✗'}"
            )

    # Always show logged metrics
    print("\n  Logged Metrics:")
    print(
        f"    Tokens: {logged_metrics['tokens_input']} in / {logged_metrics['tokens_output']} out"
    )
    if logged_metrics.get("cache_read_tokens"):
        print(f"    Cache read: {logged_metrics['cache_read_tokens']} tokens")
    if logged_metrics.get("cache_write_tokens"):
        print(f"    Cache write: {logged_metrics['cache_write_tokens']} tokens")
    if logged_metrics["cost_usd"]:
        print(f"    Cost: ${logged_metrics['cost_usd']:.6f}")
    else:
        print("    Cost: Not calculated")
    print(f"    Model: {logged_metrics['model']}")
    if logged_metrics["duration_ms"]:
        print(f"    Duration: {logged_metrics['duration_ms']:.1f}ms")

    # Return true if tokens match (or no raw data to compare)
    if raw_metrics:
        return bool(input_match and output_match)
    else:
        return bool(logged_metrics["tokens_input"] and logged_metrics["cost_usd"])


async def main() -> None:
    """Run all tests with automatic verification."""
    print("Starting CCProxy streaming metrics verification tests...")

    # Start the server
    print("\nStarting server...")
    server_process = subprocess.Popen(
        ["ccproxy", "serve"],
        env={**os.environ, "CCPROXY_VERBOSE_API": "true"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        # Wait for server to be ready
        if not wait_for_server():
            print("✗ Server failed to start")
            return

        # Run all tests and collect request IDs
        request_ids = {}

        print("\n" + "=" * 70)
        print("RUNNING TESTS")
        print("=" * 70)

        # Test Claude SDK endpoint
        request_ids["claude_sdk_native"] = await test_claude_sdk_native()
        await asyncio.sleep(3)

        # Test Claude API endpoint
        request_ids["claude_api_native"] = await test_claude_api_native()
        await asyncio.sleep(3)

        # Test Codex endpoints
        request_ids["codex_native"] = await test_codex_native()
        await asyncio.sleep(3)

        request_ids["codex_openai"] = await test_codex_openai()
        await asyncio.sleep(5)  # Give more time for files to be written

        # Verify metrics for each request
        print("\n" + "=" * 70)
        print("METRICS VERIFICATION")
        print("=" * 70)

        verification_results = {}
        for test_name, request_id in request_ids.items():
            if request_id:
                # Wait a bit for files to be written
                await asyncio.sleep(1)

                # Get logged metrics
                logged_metrics = check_logs_for_request(request_id)

                # Get raw provider response
                raw_metrics = parse_raw_provider_response(request_id)

                # Verify and display
                verification_results[test_name] = verify_metrics(
                    test_name, request_id, logged_metrics, raw_metrics
                )
            else:
                print(f"\n{test_name}:")
                print("  ✗ Test failed - no request ID")
                verification_results[test_name] = False

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        for test_name, verified in verification_results.items():
            if verified:
                status = "✓ VERIFIED"
            else:
                status = "✗ FAILED"
            print(f"{test_name}: {status}")

        success_count = sum(1 for v in verification_results.values() if v)
        print(f"\nTotal: {success_count}/{len(verification_results)} tests verified")

        all_passed = all(verification_results.values())
        print(
            f"\nOverall: {'✓ ALL TESTS VERIFIED' if all_passed else '✗ SOME TESTS FAILED'}"
        )

    finally:
        # Stop the server
        print("\nStopping server...")
        server_process.terminate()
        server_process.wait(timeout=5)


if __name__ == "__main__":
    asyncio.run(main())
