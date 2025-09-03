#!/usr/bin/env python
"""Test streaming metrics for all provider endpoints."""

import asyncio
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

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

        # Capture request ID from headers
        request_id: str = response.headers.get("x-request-id", "unknown")
        print(f"Request ID: {request_id}")
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            chunks = []
            async for line in response.aiter_lines():
                if line and line.startswith("data: "):
                    chunks.append(line)
                    if "message_start" in line or "message_delta" in line:
                        print(f"  Chunk: {line[:100]}...")
            print(f"  Total chunks: {len(chunks)}")
            return request_id
        else:
            print(f"  Error: {response.text}")
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

        # Capture request ID from headers
        request_id: str = response.headers.get("x-request-id", "unknown")
        print(f"Request ID: {request_id}")
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            chunks = []
            async for line in response.aiter_lines():
                if line and line.startswith("data: "):
                    chunks.append(line)
                    if "message_start" in line or "message_delta" in line:
                        print(f"  Chunk: {line[:100]}...")
            print(f"  Total chunks: {len(chunks)}")
            return request_id
        else:
            print(f"  Error: {response.text}")
            return None


async def test_claude_sdk_openai() -> str | None:
    """Test Claude SDK OpenAI-compatible endpoint."""
    print("\n=== Testing Claude SDK OpenAI-Compatible Endpoint ===")

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "http://127.0.0.1:8000/claude/v1/chat/completions",
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

        # Capture request ID from headers
        request_id: str = response.headers.get("x-request-id", "unknown")
        print(f"Request ID: {request_id}")
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            chunks = []
            async for line in response.aiter_lines():
                if line and line.startswith("data: "):
                    chunks.append(line)
                    if "chat.completion" in line:
                        print(f"  Chunk: {line[:100]}...")
            print(f"  Total chunks: {len(chunks)}")
            return request_id
        else:
            print(f"  Error: {response.text}")
            return None


async def test_claude_api_openai() -> str | None:
    """Test Claude API OpenAI-compatible endpoint."""
    print("\n=== Testing Claude API OpenAI-Compatible Endpoint ===")

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "http://127.0.0.1:8000/api/v1/chat/completions",
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

        # Capture request ID from headers
        request_id: str = response.headers.get("x-request-id", "unknown")
        print(f"Request ID: {request_id}")
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            chunks = []
            async for line in response.aiter_lines():
                if line and line.startswith("data: "):
                    chunks.append(line)
                    if "chat.completion" in line:
                        print(f"  Chunk: {line[:100]}...")
            print(f"  Total chunks: {len(chunks)}")
            return request_id
        else:
            print(f"  Error: {response.text}")
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

        # Capture request ID from headers
        request_id: str = response.headers.get("x-request-id", "unknown")
        print(f"Request ID: {request_id}")
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            chunks = []
            async for line in response.aiter_lines():
                if line and line.startswith("data: "):
                    chunks.append(line)
                    if "response.text" in line or "response.completed" in line:
                        print(f"  Chunk type: {line[6:50]}...")
            print(f"  Total chunks: {len(chunks)}")
            return request_id
        else:
            print(f"  Error: {response.text}")
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

        # Capture request ID from headers
        request_id: str = response.headers.get("x-request-id", "unknown")
        print(f"Request ID: {request_id}")
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            chunks = []
            async for line in response.aiter_lines():
                if line and line.startswith("data: "):
                    chunks.append(line)
                    if "chat.completion" in line:
                        print(f"  Chunk: {line[:100]}...")
            print(f"  Total chunks: {len(chunks)}")
            return request_id
        else:
            print(f"  Error: {response.text}")
            return None


def check_logs_for_request(request_id: str | None) -> dict[str, Any] | None:
    """Check logs for a specific request ID."""
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
                "cost_usd": log.get("cost_usd"),
                "provider": log.get("provider")
                or log.get("metadata", {}).get("provider"),
                "model": log.get("model") or log.get("metadata", {}).get("model"),
                "duration_ms": log.get("duration_ms"),
            }

    return metrics


async def main() -> None:
    """Run all tests."""
    print("Starting CCProxy streaming metrics tests...")

    # Record start time for log checking
    start_time = time.strftime("%Y-%m-%dT%H:%M:%S")

    # Start the server
    print("\nStarting server...")
    server_process = subprocess.Popen(
        ["ccproxy", "serve"],
        env={**os.environ, "LOGGING__VERBOSE_API": "true"},
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

        # Test Claude SDK endpoints (using /claude/v1/)
        print("\n" + "=" * 70)
        request_ids["claude_sdk_native"] = await test_claude_sdk_native()
        await asyncio.sleep(
            2
        )  # Give time for request to complete and logs to be written

        request_ids["claude_sdk_openai"] = await test_claude_sdk_openai()
        await asyncio.sleep(2)

        # Test Claude API endpoints (using /api/v1/)
        request_ids["claude_api_native"] = await test_claude_api_native()
        await asyncio.sleep(2)

        request_ids["claude_api_openai"] = await test_claude_api_openai()
        await asyncio.sleep(2)

        # Test Codex endpoints
        request_ids["codex_native"] = await test_codex_native()
        await asyncio.sleep(2)

        request_ids["codex_openai"] = await test_codex_openai()
        await asyncio.sleep(3)  # Give extra time for final logs

        # Check logs for metrics for each request
        print("\n" + "=" * 70)
        print("METRICS FROM LOGS")
        print("=" * 70)

        all_metrics = {}
        for test_name, request_id in request_ids.items():
            if request_id:
                metrics = check_logs_for_request(request_id)
                all_metrics[test_name] = metrics

                print(f"\n{test_name} (Request: {request_id[:12]}...):")
                if metrics:
                    print(
                        f"  ✓ Tokens: {metrics['tokens_input']} in / {metrics['tokens_output']} out"
                    )
                    if metrics["cost_usd"]:
                        print(f"  ✓ Cost: ${metrics['cost_usd']:.6f}")
                    else:
                        print("  ⚠ Cost: Not calculated")
                    print(f"  ✓ Model: {metrics['model']}")
                    print(
                        f"  ✓ Duration: {metrics['duration_ms']:.1f}ms"
                        if metrics["duration_ms"]
                        else "  Duration: N/A"
                    )
                else:
                    print("  ✗ No metrics found in logs")
            else:
                print(f"\n{test_name}:")
                print("  ✗ Test failed - no request ID")
                all_metrics[test_name] = None

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        success_count = 0
        for test_name, metrics in all_metrics.items():
            if metrics and metrics.get("tokens_input") and metrics.get("cost_usd"):
                status = "✓ COMPLETE"
                success_count += 1
            elif metrics and metrics.get("tokens_input"):
                status = "⚠ PARTIAL (no cost)"
            else:
                status = "✗ FAILED"
            print(f"{test_name}: {status}")

        print(
            f"\nTotal: {success_count}/{len(all_metrics)} tests with complete metrics"
        )

        all_passed = success_count == len(all_metrics)
        print(
            f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS INCOMPLETE'}"
        )

    finally:
        # Stop the server
        print("\nStopping server...")
        server_process.terminate()
        server_process.wait(timeout=5)


if __name__ == "__main__":
    asyncio.run(main())
