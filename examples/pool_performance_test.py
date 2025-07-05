#!/usr/bin/env python3
"""Test connection pool performance with a single server instance."""

import asyncio
import json
import statistics
import time
from typing import Any

import httpx


# Configuration
SERVER_URL = "http://localhost:8000"
NUM_WARMUP = 2
NUM_REQUESTS = 20


async def make_request(
    client: httpx.AsyncClient, request_num: int
) -> tuple[float, bool]:
    """Make a single request and return (response_time, is_first_request)."""
    data = {
        "model": "claude-3-5-sonnet-20241022",
        "messages": [
            {
                "role": "user",
                "content": f"What is {request_num} + 1? Answer in one number only.",
            }
        ],
        "max_tokens": 10,
    }

    start_time = time.time()

    try:
        response = await client.post(
            f"{SERVER_URL}/v1/messages",
            json=data,
            headers={"Content-Type": "application/json"},
            timeout=30.0,
        )

        if response.status_code == 200:
            end_time = time.time()
            return end_time - start_time, request_num == 1
        else:
            print(f"Error: {response.status_code}")
            return -1, False
    except Exception as e:
        print(f"Request failed: {e}")
        return -1, False


async def get_pool_stats() -> dict[str, Any]:
    """Get current pool statistics."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{SERVER_URL}/pool/stats")
            if response.status_code == 200:
                return response.json()  # type: ignore[no-any-return]
        except Exception:
            pass
    return {}


async def main() -> None:
    """Run the performance test."""
    print("Connection Pool Performance Test")
    print("=" * 60)

    # Check if server is running
    print("Checking server availability...")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{SERVER_URL}/health")
            if response.status_code != 200:
                print("Server is not healthy")
                return
            print("✓ Server is running")
        except Exception as e:
            print(f"✗ Server is not running: {e}")
            return

    # Get initial pool stats
    initial_stats = await get_pool_stats()
    pool_enabled = initial_stats.get("pool_enabled", False)

    if pool_enabled:
        print("\n✓ Connection pooling is ENABLED")
        stats = initial_stats.get("stats", {})
        print(f"  Current pool size: {stats.get('total_connections', 0)}")
        print(f"  Connections created: {stats.get('connections_created', 0)}")
        print(f"  Connections reused: {stats.get('connections_reused', 0)}")
    else:
        print("\n✗ Connection pooling is DISABLED")
        print("  Each request will create a new Claude subprocess")

    print(f"\nRunning {NUM_REQUESTS} requests (after {NUM_WARMUP} warmup requests)...")
    print("-" * 60)

    response_times: list[float] = []
    first_request_time = 0.0

    async with httpx.AsyncClient() as client:
        # Warmup requests
        print("Warming up...")
        for _ in range(NUM_WARMUP):
            await make_request(client, 0)

        # Actual test requests
        print("\nRunning test requests:")
        for i in range(1, NUM_REQUESTS + 1):
            response_time, is_first = await make_request(client, i)
            if response_time > 0:
                response_times.append(response_time)
                if is_first:
                    first_request_time = response_time

                # Show progress with timing
                marker = "*" if response_time < 1.0 else ""
                print(f"  Request {i:2d}: {response_time:6.3f}s {marker}")

                # Small delay to make logs easier to read
                if i < NUM_REQUESTS:
                    await asyncio.sleep(0.1)

    # Get final pool stats
    final_stats = await get_pool_stats()

    # Analyze results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    if response_times:
        # Remove the first request from averages as it's typically slower
        subsequent_times = (
            response_times[1:] if len(response_times) > 1 else response_times
        )

        print("\nResponse Time Statistics:")
        print(f"  First request:  {response_times[0]:.3f}s")
        print(f"  Subsequent avg: {statistics.mean(subsequent_times):.3f}s")
        print(f"  Minimum:        {min(response_times):.3f}s")
        print(f"  Maximum:        {max(response_times):.3f}s")
        print(f"  Median:         {statistics.median(response_times):.3f}s")

        if len(response_times) > 1:
            print(f"  Std deviation:  {statistics.stdev(response_times):.3f}s")

        # Show improvement from first to subsequent
        if len(subsequent_times) > 0:
            improvement = response_times[0] - statistics.mean(subsequent_times)
            improvement_pct = (improvement / response_times[0]) * 100
            print("\n  Speed improvement after first request:")
            print(f"    Time saved: {improvement:.3f}s ({improvement_pct:.1f}% faster)")

    if pool_enabled and final_stats.get("pool_enabled"):
        print("\nConnection Pool Statistics:")
        initial = initial_stats.get("stats", {})
        final = final_stats.get("stats", {})

        created = final.get("connections_created", 0) - initial.get(
            "connections_created", 0
        )
        reused = final.get("connections_reused", 0) - initial.get(
            "connections_reused", 0
        )

        print(f"  Connections created during test: {created}")
        print(f"  Connections reused during test:  {reused}")
        print(f"  Current pool size: {final.get('total_connections', 0)}")
        print(f"  Available connections: {final.get('available_connections', 0)}")

        if reused > 0:
            print("\n✓ Connection pooling is working!")
            print(
                f"  {reused} out of {len(response_times)} requests reused existing connections"
            )
            reuse_rate = (reused / len(response_times)) * 100
            print(f"  Reuse rate: {reuse_rate:.1f}%")
        else:
            print("\n⚠ No connection reuse detected")
            print("  This might indicate the pool is not working as expected")

    # Performance summary
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)

    if pool_enabled:
        print("\nWith connection pooling enabled:")
        print("  • First request creates a new connection (slower)")
        print("  • Subsequent requests reuse connections (faster)")
        print("  • Typical speedup: 200-500ms per request")
        print("  • Best for: High-traffic APIs, multiple requests")
    else:
        print("\nWith connection pooling disabled:")
        print("  • Every request creates a new subprocess")
        print("  • Consistent but slower response times")
        print("  • More resource intensive")

    print("\nTo compare with pooling disabled, run:")
    print(
        "  POOL_SETTINGS__ENABLED=false uv run python examples/pool_performance_test.py"
    )


if __name__ == "__main__":
    asyncio.run(main())
