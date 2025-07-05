#!/usr/bin/env python3
"""Compare speed with and without connection pooling."""

import asyncio
import json
import statistics
import time
from typing import Any, Dict, List

import httpx


# Server configurations
POOL_ENABLED_URL = "http://localhost:8000"
POOL_DISABLED_URL = (
    "http://localhost:8001"  # Run a second instance with pooling disabled
)


async def make_request(client: httpx.AsyncClient, base_url: str) -> float:
    """Make a single request and return the response time."""
    data = {
        "model": "claude-3-5-sonnet-20241022",
        "messages": [{"role": "user", "content": "Say 'hello' in one word"}],
        "max_tokens": 10,
    }

    start_time = time.time()

    try:
        response = await client.post(
            f"{base_url}/v1/messages",
            json=data,
            headers={"Content-Type": "application/json"},
            timeout=30.0,
        )

        if response.status_code == 200:
            end_time = time.time()
            return end_time - start_time
        else:
            print(f"Error: {response.status_code}")
            return -1
    except Exception as e:
        print(f"Request failed: {e}")
        return -1


async def run_benchmark(base_url: str, num_requests: int, label: str) -> dict[str, Any]:
    """Run benchmark with specified number of requests."""
    print(f"\n{label} Benchmark")
    print("=" * 60)

    response_times: list[float] = []

    async with httpx.AsyncClient() as client:
        # Warmup request
        print("Warming up...")
        await make_request(client, base_url)

        # Sequential requests
        print(f"Running {num_requests} sequential requests...")
        sequential_start = time.time()

        for i in range(num_requests):
            response_time = await make_request(client, base_url)
            if response_time > 0:
                response_times.append(response_time)
                print(f"  Request {i + 1}: {response_time:.3f}s")

        sequential_total = time.time() - sequential_start

        # Concurrent requests
        print(f"\nRunning {num_requests} concurrent requests...")
        concurrent_start = time.time()

        tasks = [make_request(client, base_url) for _ in range(num_requests)]
        concurrent_times = await asyncio.gather(*tasks)
        concurrent_times = [t for t in concurrent_times if t > 0]

        concurrent_total = time.time() - concurrent_start

    # Calculate statistics
    if response_times:
        results: dict[str, Any] = {
            "label": label,
            "sequential": {
                "total_time": sequential_total,
                "avg_response_time": statistics.mean(response_times),
                "min_response_time": min(response_times),
                "max_response_time": max(response_times),
                "median_response_time": statistics.median(response_times),
                "requests_per_second": len(response_times) / sequential_total,
            },
            "concurrent": {
                "total_time": concurrent_total,
                "avg_response_time": statistics.mean(concurrent_times)
                if concurrent_times
                else 0,
                "requests_per_second": len(concurrent_times) / concurrent_total
                if concurrent_total > 0
                else 0,
            },
        }

        print(f"\nResults for {label}:")
        seq_results: dict[str, Any] = results["sequential"]
        conc_results: dict[str, Any] = results["concurrent"]
        print(f"  Sequential total time: {seq_results['total_time']:.2f}s")
        print(f"  Sequential avg response: {seq_results['avg_response_time']:.3f}s")
        print(f"  Sequential RPS: {seq_results['requests_per_second']:.2f}")
        print(f"  Concurrent total time: {conc_results['total_time']:.2f}s")
        print(f"  Concurrent RPS: {conc_results['requests_per_second']:.2f}")

        return results
    else:
        print("No successful requests")
        return {}


async def get_pool_stats(base_url: str) -> dict[str, Any]:
    """Get pool statistics from the server."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{base_url}/pool/stats")
            if response.status_code == 200:
                return response.json()  # type: ignore[no-any-return]
        except Exception:
            pass
    return {}


async def main() -> None:
    """Run the speed comparison."""
    print("Connection Pool Speed Comparison")
    print("=" * 60)
    print("\nThis test compares the performance of the API with and without")
    print("connection pooling enabled.")
    print("\nMake sure to run two instances of the server:")
    print("1. Port 8000 with pooling enabled (default)")
    print("2. Port 8001 with pooling disabled (POOL_SETTINGS__ENABLED=false)")
    print("\nExample for second instance:")
    print("POOL_SETTINGS__ENABLED=false PORT=8001 uv run python main.py")

    input("\nPress Enter to start the benchmark...")

    num_requests = 10

    # Check if both servers are running
    print("\nChecking server availability...")
    async with httpx.AsyncClient() as client:
        try:
            await client.get(f"{POOL_ENABLED_URL}/health")
            print("✓ Pool-enabled server is running on port 8000")
        except:
            print("✗ Pool-enabled server is not running on port 8000")
            return

        try:
            await client.get(f"{POOL_DISABLED_URL}/health")
            print("✓ Pool-disabled server is running on port 8001")
        except:
            print("✗ Pool-disabled server is not running on port 8001")
            print("\nPlease start a second instance with pooling disabled:")
            print("POOL_SETTINGS__ENABLED=false PORT=8001 uv run python main.py")
            return

    # Get initial pool stats
    pool_stats_before = await get_pool_stats(POOL_ENABLED_URL)
    if pool_stats_before.get("pool_enabled"):
        print(f"\nPool stats before test: {json.dumps(pool_stats_before, indent=2)}")

    # Run benchmarks
    results = []

    # Test with pooling enabled
    pool_enabled_results = await run_benchmark(
        POOL_ENABLED_URL, num_requests, "WITH Connection Pool"
    )
    results.append(pool_enabled_results)

    # Get pool stats after
    pool_stats_after = await get_pool_stats(POOL_ENABLED_URL)
    if pool_stats_after.get("pool_enabled"):
        print(f"\nPool stats after test: {json.dumps(pool_stats_after, indent=2)}")
        stats = pool_stats_after.get("stats", {})
        print(f"\nConnections reused: {stats.get('connections_reused', 0)}")
        print(f"Connections created: {stats.get('connections_created', 0)}")

    # Test with pooling disabled
    pool_disabled_results = await run_benchmark(
        POOL_DISABLED_URL, num_requests, "WITHOUT Connection Pool"
    )
    results.append(pool_disabled_results)

    # Compare results
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)

    if len(results) == 2 and results[0] and results[1]:
        pool_seq = results[0]["sequential"]
        no_pool_seq = results[1]["sequential"]

        # Calculate improvements
        time_saved = no_pool_seq["avg_response_time"] - pool_seq["avg_response_time"]
        time_saved_pct = (time_saved / no_pool_seq["avg_response_time"]) * 100

        rps_improvement = (
            pool_seq["requests_per_second"] - no_pool_seq["requests_per_second"]
        )
        rps_improvement_pct = (
            rps_improvement / no_pool_seq["requests_per_second"]
        ) * 100

        print("\nSequential Requests:")
        print("  Average response time:")
        print(f"    With pool:    {pool_seq['avg_response_time']:.3f}s")
        print(f"    Without pool: {no_pool_seq['avg_response_time']:.3f}s")
        print(f"    Improvement:  {time_saved:.3f}s ({time_saved_pct:.1f}% faster)")

        print("\n  Requests per second:")
        print(f"    With pool:    {pool_seq['requests_per_second']:.2f} RPS")
        print(f"    Without pool: {no_pool_seq['requests_per_second']:.2f} RPS")
        print(
            f"    Improvement:  {rps_improvement:.2f} RPS ({rps_improvement_pct:.1f}% higher)"
        )

        # Concurrent comparison
        pool_conc = results[0]["concurrent"]
        no_pool_conc = results[1]["concurrent"]

        conc_rps_improvement = (
            pool_conc["requests_per_second"] - no_pool_conc["requests_per_second"]
        )
        conc_rps_improvement_pct = (
            conc_rps_improvement / no_pool_conc["requests_per_second"]
        ) * 100

        print("\nConcurrent Requests:")
        print("  Requests per second:")
        print(f"    With pool:    {pool_conc['requests_per_second']:.2f} RPS")
        print(f"    Without pool: {no_pool_conc['requests_per_second']:.2f} RPS")
        print(
            f"    Improvement:  {conc_rps_improvement:.2f} RPS ({conc_rps_improvement_pct:.1f}% higher)"
        )

        print("\nSUMMARY:")
        print(
            f"  Connection pooling provides ~{time_saved_pct:.0f}% faster response times"
        )
        print(
            f"  and ~{rps_improvement_pct:.0f}% higher throughput for sequential requests."
        )
        print(
            f"  For concurrent requests, throughput is ~{conc_rps_improvement_pct:.0f}% higher."
        )


if __name__ == "__main__":
    asyncio.run(main())
