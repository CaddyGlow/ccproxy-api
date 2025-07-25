#!/usr/bin/env python3
"""Benchmark different approaches to reduce Claude CLI startup time."""

import asyncio
import os
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from statistics import mean, stdev
from typing import Any


# Number of runs for each benchmark
BENCHMARK_RUNS = 20  # Using version check which should be fast
WARMUP_RUNS = 5

# Test prompt - using version command to avoid LLM requests
TEST_PROMPT = None  # Not needed for version command


class StartupBenchmark:
    """Benchmark different startup approaches for Claude CLI."""

    def __init__(self) -> None:
        self.results: dict[str, list[float]] = {}
        # Use the actual Claude CLI path instead of the npm symlink
        self.claude_path = "/home/rick/.claude/local/claude"
        print(f"Using Claude CLI at: {self.claude_path}")

    def _find_claude_cli(self) -> str:
        """Find the Claude CLI executable."""
        # Try to find claude in PATH
        try:
            result = subprocess.run(
                ["which", "claude"], capture_output=True, text=True, check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            # Check common locations
            common_paths = [
                Path.home() / ".claude" / "local" / "claude",
                Path("/usr/local/bin/claude"),
                Path("/opt/homebrew/bin/claude"),
            ]
            for path in common_paths:
                if path.exists() and path.is_file():
                    return str(path)
            raise RuntimeError("Claude CLI not found") from None

    def run_benchmark(
        self, name: str, runs: int = BENCHMARK_RUNS
    ) -> Callable[[Callable[..., Any]], Callable[..., None]]:
        """Decorator to run a benchmark multiple times."""

        def decorator(func: Callable[..., Any]) -> Callable[..., None]:
            def wrapper(*args: Any, **kwargs: Any) -> None:
                print(f"\nRunning benchmark: {name}")
                times = []

                # Warmup runs
                print(f"Warming up ({WARMUP_RUNS} runs)...")
                for i in range(WARMUP_RUNS):
                    start = time.perf_counter()
                    result = func(*args, **kwargs)
                    if asyncio.iscoroutine(result):
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(result)
                        loop.close()
                    elapsed = time.perf_counter() - start
                    print(f"  Warmup {i + 1}: {elapsed:.3f}s")

                # Actual benchmark runs
                print(f"Benchmarking ({runs} runs)...")
                for i in range(runs):
                    start = time.perf_counter()
                    result = func(*args, **kwargs)
                    if asyncio.iscoroutine(result):
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(result)
                        loop.close()
                    elapsed = time.perf_counter() - start
                    times.append(elapsed)
                    print(f"  Run {i + 1}: {elapsed:.3f}s")

                self.results[name] = times
                avg = mean(times)
                std = stdev(times) if len(times) > 1 else 0
                print(f"Average: {avg:.3f}s ± {std:.3f}s")

            return wrapper

        return decorator

    def benchmark_normal_subprocess(self) -> None:
        """Benchmark normal subprocess execution."""

        @self.run_benchmark("Normal Subprocess")
        def run() -> None:
            cmd = [self.claude_path, "-v"]
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=2.0,  # 2 second timeout for version check
            )
            if proc.returncode != 0:
                print(f"Error running command: {' '.join(cmd)}")
                print(f"Return code: {proc.returncode}")
                print(f"Stderr: {proc.stderr}")
                print(f"Stdout: {proc.stdout}")

        # Actually run the benchmark
        run()

    def benchmark_node_direct(self) -> None:
        """Benchmark running Node.js directly with Claude CLI."""

        @self.run_benchmark(
            "Node.js Direct Execution", runs=5
        )  # Reduced runs since this will fail
        def run() -> None:
            # Find the actual Node.js script, not the shell wrapper
            # The claude shell script wraps the actual Node.js claude script
            actual_claude_path = (
                Path(self.claude_path).parent / "node_modules" / ".bin" / "claude"
            )

            if not actual_claude_path.exists():
                print(f"Actual Claude Node.js script not found at {actual_claude_path}")
                return

            # Find node executable
            node_path = subprocess.run(
                ["which", "node"], capture_output=True, text=True
            ).stdout.strip()

            if not node_path:
                print("Node.js not found, skipping")
                return

            # Run claude directly with node
            proc = subprocess.run(
                [node_path, str(actual_claude_path), "-v"],
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            if proc.returncode != 0:
                print(f"Error: {proc.stderr}")

        run()

    def benchmark_preloaded_env(self) -> None:
        """Benchmark with preloaded environment variables."""

        @self.run_benchmark("Preloaded Environment")
        def run() -> None:
            env = os.environ.copy()
            # Pre-set commonly used environment variables
            env["NODE_ENV"] = "production"
            env["NODE_NO_WARNINGS"] = "1"
            env["CLAUDE_NON_INTERACTIVE"] = "1"

            proc = subprocess.run(
                [self.claude_path, "-v"],
                capture_output=True,
                text=True,
                env=env,
                timeout=2.0,
            )
            if proc.returncode != 0:
                print(f"Error: {proc.stderr}")

        run()

    def benchmark_subprocess_popen(self) -> None:
        """Benchmark using Popen with pipes."""

        @self.run_benchmark("Subprocess Popen with Pipes")
        def run() -> None:
            proc = subprocess.Popen(
                [self.claude_path, "-v"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, stderr = proc.communicate()
            if proc.returncode != 0:
                print(f"Error: {stderr}")

        run()

    def benchmark_asyncio_subprocess(self) -> None:
        """Benchmark using asyncio subprocess."""

        @self.run_benchmark("Asyncio Subprocess")
        async def run() -> None:
            proc = await asyncio.create_subprocess_exec(
                self.claude_path,
                "-v",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                print(f"Error: {stderr.decode()}")

        # Actually run the benchmark
        run()

    def benchmark_node_snapshot(self) -> None:
        """Benchmark using Node.js startup snapshots (if available)."""
        print("\nPreparing Node.js snapshot benchmark...")

        # Check Node.js version
        try:
            node_version = subprocess.run(
                ["node", "--version"], capture_output=True, text=True
            ).stdout.strip()
            print(f"Node.js version: {node_version}")

            # Extract major version
            major_version = int(node_version.split(".")[0].replace("v", ""))
            if major_version < 18:
                print(f"Node.js snapshots require v18+, found {node_version}")
                return
        except Exception as e:
            print(f"Could not check Node.js version: {e}")
            return

        # Create a simple snapshot script
        snapshot_script = Path("claude-snapshot.js")
        snapshot_blob = Path("claude-snapshot.blob")

        # Write snapshot creation script
        snapshot_script.write_text("""
// Preload common modules
require('fs');
require('path');
require('util');
require('stream');
require('events');
require('crypto');
require('http');
require('https');
require('url');
require('querystring');
require('child_process');

console.log('Snapshot created with preloaded modules');
""")

        try:
            # Create the snapshot
            print("Creating Node.js snapshot...")
            proc = subprocess.run(
                [
                    "node",
                    "--snapshot-blob",
                    str(snapshot_blob),
                    "--build-snapshot",
                    str(snapshot_script),
                ],
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                print(f"Failed to create snapshot: {proc.stderr}")
                return

            # Benchmark with snapshot
            @self.run_benchmark("Node.js with Snapshot")
            def run() -> None:
                proc = subprocess.run(
                    [
                        "node",
                        "--snapshot-blob",
                        str(snapshot_blob),
                        self.claude_path,
                        "-v",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=2.0,
                )
                if proc.returncode != 0:
                    print(f"Error: {proc.stderr}")

            # Actually run the benchmark
            run()

        finally:
            # Cleanup
            if snapshot_script.exists():
                snapshot_script.unlink()
            if snapshot_blob.exists():
                snapshot_blob.unlink()

    def benchmark_repeated_calls(self) -> None:
        """Benchmark repeated calls to see if OS caching helps."""
        print("\nBenchmarking repeated calls to test OS caching effect...")

        times = []
        for i in range(10):  # Reduced from 20
            start = time.perf_counter()
            proc = subprocess.run(
                [self.claude_path, "-v"],
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            elapsed = time.perf_counter() - start
            times.append(elapsed)

            if i % 5 == 0:
                avg_last_5 = mean(times[-5:]) if len(times) >= 5 else elapsed
                print(
                    f"  Calls {i + 1}: {elapsed:.3f}s (avg last 5: {avg_last_5:.3f}s)"
                )

        # Compare first 5 vs last 5
        first_5_avg = mean(times[:5])
        last_5_avg = mean(times[-5:])
        improvement = ((first_5_avg - last_5_avg) / first_5_avg) * 100

        print(f"\nFirst 5 calls average: {first_5_avg:.3f}s")
        print(f"Last 5 calls average: {last_5_avg:.3f}s")
        print(f"Improvement from caching: {improvement:.1f}%")

    def print_summary(self) -> None:
        """Print a summary of all benchmark results."""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)

        if not self.results:
            print("No benchmark results collected")
            return

        # Sort by average time
        sorted_results = sorted(self.results.items(), key=lambda x: mean(x[1]))

        # Find baseline (normal subprocess)
        baseline_avg = mean(self.results.get("Normal Subprocess", [1.0]))

        print(f"\n{'Method':<30} {'Avg Time':<12} {'Std Dev':<10} {'vs Baseline':<15}")
        print("-" * 67)

        for name, times in sorted_results:
            avg = mean(times)
            std = stdev(times) if len(times) > 1 else 0
            speedup = (baseline_avg / avg - 1) * 100
            speedup_str = f"+{speedup:.1f}%" if speedup > 0 else f"{speedup:.1f}%"

            print(f"{name:<30} {avg:.3f}s{'':<6} ±{std:.3f}s{'':<3} {speedup_str:<15}")

        print("\nNote: Positive percentages indicate improvement over baseline")


def main() -> None:
    """Run all benchmarks."""
    benchmark = StartupBenchmark()

    print("Starting Claude CLI startup benchmarks...")
    print("Test command: claude -v (version check)")
    print(f"Runs per benchmark: {BENCHMARK_RUNS}")
    print(f"Warmup runs: {WARMUP_RUNS}")

    # Run synchronous benchmarks
    benchmark.benchmark_normal_subprocess()
    benchmark.benchmark_node_direct()
    benchmark.benchmark_preloaded_env()
    benchmark.benchmark_subprocess_popen()

    # Run async benchmark (will handle its own event loop)
    benchmark.benchmark_asyncio_subprocess()

    # Skip node snapshot for now (commented out due to complexity)
    # benchmark.benchmark_node_snapshot()

    # Print summary before repeated calls
    benchmark.print_summary()

    # Run repeated calls analysis
    benchmark.benchmark_repeated_calls()


if __name__ == "__main__":
    main()
