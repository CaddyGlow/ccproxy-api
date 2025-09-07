#!/usr/bin/env python3
"""Advanced integration test runner for CCProxy plugins.

Provides fast, reliable integration test execution with plugin filtering,
parallel execution, and performance monitoring.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


# Add the project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class IntegrationTestRunner:
    """Advanced integration test runner with performance optimization."""

    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.plugins_dir = self.project_root / "plugins"

    def discover_plugins(self) -> list[str]:
        """Discover all available plugins with integration tests."""
        plugins = []
        for plugin_dir in self.plugins_dir.iterdir():
            if plugin_dir.is_dir() and (plugin_dir / "tests" / "integration").exists():
                plugins.append(plugin_dir.name)
        return sorted(plugins)

    def discover_integration_test_files(
        self, plugins: list[str] | None = None
    ) -> list[Path]:
        """Discover integration test files for specified plugins."""
        test_files = []

        # Core integration tests
        core_integration = self.project_root / "tests" / "integration"
        if core_integration.exists():
            test_files.extend(core_integration.glob("test_*.py"))

        # Plugin integration tests
        target_plugins = plugins or self.discover_plugins()
        for plugin in target_plugins:
            plugin_integration = self.plugins_dir / plugin / "tests" / "integration"
            if plugin_integration.exists():
                test_files.extend(plugin_integration.glob("test_*.py"))

        return test_files

    def run_pytest_with_args(self, args: list[str]) -> subprocess.CompletedProcess:
        """Run pytest with specified arguments."""
        cmd = ["uv", "run", "pytest"] + args
        return subprocess.run(
            cmd, cwd=self.project_root, capture_output=True, text=True
        )

    def run_integration_tests(
        self,
        plugins: list[str] | None = None,
        parallel: bool = True,
        verbose: bool = True,
        fast_fail: bool = False,
        coverage: bool = False,
    ) -> bool:
        """Run integration tests with specified options."""

        test_files = self.discover_integration_test_files(plugins)
        if not test_files:
            print("No integration test files found.")
            return True

        # Build pytest arguments
        pytest_args = ["-v" if verbose else "-q"]

        # Add markers
        if plugins:
            plugin_markers = " or ".join(plugins)
            pytest_args.extend(["-m", f"integration and ({plugin_markers})"])
        else:
            pytest_args.extend(["-m", "integration"])

        # Parallel execution
        if parallel and len(test_files) > 1:
            pytest_args.extend(["-n", "auto"])

        # Fast fail
        if fast_fail:
            pytest_args.append("-x")

        # Coverage
        if coverage:
            pytest_args.extend(["--cov=ccproxy", "--cov-report=term"])

        # Add test file paths
        pytest_args.extend([str(f) for f in test_files])

        print(f"Running integration tests for: {plugins or 'all plugins'}")
        print(f"Test files: {len(test_files)}")
        print(f"Command: uv run pytest {' '.join(pytest_args)}")

        start_time = time.time()
        result = self.run_pytest_with_args(pytest_args)
        duration = time.time() - start_time

        print(f"\nTest execution completed in {duration:.2f}s")

        if result.stdout:
            print("STDOUT:")
            print(result.stdout)

        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        return result.returncode == 0

    def run_smoke_tests(self) -> bool:
        """Run quick smoke tests to verify basic functionality."""
        print("Running integration smoke tests...")

        # Run a minimal set of tests to verify the system is working
        pytest_args = [
            "-v",
            "-m",
            "integration",
            "--tb=short",
            "-x",  # Stop on first failure
            "plugins/metrics/tests/integration/",  # Focus on metrics as it's simplest
        ]

        start_time = time.time()
        result = self.run_pytest_with_args(pytest_args)
        duration = time.time() - start_time

        print(f"Smoke tests completed in {duration:.2f}s")

        if result.returncode == 0:
            print("✅ Smoke tests passed - integration test system is working")
        else:
            print("❌ Smoke tests failed")
            if result.stdout:
                print("STDOUT:")
                print(result.stdout)
            if result.stderr:
                print("STDERR:")
                print(result.stderr)

        return result.returncode == 0

    def benchmark_startup_time(self, plugins: list[str] | None = None) -> dict:
        """Benchmark the startup time for integration tests."""
        print("Benchmarking integration test startup time...")

        # Run with timing
        pytest_args = [
            "-v",
            "--tb=no",
            "--collect-only",  # Just collect, don't run
        ]

        if plugins:
            plugin_markers = " or ".join(plugins)
            pytest_args.extend(["-m", f"integration and ({plugin_markers})"])
        else:
            pytest_args.extend(["-m", "integration"])

        test_files = self.discover_integration_test_files(plugins)
        pytest_args.extend([str(f) for f in test_files])

        start_time = time.time()
        result = self.run_pytest_with_args(pytest_args)
        collection_time = time.time() - start_time

        # Count collected tests
        collected_tests = 0
        if "collected" in result.stdout:
            for line in result.stdout.split("\n"):
                if "collected" in line:
                    try:
                        collected_tests = int(line.split()[0])
                        break
                    except (ValueError, IndexError):
                        pass

        return {
            "collection_time": collection_time,
            "collected_tests": collected_tests,
            "test_files": len(test_files),
            "plugins": plugins or self.discover_plugins(),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Advanced CCProxy integration test runner"
    )
    parser.add_argument(
        "--plugins",
        nargs="*",
        help="Specific plugins to test (e.g., metrics claude_api)",
    )
    parser.add_argument(
        "--no-parallel", action="store_true", help="Disable parallel test execution"
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    parser.add_argument(
        "--fast-fail", action="store_true", help="Stop on first test failure"
    )
    parser.add_argument(
        "--coverage", action="store_true", help="Enable test coverage reporting"
    )
    parser.add_argument(
        "--smoke", action="store_true", help="Run quick smoke tests only"
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Benchmark test startup performance"
    )
    parser.add_argument(
        "--list-plugins",
        action="store_true",
        help="List available plugins with integration tests",
    )

    args = parser.parse_args()

    runner = IntegrationTestRunner()

    if args.list_plugins:
        plugins = runner.discover_plugins()
        print("Available plugins with integration tests:")
        for plugin in plugins:
            print(f"  - {plugin}")
        return

    if args.benchmark:
        stats = runner.benchmark_startup_time(args.plugins)
        print(f"Collection time: {stats['collection_time']:.2f}s")
        print(f"Collected tests: {stats['collected_tests']}")
        print(f"Test files: {stats['test_files']}")
        print(f"Plugins: {', '.join(stats['plugins'])}")
        return

    if args.smoke:
        success = runner.run_smoke_tests()
        sys.exit(0 if success else 1)

    # Validate plugin names
    if args.plugins:
        available_plugins = runner.discover_plugins()
        invalid_plugins = [p for p in args.plugins if p not in available_plugins]
        if invalid_plugins:
            print(f"Error: Unknown plugins: {', '.join(invalid_plugins)}")
            print(f"Available plugins: {', '.join(available_plugins)}")
            sys.exit(1)

    success = runner.run_integration_tests(
        plugins=args.plugins,
        parallel=not args.no_parallel,
        verbose=not args.quiet,
        fast_fail=args.fast_fail,
        coverage=args.coverage,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
