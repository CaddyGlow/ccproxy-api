"""Command-line entry point for the endpoint test runner."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

import structlog

from .config import list_available_tests
from .console import colored_header, colored_info
from .runner import run_endpoint_tests_async
from .tools import create_openai_tools


def setup_logging(level: str = "warn") -> None:
    """Setup structured logging with specified level."""
    log_level_map = {
        "warn": logging.WARNING,
        "info": logging.INFO,
        "debug": logging.DEBUG,
        "error": logging.ERROR,
    }

    logging.basicConfig(
        level=log_level_map.get(level, logging.WARNING),
        format="%(message)s",
    )

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.CallsiteParameterAdder(
                parameters=[
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.LINENO,
                ]
            ),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def main(argv: list[str] | None = None) -> None:
    """Run the CLI test harness."""
    parser = argparse.ArgumentParser(
        description=(
            "Test CCProxy endpoints with response validation, function tools, "
            "thinking mode, and structured output support"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
{list_available_tests()}

Test selection examples:
  --tests 1                           # Run test 1 only
  --tests 1,3,5                       # Run tests 1, 3, and 5
  --tests 1..3                        # Run tests 1 through 3
  --tests 4..                         # Run tests 4 through end
  --tests ..3                         # Run tests 1 through 3
  --tests 1,4..6,8                    # Run test 1, tests 4-6, and test 8
  --tests copilot_chat_completions    # Run test by exact name
  --tests copilot                     # Run all tests containing "copilot"
  --tests "copilot_.*_stream"         # Run all copilot streaming tests (regex)
  --tests ".*_stream"                 # Run all streaming tests (regex)
  --tests "claude_.*"                 # Run all claude tests (regex)
  --tests 1,copilot_.*_stream,codex   # Mix indices, regex, and partial names

Feature-specific test patterns:
  --tests ".*_tools.*"                # Run all function tool tests
  --tests ".*_thinking.*"             # Run all thinking mode tests
  --tests ".*_structured.*"           # Run all structured output tests
  --tools                             # Add function tools to compatible tests
  --thinking                          # Use thinking-capable models where available
  --structured                        # Enable structured output formatting
""",
    )
    parser.add_argument(
        "--base",
        default="http://127.0.0.1:8000",
        help="Base URL for the API server (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--tests",
        help=(
            "Select tests by index, name, regex pattern, or ranges (e.g., "
            "1,2,3 or copilot_.*_stream or 1..3)"
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available tests and exit (don't run any tests)",
    )
    parser.add_argument(
        "--tools",
        action="store_true",
        help="Add function tools to compatible test requests (weather, distance, calculate)",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable thinking mode for OpenAI requests (uses o1-preview/o1-mini models)",
    )
    parser.add_argument(
        "--structured",
        action="store_true",
        help="Enable structured output mode for detailed response analysis",
    )
    parser.add_argument(
        "--show-tools",
        action="store_true",
        help="Display available function tools and exit",
    )
    parser.add_argument(
        "-v",
        action="store_true",
        help="Set log level to INFO",
    )
    parser.add_argument(
        "-vv",
        action="store_true",
        help="Set log level to DEBUG",
    )
    parser.add_argument(
        "-vvv",
        action="store_true",
        help="Set log level to DEBUG (same as -vv)",
    )
    parser.add_argument(
        "--log-level",
        choices=["warn", "info", "debug", "error"],
        default="warn",
        help="Set log level explicitly (default: warn)",
    )

    args = parser.parse_args(argv)

    log_level = args.log_level
    if args.v:
        log_level = "info"
    elif args.vv or args.vvv:
        log_level = "debug"

    setup_logging(log_level)

    if args.show_tools:
        print(colored_header("Available Function Tools"))
        for tool in create_openai_tools():
            func = tool["function"]
            print(f"\n{colored_info('Tool:')} {func['name']}")
            print(f"{colored_info('Description:')} {func['description']}")
            print(
                f"{colored_info('Parameters:')} {json.dumps(func['parameters'], indent=2)}"
            )
        sys.exit(0)

    if args.list:
        print(list_available_tests())
        if args.tools or args.thinking or args.structured:
            print(colored_header("Available Feature Flags"))
            if args.tools:
                print(colored_info("Function tools will be added to compatible tests"))
            if args.thinking:
                print(colored_info("Thinking mode will be enabled for OpenAI tests"))
            if args.structured:
                print(colored_info("Structured output mode will be enabled"))
        sys.exit(0)

    if args.tools or args.thinking or args.structured:
        print(colored_header("Global Feature Flags Applied"))
        if args.tools:
            print(colored_info("→ Function tools enabled for compatible tests"))
        if args.thinking:
            print(colored_info("→ Thinking mode enabled for OpenAI tests"))
        if args.structured:
            print(colored_info("→ Structured output mode enabled"))

    try:
        summary = asyncio.run(
            run_endpoint_tests_async(base_url=args.base, tests=args.tests)
        )
    except ValueError as exc:
        structlog.get_logger(__name__).error(
            "Invalid test selection format", selection=args.tests, error=str(exc)
        )
        sys.exit(1)
    except KeyboardInterrupt:
        structlog.get_logger(__name__).info("Tests interrupted by user")
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        structlog.get_logger(__name__).error(
            "Test execution failed", error=str(exc), exc_info=exc
        )
        sys.exit(1)

    if summary.failure_count:
        sys.exit(1)


__all__ = ["main", "setup_logging"]
