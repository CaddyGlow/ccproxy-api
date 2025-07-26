#!/usr/bin/env python3
"""
Test script to verify logging configuration works correctly.
"""

import logging
import tempfile
from pathlib import Path

from ccproxy.core.logging import get_logger, setup_logging


def test_logging_levels() -> None:
    """Test that logging levels work correctly for different log levels."""
    print("=== Testing Logging Configuration ===\n")

    # Test 1: DEBUG level with file logging
    print("1. Testing DEBUG level with file logging...")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
        log_file = f.name

    # Setup logging in DEBUG mode
    setup_logging(json_logs=False, log_level_name="DEBUG", log_file=log_file)

    # Get different types of loggers
    structlog_logger = get_logger("ccproxy.test")
    stdlib_logger = logging.getLogger("ccproxy.api.test")
    uvicorn_logger = logging.getLogger("uvicorn.error")

    # Test log messages at different levels
    print("   Sending test messages...")
    structlog_logger.debug("Structlog DEBUG message")
    structlog_logger.info("Structlog INFO message")
    structlog_logger.warning("Structlog WARNING message")

    stdlib_logger.debug("Stdlib DEBUG message")
    stdlib_logger.info("Stdlib INFO message")
    stdlib_logger.warning("Stdlib WARNING message")

    uvicorn_logger.debug("Uvicorn DEBUG message")
    uvicorn_logger.info("Uvicorn INFO message")
    uvicorn_logger.warning("Uvicorn WARNING message")

    # Check file contents
    log_path = Path(log_file)
    if log_path.exists():
        content = log_path.read_text()
        lines = [line for line in content.strip().split("\n") if line]

        print(f"   File has {len(lines)} log entries")

        # Count debug messages
        debug_count = len([line for line in lines if '"level": "debug"' in line])
        info_count = len([line for line in lines if '"level": "info"' in line])
        warning_count = len([line for line in lines if '"level": "warning"' in line])

        print(f"   DEBUG messages: {debug_count}")
        print(f"   INFO messages: {info_count}")
        print(f"   WARNING messages: {warning_count}")

        # Look for specific logger messages
        ccproxy_debug = any("ccproxy" in line and "debug" in line for line in lines)
        uvicorn_debug = any("uvicorn" in line and "debug" in line for line in lines)

        print(f"   ccproxy DEBUG found: {ccproxy_debug}")
        print(f"   uvicorn DEBUG found: {uvicorn_debug}")

        # Show first few log entries
        print("\n   First 3 log entries:")
        for i, line in enumerate(lines[:3]):
            print(f"   {i + 1}: {line}")

    else:
        print("   ERROR: Log file was not created!")

    # Cleanup
    log_path.unlink(missing_ok=True)
    print()


def test_console_timestamps() -> None:
    """Test console timestamp formatting."""
    print("2. Testing console timestamp formatting...")

    # Test DEBUG mode (should use HH:MM:SS)
    setup_logging(json_logs=False, log_level_name="DEBUG")
    logger = get_logger("test.debug")
    print("   DEBUG mode - should show HH:MM:SS timestamps in console above")
    logger.info("Test message in DEBUG mode")

    print()

    # Test INFO mode (should use full timestamps)
    setup_logging(json_logs=False, log_level_name="INFO")
    logger = get_logger("test.info")
    print("   INFO mode - should show full timestamps in console above")
    logger.info("Test message in INFO mode")

    print()


def test_logger_hierarchy() -> None:
    """Test that logger hierarchy works correctly."""
    print("3. Testing logger hierarchy...")

    setup_logging(json_logs=False, log_level_name="DEBUG")

    # Test parent/child logger relationship
    parent_logger = logging.getLogger("ccproxy")
    child_logger = logging.getLogger("ccproxy.api.test")
    grandchild_logger = logging.getLogger("ccproxy.api.test.submodule")

    print(f"   Parent logger level: {parent_logger.level} (should be 10 for DEBUG)")
    print(f"   Child logger level: {child_logger.level}")
    print(f"   Grandchild logger level: {grandchild_logger.level}")

    print(f"   Parent propagate: {parent_logger.propagate}")
    print(f"   Child propagate: {child_logger.propagate}")
    print(f"   Grandchild propagate: {grandchild_logger.propagate}")

    # Test that messages flow up the hierarchy
    child_logger.debug("Child debug message - should appear")
    grandchild_logger.debug("Grandchild debug message - should appear")

    print()


if __name__ == "__main__":
    test_logging_levels()
    test_console_timestamps()
    test_logger_hierarchy()
    print("=== Logging tests complete ===")
