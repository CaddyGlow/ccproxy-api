"""Integration test configuration and fixtures.

This module provides integration-specific pytest configuration and imports
the shared integration fixtures for all plugin integration tests.
"""

import pytest


def pytest_configure(config):
    """Configure pytest for integration tests."""
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "metrics: mark test as metrics plugin test")
    config.addinivalue_line(
        "markers", "claude_api: mark test as claude_api plugin test"
    )
    config.addinivalue_line(
        "markers", "claude_sdk: mark test as claude_sdk plugin test"
    )
    config.addinivalue_line("markers", "codex: mark test as codex plugin test")
    config.addinivalue_line(
        "markers", "access_log: mark test as access_log plugin test"
    )
    config.addinivalue_line(
        "markers", "permissions: mark test as permissions plugin test"
    )
    config.addinivalue_line("markers", "pricing: mark test as pricing plugin test")
    config.addinivalue_line(
        "markers", "request_tracer: mark test as request_tracer plugin test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test items for better integration test handling."""
    for item in items:
        # Auto-mark tests in integration directories
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Auto-mark plugin-specific tests based on path
        item_path = str(item.fspath)
        if "plugins/metrics" in item_path:
            item.add_marker(pytest.mark.metrics)
        elif "plugins/claude_api" in item_path:
            item.add_marker(pytest.mark.claude_api)
        elif "plugins/claude_sdk" in item_path:
            item.add_marker(pytest.mark.claude_sdk)
        elif "plugins/codex" in item_path:
            item.add_marker(pytest.mark.codex)
        elif "plugins/access_log" in item_path:
            item.add_marker(pytest.mark.access_log)
        elif "plugins/permissions" in item_path:
            item.add_marker(pytest.mark.permissions)
        elif "plugins/pricing" in item_path:
            item.add_marker(pytest.mark.pricing)
        elif "plugins/request_tracer" in item_path:
            item.add_marker(pytest.mark.request_tracer)
