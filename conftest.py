"""Top-level pytest configuration for plugin fixture registration.

This file centralizes `pytest_plugins` to comply with pytest's requirement
that plugin declarations live in a top-level conftest located at the rootdir.
"""

# Register shared test fixture modules used across the suite
pytest_plugins = [
    "tests.fixtures.claude_sdk.internal_mocks",
    "tests.fixtures.claude_sdk.client_mocks",
    "tests.fixtures.external_apis.anthropic_api",
    "tests.fixtures.external_apis.copilot_api",
    "tests.fixtures.external_apis.codex_api",
    "tests.fixtures.external_apis.openai_codex_api",
    # Integration-wide fixtures
    "tests.fixtures.integration",
]
