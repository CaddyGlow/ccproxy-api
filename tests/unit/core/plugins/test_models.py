"""Tests for provider health models."""

from ccproxy.core.plugins import models


def test_provider_health_details_serialization():
    details = models.ProviderHealthDetails(
        provider="demo",
        enabled=True,
        base_url="https://api.demo",
        cli=models.CLIHealth(available=True, status="ok", version="1.0", path="/bin"),
        auth=models.AuthHealth(configured=True, token_available=True),
        config=models.ConfigHealth(model_count=2, supports_openai_format=True),
    )

    data = details.model_dump()

    assert data["provider"] == "demo"
    assert data["cli"]["status"] == "ok"
    assert data["auth"]["configured"] is True
    assert data["config"]["supports_openai_format"] is True
