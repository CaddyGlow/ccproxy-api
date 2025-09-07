import pytest


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.integration
@pytest.mark.metrics
async def test_metrics_endpoint_available_when_enabled(metrics_integration_client):
    """Test that metrics endpoint is available when plugin is enabled."""
    resp = await metrics_integration_client.get("/metrics")
    assert resp.status_code == 200
    # Prometheus exposition format usually starts with HELP/TYPE lines
    assert b"# HELP" in resp.content or b"# TYPE" in resp.content


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.integration
@pytest.mark.metrics
async def test_metrics_endpoint_absent_when_plugins_disabled(disabled_plugins_client):
    """Test that metrics endpoint is not available when plugins are disabled."""
    resp = await disabled_plugins_client.get("/metrics")
    # With plugins disabled, core does not mount /metrics
    assert resp.status_code == 404


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.integration
@pytest.mark.metrics
async def test_metrics_content_format(metrics_integration_client):
    """Test that metrics endpoint returns proper Prometheus format."""
    resp = await metrics_integration_client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "text/plain; version=0.0.4; charset=utf-8"

    content = resp.content.decode()
    # Should contain at least some basic metrics
    assert len(content.strip()) > 0
