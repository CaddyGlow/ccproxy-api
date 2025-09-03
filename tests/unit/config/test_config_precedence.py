import pytest

from ccproxy.config.settings import Settings


@pytest.mark.unit
def test_env_overrides_toml(tmp_path, monkeypatch):
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        """
    [server]
    port = 8001
    host = "127.0.0.1"
    """,
        encoding="utf-8",
    )

    monkeypatch.setenv("SERVER__PORT", "9001")

    settings = Settings.from_config(config_path=cfg)
    assert settings.server.port == 9001  # env > toml
    assert settings.server.host == "127.0.0.1"


@pytest.mark.unit
def test_cli_overrides_env(tmp_path, monkeypatch):
    # env sets INFO, CLI sets DEBUG
    monkeypatch.setenv("LOGGING__LEVEL", "INFO")

    settings = Settings.from_config(config_path=None, logging={"level": "DEBUG"})
    assert settings.logging.level == "DEBUG"  # cli > env


@pytest.mark.unit
def test_cli_overrides_toml(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        """
    [server]
    port = 8001
    """,
        encoding="utf-8",
    )

    settings = Settings.from_config(config_path=cfg, server={"port": 9002})
    assert settings.server.port == 9002  # cli > toml
