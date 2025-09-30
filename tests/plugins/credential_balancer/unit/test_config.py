"""Tests covering credential balancer configuration helpers."""

from __future__ import annotations

from pathlib import Path

from ccproxy.plugins.credential_balancer.config import CredentialFile


def test_credential_file_expands_user_directory(monkeypatch, tmp_path: Path) -> None:
    fake_home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(fake_home))

    cred = CredentialFile(path="~/tokens/sample.json")

    assert cred.path == fake_home / "tokens" / "sample.json"


def test_credential_file_expands_environment_variables(
    monkeypatch, tmp_path: Path
) -> None:
    credentials_dir = tmp_path / "configured"
    monkeypatch.setenv("CREDENTIALS_DIR", str(credentials_dir))

    cred = CredentialFile(path="${CREDENTIALS_DIR}/token.json")

    assert cred.path == credentials_dir / "token.json"
