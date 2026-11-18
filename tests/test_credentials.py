"""Ensure renewal never uploads admin credentials or tokens without opt-in."""
import base64
import json
import sys

from scripts import renew_cluster_credentials as renewal


def test_creates_private_scoped_file_without_implicit_upload(monkeypatch, tmp_path, capsys):
    payload = base64.urlsafe_b64encode(json.dumps({"exp": 1800000000}).encode()).decode().rstrip("=")
    token = "header." + payload + ".signature"
    config = {"clusters": [{"name": "test", "cluster": {"server": "https://example.invalid"}}],
              "users": [{"user": {"client-key-data": "ADMIN_PRIVATE_KEY"}}]}
    def output(command, **kwargs):
        return json.dumps(config) if command[1] == "config" else token
    monkeypatch.setattr(renewal.subprocess, "check_output", output)
    def unexpected(*args, **kwargs):
        raise AssertionError("No upload was requested")
    monkeypatch.setattr(renewal.subprocess, "run", unexpected)
    target = tmp_path / "deployer.json"
    monkeypatch.setattr(sys, "argv", ["renew", "--admin-kubeconfig", str(tmp_path / "admin"), "--output", str(target)])
    renewal.main()
    assert target.stat().st_mode & 0o777 == 0o600
    body = json.loads(target.read_text())
    assert body["users"][0]["name"] == "copilot-deployer"
    assert body["contexts"][0]["context"]["namespace"] == "fintech-support"
    assert "ADMIN_PRIVATE_KEY" not in target.read_text()
    assert token not in capsys.readouterr().out


def test_refuses_filesystem_that_cannot_enforce_private_permissions(monkeypatch, tmp_path):
    import pytest
    payload = base64.urlsafe_b64encode(json.dumps({"exp": 1800000000}).encode()).decode().rstrip("=")
    token = "header." + payload + ".signature"
    config = {"clusters": [{"name": "test", "cluster": {"server": "https://example.invalid"}}]}
    monkeypatch.setattr(renewal.subprocess, "check_output",
                        lambda command, **kwargs: json.dumps(config) if command[1] == "config" else token)
    target = tmp_path / "insecure.json"
    target.write_text("placeholder")
    target.chmod(0o644)
    monkeypatch.setattr(renewal.os, "fchmod", lambda *args: None)
    monkeypatch.setattr(sys, "argv", ["renew", "--admin-kubeconfig", str(tmp_path / "admin"), "--output", str(target)])
    with pytest.raises(PermissionError, match="private"):
        renewal.main()
    assert not target.exists()
