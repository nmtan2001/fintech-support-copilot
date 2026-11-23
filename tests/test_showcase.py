"""Showcase startup must fail clearly and clean up only its own processes."""

import pytest
import hashlib
from pathlib import Path
from types import SimpleNamespace


def test_model_preload_requires_pinned_weights(tmp_path):
    from scripts.showcase import verified_model

    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"test-weights")
    manifest = {"weights_sha256": hashlib.sha256(b"test-weights").hexdigest()}
    assert verified_model(Path(tmp_path), manifest)
    (tmp_path / "model.safetensors").write_bytes(b"changed")
    assert not verified_model(Path(tmp_path), manifest)


def test_missing_docker_has_actionable_error(monkeypatch):
    from scripts import showcase

    monkeypatch.setattr(showcase.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="Docker Desktop.*WSL"):
        showcase.preflight()


def test_cleanup_stops_all_children_even_after_one_exits():
    from scripts import showcase

    class Process:
        def __init__(self, exited=False):
            self.exited = exited
            self.stopped = False

        def poll(self):
            return 0 if self.exited else None

        def terminate(self):
            self.stopped = True

        def wait(self, timeout):
            return 0

    exited, active = Process(True), Process()
    showcase.stop_children([exited, active])
    assert active.stopped
    assert not exited.stopped


def test_startup_failure_scales_down_showcase_and_preserves_error(monkeypatch, tmp_path):
    from scripts import showcase

    config = tmp_path / "kubeconfig"
    config.write_text("test")
    monkeypatch.setattr(showcase, "KUBECONFIG", config)
    monkeypatch.setattr(showcase, "preflight", lambda: None)

    def fail():
        raise RuntimeError("rollout failed")

    monkeypatch.setattr(showcase, "prepare_cluster", fail)
    commands = []

    def kubectl(*args, **kwargs):
        commands.append(args)
        return SimpleNamespace(stdout=showcase.NAME)

    monkeypatch.setattr(showcase, "kubectl", kubectl)
    with pytest.raises(RuntimeError, match="rollout failed"):
        showcase.run()
    assert any(command[-1] == "--replicas=0" for command in commands)


def test_credential_update_preserves_existing_hf_token(monkeypatch, capsys):
    import json
    from scripts import showcase

    monkeypatch.setattr(showcase, "get_secret", lambda name, **kwargs: "new-zai-key" if name == "ZAI_API_KEY" else None)
    applied = []

    def kubectl(*args, **kwargs):
        if "get" in args:
            return SimpleNamespace(stdout=json.dumps({"data": {"HF_TOKEN": "existing-encoded-token"}}))
        applied.append(json.loads(kwargs["data"]))
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(showcase, "kubectl", kubectl)
    showcase.configure_credentials()
    assert applied[0]["data"]["HF_TOKEN"] == "existing-encoded-token"
    assert applied[0]["stringData"]["ZAI_API_KEY"] == "new-zai-key"
    output = capsys.readouterr()
    assert "new-zai-key" not in output.out + output.err


def test_startup_failure_stops_redis_even_when_api_was_not_created(monkeypatch, tmp_path):
    from scripts import showcase

    config = tmp_path / "kubeconfig"
    config.write_text("test")
    monkeypatch.setattr(showcase, "KUBECONFIG", config)
    monkeypatch.setattr(showcase, "preflight", lambda: None)

    def fail():
        raise RuntimeError("Redis readiness failed")

    monkeypatch.setattr(showcase, "prepare_cluster", fail)
    commands = []

    def kubectl(*args, **kwargs):
        commands.append(args)
        return SimpleNamespace(stdout="copilot-sessions" if "copilot-sessions" in args else "")

    monkeypatch.setattr(showcase, "kubectl", kubectl)
    with pytest.raises(RuntimeError, match="Redis readiness failed"):
        showcase.run()
    assert ("-n", showcase.NAMESPACE, "scale", "deployment/copilot-sessions", "--replicas=0") in commands
