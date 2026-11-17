"""Exercise release recovery without mutating a cluster."""
import json

import pytest

from scripts import deploy


@pytest.mark.parametrize("failure", ["apply", "rollout", "verify"])
def test_failed_release_restores_previous_resources(monkeypatch, failure):
    old = {kind: {"kind": kind, "metadata": {"name": deploy.NAME}} for kind in
           ("configmap", "deployment", "service")}
    monkeypatch.setattr(deploy, "snapshot", lambda kind: old[kind])
    applied = []
    failed = False

    def kubectl(*args, data=None):
        nonlocal failed
        if args[0] == "create":
            return json.dumps({"metadata": {}, "spec": {"template": {"spec": {"containers": [{}]}}}})
        if args[:2] == ("apply", "-f"):
            if data:
                applied.append(json.loads(data))
            if failure == "apply" and not failed:
                failed = True
                raise RuntimeError("apply failed")
        if args[0] == "rollout" and failure == "rollout" and not failed:
            failed = True
            raise RuntimeError("rollout failed")
        return ""

    monkeypatch.setattr(deploy, "kubectl", kubectl)

    def verify(*args):
        if failure == "verify":
            raise RuntimeError("verify failed")
    monkeypatch.setattr(deploy, "check_service", verify)
    with pytest.raises(RuntimeError):
        deploy.deploy("registry/image@sha256:" + "a" * 64, "1.1.0", "commit")
    for resource in old.values():
        assert resource in applied


def test_mutable_image_is_rejected_before_cluster_changes(monkeypatch):
    def unexpected(*args):
        pytest.fail("Cluster was accessed for a mutable image")
    monkeypatch.setattr(deploy, "snapshot", unexpected)
    with pytest.raises(ValueError, match="digest"):
        deploy.deploy("registry/image:latest", "1.0.0", "commit")
