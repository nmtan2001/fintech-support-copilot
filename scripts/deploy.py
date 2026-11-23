"""Apply an immutable release; restore previous resources if verification fails."""
import argparse
import hashlib
import json
import re
import socket
import subprocess
from pathlib import Path

from scripts.verify_release import verify

NAMESPACE = "fintech-support"
NAME = "fintech-support-copilot"


def kubectl(*args, data=None):
    result = subprocess.run(["kubectl", "-n", NAMESPACE, *args], input=data,
                            text=True, capture_output=True, check=True)
    return result.stdout


def snapshot(kind):
    value = kubectl("get", kind, NAME, "--ignore-not-found", "-o", "json")
    if not value.strip():
        return None
    document = json.loads(value)
    document.pop("status", None)
    for key in ("resourceVersion", "uid", "creationTimestamp", "generation", "managedFields"):
        document["metadata"].pop(key, None)
    document["metadata"].get("annotations", {}).pop("kubectl.kubernetes.io/last-applied-configuration", None)
    return document


def check_service(version, commit):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    process = subprocess.Popen(
        ["kubectl", "-n", NAMESPACE, "port-forward", "service/" + NAME, f"{port}:80"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        # Retry only transport/HTTP errors; wrong identity/content fails immediately.
        import time
        from urllib.error import URLError
        for attempt in range(30):
            try:
                verify(f"http://127.0.0.1:{port}", version, commit)
                return
            except (URLError, TimeoutError, ConnectionError):
                if process.poll() is not None or attempt == 29:
                    raise
                time.sleep(2)
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def deploy(image, version, commit):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9./:_-]*@sha256:[a-f0-9]{64}", image):
        raise ValueError("Deployment requires an image digest")
    previous = {kind: snapshot(kind) for kind in ("configmap", "deployment", "service")}
    deployment = json.loads(kubectl("create", "--dry-run=client", "-f", "k8s/deployment.yaml", "-o", "json"))
    deployment["spec"]["template"]["spec"]["containers"][0]["image"] = image
    deployment["spec"]["template"].setdefault("metadata", {}).setdefault("annotations", {})["checksum/config"] = (
        hashlib.sha256(Path("k8s/configmap.yaml").read_bytes()).hexdigest()
    )
    deployment["metadata"].setdefault("annotations", {}).update({
        "copilot/version": version, "copilot/commit": commit})
    # Snapshot before mutation. Rollback restores config as well as the pod template.
    try:
        kubectl("apply", "-f", "k8s/sessions.yaml")
        kubectl("rollout", "status", "deployment/copilot-sessions", "--timeout=3m")
        kubectl("apply", "-f", "k8s/configmap.yaml")
        kubectl("apply", "-f", "-", data=json.dumps(deployment))
        kubectl("apply", "-f", "k8s/service.yaml")
        kubectl("rollout", "status", "deployment/" + NAME, "--timeout=30m")
        check_service(version, commit)
    except Exception:
        for args in (("get", "pods", "-o", "wide"),
                     ("get", "events", "--sort-by=.lastTimestamp"),
                     ("logs", "-l", "app.kubernetes.io/name=" + NAME, "--tail=100", "--prefix")):
            try:
                print(kubectl(*args), flush=True)
            except subprocess.CalledProcessError:
                pass
        if previous["deployment"] is not None:
            for document in previous.values():
                if document is not None:
                    kubectl("apply", "-f", "-", data=json.dumps(document))
            kubectl("rollout", "status", "deployment/" + NAME, "--timeout=30m")
            identity = previous["deployment"]["metadata"].get("annotations", {})
            if "copilot/version" in identity and "copilot/commit" in identity:
                check_service(identity["copilot/version"], identity["copilot/commit"])
            print("Previous deployment restored", flush=True)
        else:
            print("Initial deployment failed; no previous release exists", flush=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    deploy(args.image, args.version, args.commit)
    Path("release-evidence.json").write_text(json.dumps(vars(args), indent=2) + "\n")


if __name__ == "__main__":
    main()
