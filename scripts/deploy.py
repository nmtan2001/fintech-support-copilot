"""Apply an immutable release; restore previous resources if verification fails."""
import argparse
import json
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
    process = subprocess.Popen(
        ["kubectl", "-n", NAMESPACE, "port-forward", "service/" + NAME, "18000:80"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        # Retry only transport/HTTP errors; wrong identity/content fails immediately.
        import time
        from urllib.error import URLError
        for attempt in range(30):
            try:
                verify("http://127.0.0.1:18000", version, commit)
                return
            except (URLError, TimeoutError):
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
    if "@sha256:" not in image:
        raise ValueError("Deployment requires an image digest")
    previous = {kind: snapshot(kind) for kind in ("configmap", "deployment", "service")}
    deployment = json.loads(kubectl("create", "--dry-run=client", "-f", "k8s/deployment.yaml", "-o", "json"))
    deployment["spec"]["template"]["spec"]["containers"][0]["image"] = image
    deployment["metadata"].setdefault("annotations", {}).update({
        "copilot/version": version, "copilot/commit": commit})
    # Snapshot before mutation. Rollback restores config as well as the pod template.
    try:
        kubectl("apply", "-f", "k8s/configmap.yaml")
        kubectl("apply", "-f", "-", data=json.dumps(deployment))
        kubectl("apply", "-f", "k8s/service.yaml")
        kubectl("rollout", "status", "deployment/" + NAME, "--timeout=12m")
        check_service(version, commit)
    except Exception:
        if previous["deployment"] is not None:
            for document in previous.values():
                if document is not None:
                    kubectl("apply", "-f", "-", data=json.dumps(document))
            kubectl("rollout", "status", "deployment/" + NAME, "--timeout=12m")
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
