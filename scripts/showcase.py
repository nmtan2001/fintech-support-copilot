"""Run the free Kubernetes showcase until Ctrl+C; retain the cluster for reuse."""

import argparse
import importlib.util
import json
import hashlib
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import URLError
from urllib.request import urlopen

from scripts.verify_release import verify
from scripts.verify_conversation import verify_conversation
from fintech_support.settings import get_secret

ROOT = Path(__file__).resolve().parents[1]
CLUSTER = "fintech-showcase"
NAMESPACE = "fintech-support"
NAME = "fintech-support-copilot"
CONTEXT = "kind-" + CLUSTER
KUBECONFIG = ROOT / ".runtime/showcase/kubeconfig"


def tool(name):
    installed = shutil.which(name)
    bundled = ROOT / ".runtime/tools" / name
    if installed:
        return installed
    if bundled.is_file() and os.access(bundled, os.X_OK):
        return str(bundled)
    raise RuntimeError(f"Missing {name}. Install it before starting the showcase.")


def preflight():
    if not shutil.which("docker"):
        raise RuntimeError("Docker is unavailable. Start Docker Desktop and enable integration for this WSL distro.")
    for name in ("kind", "kubectl", "cloudflared"):
        tool(name)
    if importlib.util.find_spec("streamlit") is None:
        raise RuntimeError("Streamlit is missing. Use the project's demo Python environment.")
    try:
        subprocess.run(["docker", "info"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Docker is unavailable. Start Docker Desktop and enable integration for this WSL distro.") from exc
    for port in (8000, 8501):
        try:
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", port))
        except OSError as exc:
            if port == 8000 and showcase_port_mapped():
                continue
            raise RuntimeError(f"Port {port} is occupied. Stop the existing service before starting.") from exc


def showcase_port_mapped():
    result = subprocess.run(["docker", "inspect", "--format", "{{json .HostConfig.PortBindings}}",
                             CLUSTER + "-control-plane"], text=True, capture_output=True)
    if result.returncode != 0:
        return False
    bindings = json.loads(result.stdout) or {}
    return {"HostIp": "127.0.0.1", "HostPort": "8000"} in (bindings.get("30080/tcp") or [])


def kubectl(*args, data=None, capture=False):
    return subprocess.run([tool("kubectl"), "--kubeconfig", str(KUBECONFIG), "--context", CONTEXT, *args],
                          input=data, text=True, capture_output=capture, check=True)


def stop_children(processes):
    for process in reversed(processes):
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def wait_http(url, processes, *, seconds=120):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if any(process.poll() is not None for process in processes):
            raise RuntimeError("A showcase process exited during startup. Check its output.")
        try:
            with urlopen(url, timeout=3) as response:
                if response.status == 200:
                    return
        except (URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(1)
    raise RuntimeError(f"Service did not become ready within {seconds} seconds: {url}")


def configure_credentials():
    existing = kubectl("-n", NAMESPACE, "get", "secret", NAME, "--ignore-not-found", "-o", "json", capture=True).stdout
    data = json.loads(existing).get("data", {}) if existing.strip() else {}
    token = get_secret("HF_TOKEN", root=ROOT, aliases=("HUGGINGFACE_HUB_TOKEN",))
    zai_key = get_secret("ZAI_API_KEY", root=ROOT)
    values = {key: value for key, value in {"HF_TOKEN": token, "ZAI_API_KEY": zai_key}.items() if value}
    for key in ("HF_TOKEN", "ZAI_API_KEY"):
        if key not in values and not data.get(key):
            raise RuntimeError(f"Add {key} to .streamlit/secrets.toml before starting the conversational showcase.")
    if values:
        secret = {"apiVersion": "v1", "kind": "Secret", "metadata": {
            "name": NAME, "namespace": NAMESPACE}, "type": "Opaque", "data": data, "stringData": values}
        # stdin keeps the credential out of process arguments and files.
        kubectl("apply", "-f", "-", data=json.dumps(secret), capture=True)


def verified_model(directory, manifest):
    weights = directory / "model.safetensors"
    if not weights.is_file() or not (directory / "config.json").is_file():
        return False
    digest = hashlib.sha256()
    with weights.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == manifest["weights_sha256"]


def preload_model():
    manifest = json.loads((ROOT / "model-manifest.json").read_text())
    source = ROOT / "outputs/distilbert/model"
    if not verified_model(source, manifest):
        from huggingface_hub import snapshot_download
        source = Path(snapshot_download(manifest["model_id"], revision=manifest["revision"],
                                       token=get_secret("HF_TOKEN", root=ROOT),
                                       allow_patterns=["*.json", "*.txt", "model.safetensors"]))
    if not verified_model(source, manifest):
        raise RuntimeError("The model weights do not match the pinned release checksum")
    destination = "/var/local/fintech-model/" + manifest["revision"]
    node = CLUSTER + "-control-plane"
    subprocess.run(["docker", "exec", node, "mkdir", "-p", destination], check=True)
    with tempfile.TemporaryDirectory(prefix="model-", dir=KUBECONFIG.parent) as directory:
        for name in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
                     "special_tokens_map.json", "vocab.txt"):
            if (source / name).is_file():
                shutil.copyfile(source / name, Path(directory) / name)
        subprocess.run(["docker", "cp", directory + "/.", node + ":" + destination], check=True)
    subprocess.run(["docker", "exec", node, "chmod", "-R", "a+rX", destination], check=True)
    return destination


def prepare_cluster():
    KUBECONFIG.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    clusters = subprocess.check_output([tool("kind"), "get", "clusters"], text=True).splitlines()
    if CLUSTER not in clusters:
        subprocess.run([tool("kind"), "create", "cluster", "--name", CLUSTER,
                        "--config", str(ROOT / "k8s/showcase-kind.yaml"),
                        "--kubeconfig", str(KUBECONFIG), "--wait", "120s"], check=True)
    else:
        if not showcase_port_mapped():
            raise RuntimeError("Existing fintech-showcase cluster lacks the localhost NodePort mapping. "
                               "Preserve any needed data and recreate it using k8s/showcase-kind.yaml.")
        subprocess.run([tool("kind"), "export", "kubeconfig", "--name", CLUSTER,
                        "--kubeconfig", str(KUBECONFIG)], check=True)
    kubectl("apply", "-f", "k8s/namespace.yaml")
    configure_credentials()
    model_path = preload_model()
    version = (ROOT / "VERSION").read_text().strip()
    revision = subprocess.check_output(["git", "rev-parse", "--short=12", "HEAD"], text=True).strip()
    # A distinct local tag prevents kind nodes from reusing an older dirty build.
    commit = f"{revision}-showcase-{time.time_ns()}"
    image = f"fintech-support-copilot:{commit}"
    subprocess.run(["docker", "build", "--build-arg", f"APP_VERSION={version}",
                    "--build-arg", f"APP_COMMIT={commit}", "-t", image, "."], check=True)
    subprocess.run([tool("kind"), "load", "docker-image", "--name", CLUSTER, image], check=True)
    redis_image = "redis:7.4-alpine"
    available = subprocess.run(["docker", "image", "inspect", redis_image],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if available.returncode != 0:
        subprocess.run(["docker", "pull", redis_image], check=True)
    archive = KUBECONFIG.parent / "redis-image.tar"
    subprocess.run(["docker", "image", "save", "--platform", "linux/amd64", "--output", str(archive), redis_image], check=True)
    try:
        subprocess.run([tool("kind"), "load", "image-archive", "--name", CLUSTER, str(archive)], check=True)
    finally:
        archive.unlink(missing_ok=True)
    deployment = json.loads(kubectl("create", "--dry-run=client", "-f", "k8s/deployment.yaml",
                                    "-o", "json", capture=True).stdout)
    pod = deployment["spec"]["template"]["spec"]
    pod.pop("imagePullSecrets", None)
    pod["containers"][0]["image"] = image
    pod["containers"][0]["imagePullPolicy"] = "Never"
    pod["containers"][0].setdefault("env", []).append({"name": "SHOWCASE_RUN_ID", "value": commit})
    pod["containers"][0]["env"].append({"name": "HF_MODEL_PATH", "value": "/opt/fintech-model"})
    pod["containers"][0].setdefault("volumeMounts", []).append({"name": "pinned-model", "mountPath": "/opt/fintech-model", "readOnly": True})
    pod.setdefault("volumes", []).append({"name": "pinned-model", "hostPath": {"path": model_path, "type": "Directory"}})
    kubectl("apply", "-f", "k8s/configmap.yaml")
    kubectl("apply", "-f", "k8s/sessions.yaml")
    kubectl("-n", NAMESPACE, "rollout", "status", "deployment/copilot-sessions", "--timeout=3m")
    kubectl("apply", "-f", "-", data=json.dumps(deployment))
    service = json.loads(kubectl("create", "--dry-run=client", "-f", "k8s/service.yaml", "-o", "json", capture=True).stdout)
    service["spec"]["type"] = "NodePort"
    service["spec"]["ports"][0]["nodePort"] = 30080
    kubectl("apply", "-f", "-", data=json.dumps(service))
    kubectl("-n", NAMESPACE, "rollout", "status", "deployment/" + NAME, "--timeout=30m")
    return version, commit


def run():
    preflight()
    processes = []
    prepared = False
    try:
        # Keep cloudflared output on the terminal so its temporary public URL is visible.
        prepared = True
        version, commit = prepare_cluster()
        wait_http("http://127.0.0.1:8000/ready", processes)
        verify("http://127.0.0.1:8000", version, commit)
        verify_conversation("http://127.0.0.1:8000", require_generation=True)
        print("Kubernetes API identity, inference, retrieval and synthetic lookup verified.", flush=True)
        env = dict(os.environ, SUPPORT_API_URL="http://127.0.0.1:8000")
        env.pop("NVIDIA_API_KEY", None)
        processes.append(subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py",
                                           "--server.address=127.0.0.1", "--server.port=8501",
                                           "--server.fileWatcherType=none",
                                           "--server.headless=true", "--browser.gatherUsageStats=false"], env=env))
        wait_http("http://127.0.0.1:8501/_stcore/health", processes)
        print("Opening the public UI. Share the trycloudflare.com URL below. Stop with Ctrl+C.", flush=True)
        # An empty config avoids accidentally using an existing named tunnel configuration.
        with tempfile.TemporaryDirectory(prefix="fintech-showcase-") as directory:
            config = Path(directory) / "config.yml"
            config.write_text("{}\n")
            processes.append(subprocess.Popen([tool("cloudflared"), "tunnel", "--config", str(config),
                                               "--protocol", "http2", "--edge-ip-version", "4",
                                               "--url", "http://127.0.0.1:8501", "--no-autoupdate"]))
            while True:
                if any(process.poll() is not None for process in processes):
                    raise RuntimeError("A showcase process exited. The session is stopping.")
                time.sleep(1)
    finally:
        stop_children(processes)
        if prepared and KUBECONFIG.exists():
            for name in (NAME, "copilot-sessions"):
                try:
                    if kubectl("-n", NAMESPACE, "get", "deployment", name, "--ignore-not-found",
                               "-o", "name", capture=True).stdout.strip():
                        kubectl("-n", NAMESPACE, "scale", "deployment/" + name, "--replicas=0")
                except subprocess.CalledProcessError:
                    print(f"Could not scale down {name}. Check the fintech-showcase cluster.", file=sys.stderr)
            print("Showcase processes stopped; cluster retained for the next start.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check tools, Docker and ports without deploying")
    args = parser.parse_args()
    os.chdir(ROOT)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        if args.check:
            preflight()
            print("Showcase prerequisites passed.")
        else:
            run()
    except KeyboardInterrupt:
        pass
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Showcase failed: {exc}\n")


if __name__ == "__main__":
    main()
