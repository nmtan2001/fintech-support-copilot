"""Create a namespace-scoped kubeconfig; optionally upload it to GitHub."""
import argparse
import base64
import datetime
import json
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admin-kubeconfig", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--duration", default="24h")
    parser.add_argument("--repository", default="nmtan2001/fintech-support-copilot")
    parser.add_argument("--upload", action="store_true", help="Explicitly store credentials in the GitHub deployment environment")
    args = parser.parse_args()
    env = {**os.environ, "KUBECONFIG": str(args.admin_kubeconfig.resolve())}
    config = json.loads(subprocess.check_output(["kubectl", "config", "view", "--minify", "--raw", "-o", "json"],
                                              env=env, text=True))
    token = subprocess.check_output(["kubectl", "-n", "fintech-support", "create", "token",
                                     "copilot-deployer", "--duration=" + args.duration], env=env, text=True).strip()
    content = {"apiVersion": "v1", "kind": "Config", "clusters": config["clusters"],
               "users": [{"name": "copilot-deployer", "user": {"token": token}}],
               "contexts": [{"name": "fintech", "context": {"cluster": config["clusters"][0]["name"],
                             "user": "copilot-deployer", "namespace": "fintech-support"}}], "current-context": "fintech"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(descriptor, 0o600)
    if os.fstat(descriptor).st_mode & 0o077:
        os.close(descriptor)
        args.output.unlink()
        raise PermissionError("Credential output must support private permissions; use a private Linux directory")
    with os.fdopen(descriptor, "w") as target:
        json.dump(content, target)
    claims = json.loads(base64.urlsafe_b64decode(token.split(".")[1] + "==="))
    expiry = datetime.datetime.fromtimestamp(claims["exp"], datetime.timezone.utc)
    if args.upload:
        subprocess.run(["gh", "secret", "set", "KUBECONFIG_CONTENT", "--repo", args.repository,
                        "--env", "kubernetes-demo"], input=json.dumps(content), text=True, check=True)
    print("Credential expires at " + expiry.isoformat())
    print("Output contains a bearer token: keep it outside Git and restrict access.")


if __name__ == "__main__":
    main()
