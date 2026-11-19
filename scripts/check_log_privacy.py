"""Verify that a synthetic request query never reaches container logs."""
import argparse
import subprocess
import uuid
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--container", required=True)
    args = parser.parse_args()
    canary = "privacy-canary-" + uuid.uuid4().hex
    with urlopen(args.url.rstrip("/") + "/health?diagnostic=" + canary, timeout=10) as response:
        response.read()
    logs = subprocess.check_output(["docker", "logs", args.container], stderr=subprocess.STDOUT, text=True)
    if canary in logs:
        raise SystemExit("Request query data appeared in container access logs")
    print("Request query data is absent from container logs")


if __name__ == "__main__":
    main()
