"""Verify release identity and synthetic functionality over real HTTP."""
import argparse
import json
import time
from urllib.error import URLError
from urllib.request import Request, urlopen


def verify(base_url: str, version: str, commit: str) -> None:
    def request(path, payload=None):
        data = json.dumps(payload).encode() if payload is not None else None
        req = Request(base_url.rstrip("/") + path, data=data,
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=20) as response:
            return json.load(response)

    assert request("/ready")["status"] == "ready", "Application not ready"
    identity = request("/version")
    assert identity["version"] == version, f"Unexpected version: {identity}"
    assert identity["commit"] == commit, f"Unexpected commit: {identity}"
    assert request("/health")["status"] == "ok", "Liveness failed"
    assert request("/faqs")["count"] > 0, "FAQ fixtures missing"
    record = request("/demo-status/TX-CARD-001")
    assert record["status"] == "declined", "Synthetic lookup failed"
    body = request("/support", {"question": "My card payment was declined",
                               "record_reference": "TX-CARD-001"})
    assert body["synthetic_only"] is True, "Synthetic boundary missing"
    assert body["answer"], "Support response is empty"
    assert body["intent"] == "declined_card_payment", "Unexpected support routing"
    assert body["tool_result"]["reason_code"] == "CARD_FROZEN", "Support lookup failed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--version", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--wait-seconds", type=float, default=0)
    args = parser.parse_args()
    deadline = time.monotonic() + args.wait_seconds
    while True:
        try:
            verify(args.url, args.version, args.commit)
            break
        except (URLError, TimeoutError, ConnectionError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(2)
    print(json.dumps({"verified": True, "version": args.version, "commit": args.commit}))


if __name__ == "__main__":
    main()
