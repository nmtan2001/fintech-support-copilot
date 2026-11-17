"""Measure synthetic support latency at a chosen bounded concurrency."""
import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args()
    if not (1 <= args.requests <= 10000 and 1 <= args.concurrency <= 100):
        parser.error("Use 1–10000 requests and concurrency 1–100")

    def request(_):
        start = time.monotonic()
        try:
            payload = json.dumps({"question": "My card payment was declined",
                                  "record_reference": "TX-CARD-001"}).encode()
            req = Request(args.url.rstrip("/") + "/support", data=payload,
                          headers={"Content-Type": "application/json"})
            with urlopen(req, timeout=30) as response:
                body = json.load(response)
            valid = body.get("tool_result", {}).get("reason_code") == "CARD_FROZEN"
        except Exception:
            valid = False
        return time.monotonic() - start, valid

    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        results = list(executor.map(request, range(args.requests)))
    elapsed = time.monotonic() - start
    durations = sorted(duration for duration, _ in results)
    success = sum(valid for _, valid in results)
    print(json.dumps({"requests": args.requests, "concurrency": args.concurrency,
                      "successful": success, "failed": args.requests - success,
                      "requests_per_second": args.requests / elapsed,
                      "p50_seconds": statistics.median(durations),
                      "p95_seconds": durations[max(0, int(len(durations) * 0.95) - 1)]}, indent=2))
    if success != args.requests:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
