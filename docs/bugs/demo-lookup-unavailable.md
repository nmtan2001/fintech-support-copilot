# Unavailable synthetic lookup returned an unhandled error

The HTTP adapter caught malformed-reference errors but let database/runtime
failures escape from GET /demo-status/{reference}. This produced a generic 500
and bypassed the route's normal service-unavailable response contract.

Reproduction: inject a RuntimeError from get_demo_tools, then request
/demo-status/TX-CARD-001 over HTTP. The regression test initially failed with
that uncaught exception. The fix handles runtime/filesystem failures as a
sanitized HTTP 503, logs internal diagnostics on the server, and keeps the
X-Request-ID header for troubleshooting. Invalid references still return 400.

Failing source commit: ca18c6bdd16902d523275d66aaf98c3946874bd5
Failing PR run: https://github.com/nmtan2001/fintech-support-copilot/actions/runs/44107605890
The test job failed; container build, publication and deployment were skipped.
The regression stays in tests/test_api_http.py. A passing run must be recorded
after the fix; this document does not claim that a corrected release is deployed.
