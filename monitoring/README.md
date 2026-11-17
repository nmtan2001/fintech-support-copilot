# Service and model monitoring

GET /metrics exports Prometheus counters and latency histograms. Request labels
use route templates so record references never become labels. HTTP logs include
random request IDs, route templates, status and duration; they omit questions,
request bodies, query strings and credentials. X-Request-ID connects a response
to its log entry. Model outcome counters and confidence distributions help detect
changes in usage; confidence is uncalibrated and is not an accuracy estimate.

For a local Prometheus, port-forward the Service to port 8000, then mount
prometheus.yml into a Prometheus container. The example target assumes Docker
Desktop. In Kubernetes, use pod discovery/scraping to preserve per-replica metrics;
scraping only a Service IP can merge counters from different replicas.

Useful PromQL for a dashboard:

- Requests/second: sum(rate(copilot_http_requests_total[5m]))
- HTTP failure ratio: sum(rate(copilot_http_requests_total{status=~"5.."}[5m])) / sum(rate(copilot_http_requests_total[5m]))
- p95 latency: histogram_quantile(0.95, sum by (le) (rate(copilot_http_request_seconds_bucket{route="/support"}[5m])))
- Intent mix: sum by (intent) (rate(copilot_support_responses_total[5m]))
- Response outcomes: sum by (status) (rate(copilot_support_responses_total[5m]))

Investigate sustained 5xx errors and latency against a measured load-test baseline.
A confidence or intent-distribution shift should trigger investigation and labeled
evaluation, not automatic retraining. This project has no real customer labels
or validated production drift detector.

Import grafana-dashboard.json into Grafana and select a Prometheus datasource.
It includes six service/model panels; no credentials are embedded.
