# Raw server access logs exposed request query data

The application emitted safe structured events with route templates, but Uvicorn
also emitted its default access log with the full request target. A diagnostic
query can therefore enter logs even though the application's event logger omits
it. The README's privacy claim did not cover this duplicate server logger.

Reproduction on the original validation image: send a GET /health request with
a unique synthetic diagnostic query, then inspect container logs for the canary.
The pre-fix check returned: FAIL: request query data appeared in container access logs.
No real credential was used in the experiment.

The fix disables Uvicorn's duplicate access logger in the production entrypoint.
Structured application events remain enabled, preserving request IDs, route
names, status and duration. A real-container canary check runs before publication
in CI, alongside the existing HTTP, model, scan and Kubernetes checks.
