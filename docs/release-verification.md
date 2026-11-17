# Release verification

Docker builds accept APP_VERSION and APP_COMMIT. Both are OCI labels and runtime
values returned by /version. model-manifest.json pins the private model revision
and documents model-release thresholds. requirements-api.lock and
requirements-dev.lock pin transitive dependencies and their allowed hashes.
The base image and third-party CI actions are pinned to immutable digests/commits.
Update these pins through reviewed maintenance PRs rather than leaving them stale.

## Gates

1. Unit and HTTP tests plus Ruff.
2. Build the exact candidate image; Trivy blocks fixable HIGH/CRITICAL findings.
3. Run that image and check readiness, version, commit, FAQ fixtures and support.
4. Evaluate the real model on authored synthetic cases, with NVIDIA disabled.
5. Retain model-evaluation.json and a CycloneDX SBOM as workflow artifacts.
6. Publish only after success; pass the registry digest to delivery.
7. Reject stale/non-main commits, serialize deployment, snapshot previous resources.
8. Apply configuration, wait for rollout and rerun HTTP verification.
9. On failure restore prior resources, verify recovery, and keep the job failed.
10. Collect diagnostics and remove temporary runner credentials.

The HTTP verifier uses explicit expected version and source commit. Transport and
HTTP errors retry during startup; incorrect identity or content fails immediately.
The Kubernetes port-forward checks a Service-selected pod, not public ingress or
every replica individually. Rollout status separately checks the Deployment.

The classifier gate uses all 3,080 official BANKING77 test examples, pinned by
source commit and SHA-256, with 90% minimum accuracy and Macro-F1. The authored
workflow suite is reported separately: it includes oracle/component scenarios
(e.g. nonsense input assigned a placeholder intent) and is not a classification
benchmark. Workflow task success must reach 75% and delivered extractive
faithfulness 100%. These are project acceptance criteria, not real-bank quality
estimates. Existing benchmark data was not used to tune the deployed checkpoint.
A config update changing model revision changes the served model even with the
same application image: retain the model manifest and ConfigMap with each release.

## Reproduce

make check
make build
make run
make verify

Inside the running container: python -m scripts.evaluate_release
For a live cluster, use python3 -m scripts.deploy --image REGISTRY/IMAGE@sha256:DIGEST
with the expected --version and --commit, after bootstrap and secret setup from
the operations runbook. This command mutates the configured cluster.

The namespace, deployer credentials and private-model/registry tokens must be set
up before automated CD. Local source checks do not prove registry publication or
successful real rollouts. See assignment-evidence.md for outstanding evidence.
