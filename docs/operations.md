# Operations and reproduction

## Scope

This is a CPU model-serving application with a DevOps delivery pipeline and an
MLOps model-release gate. DistilBERT is served locally. Optional NVIDIA generation
is remote. It is not a self-hosted generative LLM service. All banking records are
synthetic and read-only. No external public ingress is provisioned by default.

## Clean checkout

Use Python 3.11, Docker, kubectl and a reachable Kubernetes cluster. Clone the
repository and create a virtual environment. Install requirements-dev.lock with
pip --require-hashes, then install the project editable with --no-deps. Run make
check. The build backend explicitly packages fintech_support, avoiding accidental
inclusion of datasets, generated models and submissions.

Export HF_TOKEN with read access to the private checkpoint. Do not put its value
in a command file, Dockerfile, image build argument, git-tracked manifest or log.
Run make build, make run in a separate terminal, then make verify. The running
container downloads the pinned safetensors model, not model weights from main.
Runtime credentials are not baked into the image. The model cache is ephemeral;
first startup depends on Hugging Face availability and may take several minutes.

Run python3 -m scripts.evaluate_release inside the container and retain
/tmp/model-evaluation.json. Its report separates the official BANKING77 classification benchmark from small
authored workflow fixtures; neither measures live traffic or live LLM quality. Thresholds live with the model
identity in model-manifest.json. Changing a threshold requires code review.

## Cluster bootstrap (administrator, once)

Apply k8s/namespace.yaml followed by k8s/rbac.yaml. Grant the runner only the
namespace-scoped copilot-deployer Role. Use short-lived credentials where your
cluster supports them. The example does not grant cluster-admin or namespace
creation privileges to CI. Rotate tokens and clean up runner workspaces.

For local experiments, create an isolated kind cluster with a dedicated
kubeconfig file. Do not overwrite an existing default kubeconfig or cluster.
A private local cluster requires a runner inside its reachable network.

## GitHub configuration

Repository secret HF_TOKEN: read-only access to the private inference model.
Environment kubernetes-demo secrets: KUBECONFIG_CONTENT, GHCR_USERNAME,
GHCR_READ_TOKEN (read:packages) and HF_TOKEN, or inherit the repository HF_TOKEN.
The workflow creates the registry and model access Secrets in fintech-support.
The namespace must already exist. Kubernetes Secrets need appropriate RBAC and
at-rest encryption on the cluster; base64 encoding alone is not encryption.

Set DEPLOY_RUNNER to a trusted self-hosted runner label for a local cluster.
Never execute untrusted PR jobs on that runner. Set KUBECTL_VERSION to the
cluster-compatible version instead of relying on the workflow's latest fallback.
Protect main, require tests and container checks, restrict who may push v* tags,
and configure environment permissions to fit your release policy.
Fork PRs cannot access the private model token: a maintainer must run the full
container/model check from a trusted branch before merging such a contribution.
Do not use pull_request_target to run untrusted code with this secret.

## Delivery and recovery

PRs and feature branches run tests, lint, a Docker build, Trivy and container/model
checks. Publication waits for these checks on the release branch. Main and release
tag pushes publish an image; CD consumes the pushed image digest. Only the current
main commit may deploy, and deployment jobs share a concurrency group.

The deployment script snapshots the previous Deployment, ConfigMap and Service,
applies the candidate, waits for rollout and checks its version, source commit and
support behavior. Any failure restores the saved resources and waits for recovery.
Failure still fails the job even after a successful recovery. On a first failed
release there is no previous release to restore; diagnostics remain available.
This does not roll back registry secrets, external models, or external providers.

Inspect: kubectl -n fintech-support get deployment,pods,service
Debug: kubectl -n fintech-support describe deployment fintech-support-copilot
Logs: kubectl -n fintech-support logs -l app.kubernetes.io/name=fintech-support-copilot
Access: kubectl -n fintech-support port-forward service/fintech-support-copilot 8000:80

For emergency manual recovery, select the last verified image digest and restore
its matching ConfigMap, then wait for rollout and rerun verification with its
version and source commit. A rollback is incomplete until the running app is
verified. Collect failure evidence before changing resources.

## Performance and monitoring

See monitoring/README.md. Warm the model, then run scripts/load_test.py at
concurrency 1, 2 and 4. Retain JSON output with CPU/memory limits, model revision,
image digest and hardware. Compare throughput, failures and p95 latency. A local
measurement is not a production SLA. Two replicas require resources for a third
pod during a zero-unavailable rolling update.

Releases from 1.1.0 allow a 30-minute rollout/progress budget and 15 minutes
of startup probes per container, accounting for measured private-model cold
starts and sequential surge replicas. Old replicas stay available during
updates. The first local 447 MB image pull took almost nine minutes.

Create a semantic release tag after the corresponding main pipeline succeeds.
Tag publication pulls and re-verifies that source commit's published SHA image
and assigns the semantic alias to the same artifact. It fails if the SHA image
is absent or its revision label disagrees; it does not silently rebuild a tag.
