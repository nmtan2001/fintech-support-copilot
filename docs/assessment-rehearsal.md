# Assessment rehearsal

Use the release evidence index to show the three completed releases. The teacher
also requires a source change during the assessment: saved evidence cannot
replace that live exercise.

## Before the session

Confirm that the instructor can access the GitHub repository, its Actions runs,
and the submitted PDF/PowerPoint. A private repository needs an invitation
accepted before assessment; deployment secrets should remain private.

1. Start Docker Desktop and inspect the existing `fintech-validation` cluster.
2. Read `runtime-state.json`. Renew the scoped kubeconfig if expired and replace
   `kubernetes-demo/KUBECONFIG_CONTENT`, following `reproduce-and-renew.md`.
3. Check GHCR and Hugging Face token access. Register the temporary trusted runner
   again; the demonstration runner is removed after this implementation session.
4. Confirm two ready replicas and port-forward the Service. Show `/ready`,
   `/version`, `/capabilities` and a synthetic `/support` request.
5. Open Actions and the cluster terminal together. Allow time for model download
   on replacement Pods; the application is local and requires the machine online.

## Perform the live change

Create `feature/assessment-update` from current main. Make a small observable
application change, such as adding a useful field to `/capabilities`, and update
its existing HTTP test. Increase both `VERSION` and `pyproject.toml` to a new
unused version, incrementing the version on current main. Run `make check`,
push and open a PR. Explain the diff and show successful required checks before
merging. This must change application source, because docs-only pushes do not
deploy.

Watch the main workflow build, scan, evaluate, publish and deploy. While the new
Pod starts, inspect old ready Pods and the zero-unavailable rollout policy. After
main succeeds, run `scripts/verify_release.py` with the new version and full merge
commit. Show the changed endpoint. Create the matching unused semantic tag on
that exact commit, push it, and wait for tag verification before any next release.
Record the run URL and deployed digest. Stop and unregister the runner afterwards.

## Explain the engineering decisions

- CI checks candidate changes. Delivery prepares a releasable artifact;
  deployment automatically applies it to the configured environment.
- A failed test prevents dependent image-publication and deployment jobs. Show
  failed run 44107605890 and the corrected PR rather than creating a fake failure.
- Unique tags identify releases; deployment uses an immutable image digest.
  Semantic tags promote the verified main image, with its source label checked.
- An image packages the application. A container runs it. A Pod is Kubernetes'
  scheduling unit containing containers and shared resources.
- The Deployment gradually replaces Pods. Readiness controls traffic eligibility;
  liveness restarts a stuck process; startup allows model initialization first.
- CI uses namespace-scoped credentials through encrypted environment secrets and
  a runner that can reach the local API. It does not receive cluster-admin access.
- A failed rollout collects diagnostics, restores Deployment, ConfigMap and
  Service snapshots, and verifies the previous identity. The workflow still fails.
- Versioned configuration makes changes reviewable and recovery reproducible.
- The MLOps contribution is model and dataset identity, reproducible evaluation,
  explicit acceptance gates and inference monitoring. Automatic retraining and
  validated drift detection are future work.
