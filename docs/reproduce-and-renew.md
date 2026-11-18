# Reproduce the deployment and renew temporary resources

Read runtime-state.json first. It records what was created, whether it is still
running, and the actual cluster-credential expiration. Do not commit credentials.
The cluster created for this work is a local demonstration environment, not an
always-on managed service. Docker must remain running for its application to run.

## What expires or stops

| Item | Lifetime / failure trigger | How to recover |
| --- | --- | --- |
| KUBECONFIG_CONTENT | See deployer.expires_utc and deployer.expires_budapest in runtime-state.json | Generate a new scoped kubeconfig and replace the environment secret |
| GHCR_READ_TOKEN | Expiry chosen when you created the classic PAT; encrypted secret does not expose it | Check GitHub Settings → Developer settings → Tokens (classic), create read:packages replacement, update the environment secret |
| HF_TOKEN | Expiry/revocation controlled in your Hugging Face token settings; not visible here | Create a private-model read token and replace the repository secret |
| Runner registration token | GitHub short-lived registration token; request just before setup | Request a new registration token; never reuse one from an old guide |
| Temporary deployment runner | Approved for this demonstration only; removed afterwards | Register and start a trusted runner again before pushing a release |
| Local kind cluster | Persists in Docker until deleted; needs Docker running | Start Docker and inspect the named cluster; recreate if it was removed |
| Model cache | Pod-local temporary volume; lost when Pod is replaced | New Pod downloads the pinned model again using HF_TOKEN |
| Versioned images / evidence | Retention and package-deletion policy | Keep release tags/digests and download important Actions artifacts |

## 1. Prerequisites

Use WSL/Linux with Python 3.11, Git, GitHub CLI, Docker Desktop WSL integration,
kubectl v1.34.0 and kind v0.30.0. Sign in to GitHub as the repository owner; verify
`gh auth status`. If several accounts are configured, use `gh auth switch --user
nmtan2001` deliberately. Keep the repo's Python environment active for make check.

Install kind and kubectl using their official instructions and verify versions.
Use the node image pinned in .github/actions/build-and-scan/action.yml. No GPU is
required. Allow disk/memory for Docker images, the kind node, two model replicas
and a third Pod during a rolling update.

## 2. Restore or create the cluster

Use a dedicated kubeconfig, never overwrite a personal/default cluster context:

    mkdir -p .runtime
    chmod 700 .runtime
    kind get clusters
    kind create cluster --name fintech-validation --image kindest/node:v1.34.0 --kubeconfig .runtime/admin-kubeconfig --wait 120s
    export KUBECONFIG="$PWD/.runtime/admin-kubeconfig"
    kubectl get nodes
    kubectl apply -f k8s/namespace.yaml
    kubectl apply -f k8s/rbac.yaml

Create .runtime first with permissions 700. It is ignored by Git and Docker.
If the cluster already exists but you lost its config, use:

    kind export kubeconfig --name fintech-validation --kubeconfig .runtime/admin-kubeconfig

During this session ordinary Docker pulls had DNS failures; the build engine
could fetch the pinned image. The optional recovery is a temporary Dockerfile
containing FROM kindest/node:v1.34.0@sha256:7416a61b42b1662ca6ca89f02028ac133a309a2a30ba309614e8ec94d976dc5a,
built with tag kindest/node:v1.34.0. Then rerun kind with that local image. Do not
change system DNS or restart unrelated running containers as a first response.

## 3. Renew namespace-scoped deployment access

The deployer cannot create namespaces or administer the cluster. The administrator
bootstraps RBAC once. Then run from the repository root:

    python scripts/renew_cluster_credentials.py --admin-kubeconfig .runtime/admin-kubeconfig --output /tmp/fintech-deployer-kubeconfig --duration 24h --upload

--upload explicitly transmits the scoped bearer-token kubeconfig to this repo's
kubernetes-demo environment as KUBECONFIG_CONTENT. Omit it for local-only use.
The script prints the actual expiry, never the token. Use a private Linux
filesystem for credential files: Windows-mounted folders may not enforce chmod
600, in which case renewal deliberately refuses to write a bearer token. Update runtime-state.json
with that printed timestamp after renewal. A server can cap requested durations.
Verify the permissions without printing any credential:

    KUBECONFIG=/tmp/fintech-deployer-kubeconfig kubectl auth can-i patch deployments -n fintech-support
    KUBECONFIG=/tmp/fintech-deployer-kubeconfig kubectl auth can-i create namespaces

The expected answers are yes and no respectively. Keep administrator credentials
local. Remove scoped credential files when no longer needed.

## 4. Restore GitHub secrets and settings

Settings → Environments → kubernetes-demo must contain KUBECONFIG_CONTENT,
GHCR_USERNAME and GHCR_READ_TOKEN. Repository Actions secrets must contain HF_TOKEN.
The registry token is a classic PAT with read:packages; model token needs read
access to the pinned private repository. Store values through GitHub's UI or
CLI stdin, not in source code, shell history, screenshots or report attachments.

Repository variable DEPLOY_RUNNER is fintech-deploy. KUBECTL_VERSION is v1.34.0.
The runner must reach the API endpoint in the kubeconfig. A GitHub-hosted runner
cannot reach a private local cluster just because a kubeconfig exists.

## 5. Register and run a trusted deployment runner

In repo Settings → Actions → Runners → New self-hosted runner, choose Linux x64.
Download the official archive and verify its published SHA-256. Extract to a
private Linux directory such as "$HOME/.local/share/fintech-runner". Run GitHub's generated
configuration command from that directory, with these additional choices:

    ./config.sh --url https://github.com/nmtan2001/fintech-support-copilot --token REGISTRATION_TOKEN --unattended --name fintech-validation-deployer --labels fintech-deploy --work _work
    ./run.sh

REGISTRATION_TOKEN is GitHub's short-lived setup token, not your broad personal
access token. Enter it immediately before configuration. Keep the runner terminal
open during releases. This runs workflow commands as your local user: use it only
for the trusted main/tag deploy job, not untrusted pull-request code.

When finished, stop the listener, obtain a fresh removal token from GitHub's
runner page and follow its ./config.sh remove command. Verify the runner no longer
appears in Settings → Actions → Runners. Its directory contains credentials and
job workspaces; protect or remove it deliberately. No always-on runner service is
installed by this project.

## 6. Release, verify, and demonstrate

Run make check, push a feature branch and create a PR. Require the tests and
container gate to pass before merging. Update VERSION and pyproject.toml together
for each new semantic release. A main push publishes and deploys a SHA image; a
matching vX.Y.Z tag publishes the semantic tag. Both must refer to current main.

Watch Actions and the runner terminal. Then inspect:

    kubectl -n fintech-support get deployment,pods,service
    kubectl -n fintech-support rollout status deployment/fintech-support-copilot
    kubectl -n fintech-support port-forward service/fintech-support-copilot 8000:80

In another terminal query /version and /ready and run scripts/verify_release.py
with the exact expected version and full source commit. Save the run URL, image
digest, model report and rollout output. Repeat with genuinely different feature
and bug-fix commits for v1.1.0 and v1.2.0. Never reuse a release tag.

## Troubleshooting

- Queued deploy: the fintech-deploy runner is absent, offline or busy.
- Unauthorized cluster: kubeconfig expired, wrong API endpoint or invalid token.
- ImagePullBackOff: check GHCR username, PAT expiry and read:packages permission.
- Model startup failure: check HF_TOKEN access, model revision, network and memory.
- Wrong live model after config edit: inspect checksum/config Pod annotation and rollout.
- Failed candidate: retain diagnostics, verify previous identity after recovery.
- A green CI cluster test is not proof the separate final deployment succeeded.

Delete the demonstration cluster only when you explicitly intend to remove the
running application: kind delete cluster --name fintech-validation. That deletion
is not part of a normal release or credential renewal.

After a machine restart, start Docker Desktop first. Cached public tools and
verified archives may live under ignored .runtime/, but runner credentials and
bearer-token kubeconfigs should use a private Linux filesystem. Re-export the
kind administrator context only locally when renewal requires it, generate a
new scoped token, upload only that scoped config, and delete the temporary admin
file. A missing /tmp folder does not mean the Docker cluster was deleted.

Documentation-only pushes under docs/ do not redeploy the application. Pull
requests still run quality checks. This lets final evidence be updated without
changing the running release identity or queuing a job after the temporary
runner is removed. Functional source and configuration changes still trigger
the full main-branch delivery path.
