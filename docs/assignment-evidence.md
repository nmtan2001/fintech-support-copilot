# Assessment evidence

Do not mark an item complete without an actual run, URL or captured output.
Store sanitized evidence in docs/evidence; never save tokens or kubeconfig files.

| Requirement | Implementation | Live evidence needed |
| --- | --- | --- |
| Application functions, health and version | API + HTTP tests | Running endpoint responses |
| Branch and pull request workflow | feature branch + workflow | PR URL and required checks |
| Versioned Docker images | publish job, SHA/semantic tags | GHCR tags and image digests |
| Test failure blocks publication | job dependencies | Intentional failing PR run and its corrected run |
| Kubernetes configuration | k8s manifests, scoped RBAC | Two ready replicas, Service, probes, resource settings |
| Automated delivery | digest-based deploy + verification | Successful Actions deploy and live version |
| Recovery | resource snapshots + restore | Failed rollout followed by verified previous release |
| MLOps gate | pinned model + evaluation thresholds | Model-evaluation artifact linked to image and commit |
| Supply chain | lock hashes, pinned base/actions, SBOM | Successful scan and SBOM artifact |
| Performance | load-test utility and metrics | Hardware/limits, concurrency, throughput, p50/p95 and failures |

## Required release sequence

1. v1.0.0: establish and demonstrate the functional automated baseline.
2. v1.1.0: introduce a meaningful feature through a PR (capabilities is one candidate).
3. v1.2.0: document a real bug with a failing regression test, fix and release it.
4. Demonstrate a source change live through the complete pipeline.

For each release retain the source commit, PR URL, Actions run URL, semantic tag,
image digest, model revision/evaluation report, rollout status and /version output.
Do not fabricate older releases or retag the same application as three changes.

The 5–10 page technical PDF and 8–12 slide presentation should explain the final
verified system. Original submission/ files are preserved historical documents.
Update assessment deliverables after obtaining real release evidence.
