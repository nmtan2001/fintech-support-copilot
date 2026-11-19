# Assessment evidence

This is a DevOps project with MLOps controls for a served classifier. It has
reproducible model evaluation and release gates; it has no automatic retraining
or validated production drift detector. All banking records are synthetic.

[Model-serving scope and comparison with the supplied article](model-serving-scope.md).

## Release history

| Release | Source change / PR | Successful workflow | Captured live evidence |
| --- | --- | --- | --- |
| v1.0.0 | [Hardened baseline, PR #1](https://github.com/nmtan2001/fintech-support-copilot/pull/1) | [Tag and deployment](https://github.com/nmtan2001/fintech-support-copilot/actions/runs/44118506901) | [Commit, image digest, two ready replicas](evidence/release-v1.0.0.json) |
| v1.1.0 | [Bounded batch classification, PR #2](https://github.com/nmtan2001/fintech-support-copilot/pull/2) | [Main deployment](https://github.com/nmtan2001/fintech-support-copilot/actions/runs/44500542281), [semantic promotion](https://github.com/nmtan2001/fintech-support-copilot/actions/runs/44503313060) | [Live batch and support verification](evidence/release-v1.1.0.json), [rolling-update snapshot](evidence/v1.1.0-rolling-update.json) |
| v1.2.0 | [Server access-log privacy fix, PR #3](https://github.com/nmtan2001/fintech-support-copilot/pull/3) | [Main deployment](https://github.com/nmtan2001/fintech-support-copilot/actions/runs/44504472666), [semantic promotion](https://github.com/nmtan2001/fintech-support-copilot/actions/runs/44507546525) | [Exact live identity and readiness](evidence/release-v1.2.0.json) |

Semantic tags identify distinct source releases. For v1.1.0 onward, tag jobs
promote the previously published source image, check its OCI revision and repeat
quality gates. The v1.1.0 and v1.2.0 tags each have the same registry digest as their respective
verified main deployment. Final hosted tests: 116 passed, one optional local
baseline-artifact test skipped. The real-model benchmark passed separately.

## Verification and failures

| Requirement | Evidence |
| --- | --- |
| Failed tests prevent publication | [Failed run 44107605890](https://github.com/nmtan2001/fintech-support-copilot/actions/runs/44107605890), [regression explanation](bugs/demo-lookup-unavailable.md) |
| Corrected baseline passes | [Successful run 44111215418](https://github.com/nmtan2001/fintech-support-copilot/actions/runs/44111215418): tests, Ruff, scan, real container, model evaluation, isolated Kubernetes recovery |
| Official classifier benchmark | [Retained CI report](evidence/ci-model-evaluation.json): 3,080 examples, 92.435% accuracy, 92.439% Macro-F1; minimum 90% for each |
| Distinct workflow evaluation | Same report: 38/48 synthetic cases pass; failures remain visible; delivered extractive grounding 100% |
| Performance sample | [CI load measurement](evidence/ci-load-measurement.json); 20 warm synthetic requests, concurrency two; no production capacity claim |
| Privacy bug and regression | [Real-container canary reproduction](bugs/request-log-privacy.md); v1.2.0 checks server logs before publication |
| Supply chain | Hashed lockfiles, pinned base/actions, Trivy and retained CycloneDX `container-sbom` artifact in successful workflows |
| Kubernetes recovery | CI creates an isolated cluster, verifies two replicas, applies an invalid image and verifies restoration; deployment code separately snapshots and restores resources |

## Deliverables and repeatability

- [Seven-page technical report](deliverables/engineering-report.pdf)
- [Ten-slide editable presentation](deliverables/engineering-presentation.pptx)
- [Reproduction, token renewal and temporary runner guide](reproduce-and-renew.md)
- [Actual expirations and runtime state](runtime-state.json)
- [Live assessment rehearsal and explanation guide](assessment-rehearsal.md)

The assignment requires a fresh source change during the assessment. Saved
screenshots or these release runs do not replace that exercise. The rehearsal
guide explains how to run it through the same pipeline.

Original `submission/` and `slide-deck/` documents are preserved. The deployed
cluster is local and needs Docker and the host running. No public ingress,
managed high availability or production financial suitability is claimed.

Final release reports: [model gate](evidence/v1.2.0-model-evaluation.json),
[load sample](evidence/v1.2.0-load-measurement.json),
[CycloneDX SBOM](evidence/v1.2.0-sbom.cdx.json).

The temporary deployment runner has been stopped and unregistered; its local
credentials and job workspace were removed. Re-register before future delivery.
