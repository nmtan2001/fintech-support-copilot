# Preserved evidence

The release-v1.* JSON files contain captured live deployment identities, exact
image digests, ready-replica counts and workflow/PR links. They describe the
application when verified; they do not promise that the local host stays online.

The v1.2.0 model, load and CycloneDX SBOM reports are downloaded from successful
main run 44504472666. See ../runtime-state.json for Actions artifact expiry.
The older ci-* reports preserve the baseline measurements. Keep their results
separate: short load samples from different runs are not controlled benchmarks.

No credentials or kubeconfig files belong here. A synthetic rolling-update
snapshot records the available old replicas while a new model Pod initialized.
The isolated CI recovery exercise is distinct from the long-lived local cluster.
