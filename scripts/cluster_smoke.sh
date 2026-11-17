#!/usr/bin/env bash
# Run only against an isolated, disposable cluster with a fresh namespace.
set -euo pipefail
: "${IMAGE:?Set IMAGE}" "${APP_VERSION:?Set APP_VERSION}" "${APP_COMMIT:?Set APP_COMMIT}" "${HF_TOKEN:?Set HF_TOKEN}"
namespace=fintech-support
name=fintech-support-copilot
kubectl apply -f k8s/namespace.yaml
kubectl create secret generic "$name" -n "$namespace" \
  --from-literal=HF_TOKEN="$HF_TOKEN" --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f k8s/configmap.yaml
kubectl create --dry-run=client -f k8s/deployment.yaml -o json | python3 -c '
import json,os,sys
x=json.load(sys.stdin)
spec=x["spec"]["template"]["spec"]
spec.pop("imagePullSecrets",None)
spec["containers"][0]["image"]=os.environ["IMAGE"]
spec["containers"][0]["imagePullPolicy"]="Never"
x["metadata"].setdefault("annotations",{}).update({"copilot/version":os.environ["APP_VERSION"],"copilot/commit":os.environ["APP_COMMIT"]})
json.dump(x,sys.stdout)
' > "$RUNNER_TEMP/validated-deployment.json"
kubectl apply -f "$RUNNER_TEMP/validated-deployment.json"
kubectl apply -f k8s/service.yaml
kubectl rollout status deployment/"$name" -n "$namespace" --timeout=12m
python3 -c 'import os; from scripts.deploy import check_service; check_service(os.environ["APP_VERSION"], os.environ["APP_COMMIT"])'
# A bad candidate must fail while the previous two replicas continue serving.
kubectl set image deployment/"$name" -n "$namespace" api=invalid-candidate:unavailable
if kubectl rollout status deployment/"$name" -n "$namespace" --timeout=45s; then
  echo "Broken release unexpectedly succeeded" >&2
  exit 1
fi
kubectl apply -f "$RUNNER_TEMP/validated-deployment.json"
kubectl rollout status deployment/"$name" -n "$namespace" --timeout=3m
python3 -c 'import os; from scripts.deploy import check_service; check_service(os.environ["APP_VERSION"], os.environ["APP_COMMIT"])'
kubectl get deployment,pods,service -n "$namespace"
echo "Kubernetes rollout and recovery verified"
