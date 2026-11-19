# Why this is DevOps plus MLOps

The DevOps work automates tests, image scanning, publication and Kubernetes
release verification. The MLOps work governs the model artifact and its behavior:
an immutable model revision and weight fingerprint, a checksum-pinned benchmark,
accuracy and Macro-F1 release gates, retained evaluation reports and inference
metrics. Serving a model on Kubernetes alone would not provide all these controls.

The [article supplied for comparison](https://medium.com/@pablo.lopez.santori/deploying-an-open-weights-llm-as-an-api-in-kubernetes-01d7c6aaee43)
serves a quantized Qwen model through llama.cpp on a GPU. It describes startup
probes, persistent model storage, GPU constraints and performance measurements.
This project serves a smaller fine-tuned DistilBERT intent classifier on CPU,
with two replicas and a zero-unavailable rolling update. Optional generative
answers call a remote NVIDIA service; the project does not self-host that LLM.

The useful common lessons are allowing for model cold starts, measuring actual
resource and request behavior, handling credentials carefully, and distinguishing
an illustrative deployment from validated production capacity. This repository
adds application and model gates to its delivery path. Its current limitations
include temporary Pod-local model caches, a local single-node cluster, no public
authenticated ingress, no automated retraining, and no validated drift detector.
Persistent model caching and sustained load measurement are reasonable future
extensions after the assessed pipeline is understood and reproducible.
