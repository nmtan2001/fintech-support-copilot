PYTHON ?= python3
APP_VERSION ?= $(shell cat VERSION)
APP_COMMIT ?= $(shell git rev-parse HEAD)
IMAGE ?= fintech-support-copilot:$(APP_VERSION)

.PHONY: test lint check build run verify

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check api.py app.py fintech_support scripts tests/test_api_http.py tests/test_release_verification.py tests/test_deployment.py tests/test_model_gate.py

check: lint test

build:
	docker build --build-arg APP_VERSION=$(APP_VERSION) --build-arg APP_COMMIT=$(APP_COMMIT) -t $(IMAGE) .

run:
	docker run --rm -p 127.0.0.1:8000:8000 --env HF_TOKEN $(IMAGE)

verify:
	$(PYTHON) scripts/verify_release.py --version $(APP_VERSION) --commit $(APP_COMMIT) --wait-seconds 600
