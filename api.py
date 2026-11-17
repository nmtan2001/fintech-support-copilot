"""Optional FastAPI interface to the synthetic fintech support pipeline."""

import importlib.util
import json
import time
import uuid
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field, StringConstraints

from fintech_support.copilot import (
    DEFAULT_NVIDIA_MODEL,
    DEFAULT_NVIDIA_TOOL_MODEL,
    IntentClassifier,
    SupportCopilot,
)
from fintech_support.demo_tools import SUPPORTED_INTENTS, DemoBankingTools, initialize_demo_database
from fintech_support.retrieval import FAQRetriever, load_faqs


ROOT = Path(__file__).resolve().parent
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
if not LOGGER.handlers:
    LOGGER.addHandler(logging.StreamHandler())
LOGGER.propagate = False
MODEL_MANIFEST = json.loads((Path(__file__).resolve().parent / "model-manifest.json").read_text())
MODEL_ID = os.environ.get("HF_MODEL_ID", MODEL_MANIFEST["model_id"])
MODEL_REVISION = os.environ.get("HF_MODEL_REVISION", MODEL_MANIFEST["revision"])
APP_COMMIT = os.environ.get("APP_COMMIT", "unknown")
VERSION_FILE = ROOT / "VERSION"
APP_VERSION = os.environ.get(
    "APP_VERSION",
    VERSION_FILE.read_text(encoding="utf-8").strip() if VERSION_FILE.is_file() else "dev",
)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load inference dependencies before marking the service ready."""
    application.state.ready = False
    try:
        get_copilot()
        application.state.ready = True
    except Exception:
        LOGGER.exception("Support model could not be loaded; startup will fail")
        raise
    yield
    application.state.ready = False


app = FastAPI(title="Fintech Support Copilot (Synthetic Demo)", lifespan=lifespan)


REGISTRY = CollectorRegistry()
REQUESTS = Counter("copilot_http_requests_total", "HTTP requests", ["method", "route", "status"], registry=REGISTRY)
SUPPORT_RESPONSES = Counter("copilot_support_responses_total", "Support outcomes", ["intent", "status"], registry=REGISTRY)
CONFIDENCE = Histogram("copilot_prediction_confidence", "Uncalibrated classifier confidence",
                       buckets=(0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0), registry=REGISTRY)
LATENCY = Histogram("copilot_http_request_seconds", "HTTP request duration", ["method", "route"], registry=REGISTRY)


@app.middleware("http")
async def observe_request(request: Request, call_next):
    request_id = uuid.uuid4().hex
    start = time.monotonic()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        route = getattr(request.scope.get("route"), "path", "unmatched")
        elapsed = time.monotonic() - start
        REQUESTS.labels(request.method, route, str(status)).inc()
        LATENCY.labels(request.method, route).observe(elapsed)
        LOGGER.info(json.dumps({"event": "http_request", "request_id": request_id,
                                "method": request.method, "route": route,
                                "status": status, "duration_seconds": round(elapsed, 6)}))


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(generate_latest(REGISTRY), headers={"Content-Type": CONTENT_TYPE_LATEST})


@app.get("/capabilities")
def capabilities():
    return {"lookup_intents": sorted(SUPPORTED_INTENTS),
            "reference_examples": ["TX-CARD-001", "TX-TRF-001", "CARD-001"],
            "synthetic_only": True, "read_only": True,
            "generation_available": bool(os.environ.get("NVIDIA_API_KEY"))}


class SupportRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    record_reference: str | None = None
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    use_llm: bool = False
    llm_select_tool: bool = False


@lru_cache(maxsize=1)
def get_demo_tools() -> DemoBankingTools:
    database_path = Path(os.environ.get("DEMO_DATABASE_PATH", ROOT / "outputs/demo/demo.sqlite"))
    initialize_demo_database(database_path, ROOT / "data/transactions/demo_records.json")
    return DemoBankingTools(database_path)


@lru_cache(maxsize=1)
def get_copilot() -> SupportCopilot:
    model_dir = Path(os.environ.get("HF_MODEL_PATH", ROOT / "outputs/distilbert/model"))
    baseline_path = Path(os.environ.get("BASELINE_MODEL_PATH", ROOT / "outputs/baseline/model.joblib"))
    model_available = importlib.util.find_spec("transformers") and importlib.util.find_spec("torch")
    classifier, _ = IntentClassifier.from_artifacts(
        transformer_dir=model_dir,
        baseline_path=baseline_path,
        prefer_transformer=bool(model_available),
        model_id=MODEL_ID,
        token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
        revision=MODEL_REVISION,
    )
    return SupportCopilot(
        classifier,
        FAQRetriever(load_faqs(ROOT / "data/faq/demo_policies.json")),
        get_demo_tools(),
    )


@app.get("/health")
def health():
    return {"status": "ok", "data": "synthetic_only"}


@app.get("/ready")
def readiness(request: Request):
    if not getattr(request.app.state, "ready", False):
        raise HTTPException(status_code=503, detail="The support model is not ready")
    return {"status": "ready", "version": APP_VERSION}


@app.get("/version")
def version():
    return {"name": "fintech-support-copilot", "version": APP_VERSION, "commit": APP_COMMIT,
            "model_id": MODEL_ID, "model_revision": MODEL_REVISION}


@app.get("/faqs")
def faqs():
    entries = load_faqs(ROOT / "data/faq/demo_policies.json")
    return {
        "count": len(entries),
        "items": [
            {
                "id": entry.id,
                "title": entry.title,
                "content": entry.content,
                "intents": list(entry.intents),
                "source": entry.source,
            }
            for entry in entries
        ],
        "synthetic_only": True,
    }


@app.get("/demo-status/{reference}")
def demo_status(reference: str):
    normalized = reference.strip().upper()
    if normalized.startswith("CARD-"):
        intent = "lost_or_stolen_card"
    elif normalized.startswith("TX-CARD-"):
        intent = "declined_card_payment"
    elif normalized.startswith("TX-TRF-"):
        intent = "pending_transfer"
    else:
        raise HTTPException(status_code=400, detail="Use a supported synthetic demo reference")

    try:
        result = get_demo_tools().call(intent, normalized)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (RuntimeError, FileNotFoundError, OSError) as exc:
        LOGGER.exception("Synthetic lookup failed")
        raise HTTPException(status_code=503, detail="Lookup service temporarily unavailable") from exc
    if result is None:
        raise HTTPException(status_code=404, detail="No supported synthetic lookup was selected")
    return {**asdict(result), "synthetic_only": True}


@app.post("/support")
def support(payload: SupportRequest):
    if payload.llm_select_tool and not payload.use_llm:
        raise HTTPException(status_code=400, detail="Enable use_llm to select tools with NVIDIA NIM")
    if (payload.use_llm or payload.llm_select_tool) and not (
        os.environ.get("NVIDIA_API_KEY")
    ):
        raise HTTPException(status_code=400, detail="Set NVIDIA_API_KEY to use generation")
    try:
        result = get_copilot().respond(
            payload.question,
            record_reference=payload.record_reference,
            min_confidence=payload.min_confidence,
            llm_model=DEFAULT_NVIDIA_MODEL if (payload.use_llm or payload.llm_select_tool) else None,
            llm_api_key=os.environ.get("NVIDIA_API_KEY") if (payload.use_llm or payload.llm_select_tool) else None,
            llm_select_tool=payload.llm_select_tool,
            llm_tool_model=(
                DEFAULT_NVIDIA_TOOL_MODEL
                if payload.llm_select_tool else None
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (RuntimeError, FileNotFoundError, OSError) as exc:
        LOGGER.exception("Support request failed")
        raise HTTPException(status_code=503, detail="Support service temporarily unavailable") from exc
    SUPPORT_RESPONSES.labels(result.intent, result.status).inc()
    CONFIDENCE.observe(result.confidence)
    return {
        "intent": result.intent,
        "confidence_uncalibrated": result.confidence,
        "status": result.status,
        "answer": result.answer,
        "retrieved_sources": [
            {"id": hit.faq.id, "score": hit.score, "source": hit.faq.source}
            for hit in result.retrieved
        ],
        "tool_result": asdict(result.tool_result) if result.tool_result else None,
        "tool_routing": result.tool_routing,
        "generation_status": result.generation_status,
        "grounding_issues": list(result.grounding_issues),
        "synthetic_only": True,
    }


BatchQuestion = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class BatchClassificationRequest(BaseModel):
    questions: list[BatchQuestion] = Field(min_length=1, max_length=16)


@app.post("/classify/batch")
def classify_batch(payload: BatchClassificationRequest):
    """Classify up to sixteen questions in order, without generation or lookups."""
    try:
        predictions = get_copilot().classifier.predict_many(payload.questions, batch_size=16)
    except (RuntimeError, FileNotFoundError, OSError) as exc:
        LOGGER.exception("Batch classification failed")
        raise HTTPException(status_code=503, detail="Classification service temporarily unavailable") from exc
    return {"count": len(predictions), "predictions": [
                {"intent": row.intent, "confidence_uncalibrated": row.confidence} for row in predictions],
            "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "synthetic_only": True}
