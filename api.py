"""Optional FastAPI interface to the synthetic fintech support pipeline."""

import importlib.util
import os
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fintech_support.copilot import DEFAULT_HF_MODEL_ID, IntentClassifier, SupportCopilot
from fintech_support.demo_tools import DemoBankingTools, initialize_demo_database
from fintech_support.retrieval import FAQRetriever, load_faqs


ROOT = Path(__file__).resolve().parent
app = FastAPI(title="Fintech Support Copilot (Synthetic Demo)")


class SupportRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    record_reference: str | None = None
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    use_llm: bool = False
    llm_select_tool: bool = False


@lru_cache(maxsize=1)
def get_copilot() -> SupportCopilot:
    backend = os.environ.get("INTENT_BACKEND", "local")
    if backend not in {"local", "hf"}:
        raise ValueError("INTENT_BACKEND must be local or hf")
    model_dir = ROOT / "outputs/distilbert/model"
    model_available = importlib.util.find_spec("transformers") and importlib.util.find_spec("torch")
    if backend == "hf":
        classifier, _ = IntentClassifier.from_huggingface(
            model_id=os.environ.get("HF_MODEL_ID", DEFAULT_HF_MODEL_ID),
            endpoint_url=os.environ.get("HF_INFERENCE_ENDPOINT_URL"),
        )
    else:
        classifier, _ = IntentClassifier.from_artifacts(
            transformer_dir=model_dir,
            baseline_path=ROOT / "outputs/baseline/model.joblib",
            prefer_transformer=bool(model_available),
            model_id=os.environ.get("HF_MODEL_ID"),
            token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
            revision=os.environ.get("HF_MODEL_REVISION"),
        )
    database_path = ROOT / "outputs/demo/demo.sqlite"
    initialize_demo_database(database_path, ROOT / "data/transactions/demo_records.json")
    return SupportCopilot(
        classifier,
        FAQRetriever(load_faqs(ROOT / "data/faq/demo_policies.json")),
        DemoBankingTools(database_path),
    )


@app.get("/health")
def health():
    return {"status": "ok", "data": "synthetic_only"}


@app.post("/support")
def support(payload: SupportRequest):
    if payload.llm_select_tool and not payload.use_llm:
        raise HTTPException(status_code=400, detail="Enable use_llm to select tools with NVIDIA")
    if (payload.use_llm or payload.llm_select_tool) and not (
        os.environ.get("NVIDIA_API_KEY") and os.environ.get("NVIDIA_MODEL")
    ):
        raise HTTPException(status_code=400, detail="Set NVIDIA_API_KEY and NVIDIA_MODEL to use generation")
    try:
        result = get_copilot().respond(
            payload.question,
            record_reference=payload.record_reference,
            min_confidence=payload.min_confidence,
            llm_model=os.environ.get("NVIDIA_MODEL") if (payload.use_llm or payload.llm_select_tool) else None,
            llm_api_key=os.environ.get("NVIDIA_API_KEY") if (payload.use_llm or payload.llm_select_tool) else None,
            llm_select_tool=payload.llm_select_tool,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (RuntimeError, FileNotFoundError, OSError) as exc:
        raise HTTPException(status_code=503, detail=f"Demo service unavailable: {exc}") from exc
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
