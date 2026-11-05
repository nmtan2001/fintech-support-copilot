"""Run the local demonstration using streamlit run app.py."""

import importlib.util
import os
from pathlib import Path

import streamlit as st

from fintech_support.copilot import (
    DEFAULT_HF_MODEL_ID,
    DEFAULT_NVIDIA_MODEL,
    DEFAULT_NVIDIA_TOOL_MODEL,
    IntentClassifier,
    SupportCopilot,
)
from fintech_support.demo_tools import DemoBankingTools, initialize_demo_database
from fintech_support.retrieval import FAQRetriever, load_faqs


ROOT = Path(__file__).resolve().parent


def get_setting(name: str, default: str | None = None) -> str | None:
    try:
        value = st.secrets.get(name)
    except st.errors.StreamlitSecretNotFoundError:
        value = None
    return value or os.environ.get(name, default)


@st.cache_resource(show_spinner=False)
def load_classifier(
    model_choice: str,
    model_id: str | None,
    model_token: str | None,
    model_revision: str | None,
):
    return IntentClassifier.from_artifacts(
        transformer_dir=ROOT / "outputs/distilbert/model",
        baseline_path=ROOT / "outputs/baseline/model.joblib",
        prefer_transformer=model_choice == "DistilBERT",
        model_id=model_id,
        token=model_token,
        revision=model_revision,
    )


@st.cache_resource(show_spinner=False)
def load_retriever():
    return FAQRetriever(load_faqs(ROOT / "data/faq/demo_policies.json"), backend="tfidf")


@st.cache_resource(show_spinner=False)
def load_demo_tools():
    database_path = ROOT / "outputs/demo/demo.sqlite"
    initialize_demo_database(database_path, ROOT / "data/transactions/demo_records.json")
    return DemoBankingTools(database_path)


def main():
    st.set_page_config(page_title="Fintech Support Copilot", layout="centered")
    st.title("Fintech Customer Support Copilot")
    st.caption("Academic prototype · fictional help articles and sample records · no live account access")

    transformer_downloaded = (ROOT / "outputs/distilbert/model/config.json").is_file()
    model_id = get_setting("HF_MODEL_ID", DEFAULT_HF_MODEL_ID)
    hf_token = get_setting("HF_TOKEN") or get_setting("HUGGINGFACE_HUB_TOKEN")
    transformer_configured = transformer_downloaded or bool(model_id)
    missing = [name for name in ("torch", "transformers") if importlib.util.find_spec(name) is None]
    transformer_available = transformer_configured and not missing
    baseline_available = (ROOT / "outputs/baseline/model.joblib").is_file()
    if transformer_available:
        model_choice = "DistilBERT"
    elif baseline_available:
        model_choice = "TF-IDF + Logistic Regression"
    else:
        model_choice = None

    if not transformer_downloaded and not model_id:
        st.info("Set HF_MODEL_ID in Streamlit Secrets to load the trained DistilBERT from Hugging Face.")
    if model_choice is None:
        st.error("No usable classifier. Download a model and install its dependencies, or run baseline.py.")
        st.stop()

    try:
        classifier, model_name = load_classifier(
            model_choice,
            model_id,
            hf_token,
            get_setting("HF_MODEL_REVISION"),
        )
    except (FileNotFoundError, ImportError, OSError, ValueError, RuntimeError) as exc:
        st.error(f"Could not load a classifier: {exc}")
        st.stop()

    nvidia_nim_api_key = get_setting("NVIDIA_API_KEY")
    use_llm = bool(nvidia_nim_api_key)
    llm_api_key = nvidia_nim_api_key if use_llm else None
    llm_model = DEFAULT_NVIDIA_MODEL if use_llm else None
    llm_tool_model = DEFAULT_NVIDIA_TOOL_MODEL if use_llm else None
    llm_select_tool = use_llm
    st.caption(
        f"Classifier: {model_name} | FAQ retrieval: TF-IDF | "
        + (f"NVIDIA answer: {llm_model} | tool selection: {llm_tool_model}" if use_llm
           else "LLM disabled: add NVIDIA_API_KEY to enable generation and function calling")
    )
    if transformer_configured and missing and model_choice != "DistilBERT":
        st.info(
            "DistilBERT needs " + ", ".join(missing)
            + "; using the TF-IDF + Logistic Regression fallback."
        )

    question = st.text_area(
        "Customer question",
        value="Why was my TX-CARD-001 card payment declined?",
        max_chars=2000,
    )
    if llm_select_tool:
        st.caption("Include a reference such as TX-CARD-001, TX-TRF-001, or CARD-001 to check its status.")
    elif not use_llm:
        st.caption("Include a reference such as TX-CARD-001 to check its status.")
    if not st.button("Get support", type="primary"):
        return

    try:
        retriever = load_retriever()
        result = SupportCopilot(classifier, retriever, load_demo_tools()).respond(
            question,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            llm_select_tool=llm_select_tool,
            llm_tool_model=llm_tool_model,
        )
    except (ImportError, ValueError, RuntimeError, OSError) as exc:
        st.error(f"Could not process this request: {exc}")
        return
    except Exception as exc:
        if use_llm:
            st.error(f"LLM request failed ({type(exc).__name__}): {exc}")
            return
        raise

    st.subheader("Issue identified")
    st.write(result.intent.replace("_", " ").capitalize())
    with st.expander("Model details"):
        st.write(f"Intent label: `{result.intent}`")
        st.write(f"Model confidence: {result.confidence:.1%} (uncalibrated)")

    st.subheader("Support response")
    support_answer = (
        result.generated_answer
        if result.generation_status == "accepted" and result.generated_answer
        else result.answer
    )
    if result.tool_result:
        lookup_text = result.tool_result.render()
        if support_answer.startswith(lookup_text):
            support_answer = support_answer[len(lookup_text):].lstrip()
    st.write(support_answer)
    status_notices = {
        "generation_fallback": "The generated reply was unavailable. Showing guidance from the help articles.",
        "grounding_fallback": "The generated reply could not be verified. Showing guidance from the help articles.",
        "low_confidence": "I could not confidently identify the issue. Please contact support for help.",
    }
    if result.status in status_notices:
        st.info(status_notices[result.status])
    if result.grounding_issues:
        with st.expander("Response diagnostics"):
            st.caption("; ".join(result.grounding_issues))
    if llm_select_tool and result.tool_result is None:
        st.info("No status check was performed. Include a supported reference in the question to trigger it.")

    if result.tool_result:
        st.subheader("Account status check")
        lookup_type = "Card status" if result.tool_result.name == "lookup_demo_card" else "Transaction status"
        route_label = {
            "nvidia_nim_function_call": "Selected using NVIDIA function calling",
            "intent_router": "Selected using the intent classifier",
        }.get(result.tool_routing, "")
        st.caption(f"{lookup_type} · {route_label}" if route_label else lookup_type)
        st.write(result.tool_result.render())

    if result.retrieved:
        st.subheader("Relevant help articles")
        for hit in result.retrieved:
            with st.expander(f"[{hit.faq.id}] {hit.faq.title}"):
                st.write(hit.faq.content)
                st.caption(hit.faq.source)


if __name__ == "__main__":
    main()
