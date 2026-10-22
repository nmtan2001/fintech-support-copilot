"""Run the local demonstration using streamlit run app.py."""

import importlib.util
import os
from pathlib import Path

import streamlit as st

from fintech_support.copilot import DEFAULT_HF_MODEL_ID, IntentClassifier, SupportCopilot
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
def load_retriever(backend: str):
    return FAQRetriever(load_faqs(ROOT / "data/faq/demo_policies.json"), backend=backend)


@st.cache_resource(show_spinner=False)
def load_demo_tools():
    database_path = ROOT / "outputs/demo/demo.sqlite"
    initialize_demo_database(database_path, ROOT / "data/transactions/demo_records.json")
    return DemoBankingTools(database_path)


def main():
    st.set_page_config(page_title="Fintech Support Copilot", layout="centered")
    st.title("Fintech Customer Support Copilot")
    st.warning("Academic prototype with synthetic policies and records. No access to real bank accounts or transactions.")

    transformer_downloaded = (ROOT / "outputs/distilbert/model/config.json").is_file()
    model_id = get_setting("HF_MODEL_ID", DEFAULT_HF_MODEL_ID)
    hf_token = get_setting("HF_TOKEN") or get_setting("HUGGINGFACE_HUB_TOKEN")
    transformer_configured = transformer_downloaded or bool(model_id)
    missing = [name for name in ("torch", "transformers") if importlib.util.find_spec(name) is None]
    transformer_available = transformer_configured and not missing
    choices = []
    if transformer_available:
        choices.append("DistilBERT")
    if (ROOT / "outputs/baseline/model.joblib").is_file():
        choices.append("TF-IDF + Logistic Regression")

    if transformer_configured and missing:
        st.info(
            "Local DistilBERT inference requires: "
            + ", ".join(missing)
            + ". Install the CPU inference dependencies or select the baseline."
        )
    if not transformer_downloaded and not model_id:
        st.info("Set HF_MODEL_ID in Streamlit Secrets to load the trained DistilBERT from Hugging Face.")
    if not choices:
        st.error("No usable classifier. Download a model and install its dependencies, or run baseline.py.")
        st.stop()

    with st.sidebar:
        model_choice = st.selectbox("Intent classifier", choices)

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

    st.caption(f"Intent classifier: {model_name} | BANKING77, 77 intents")

    with st.sidebar:
        st.subheader("Demo configuration")
        retrieval_backends = ["tfidf"]
        if importlib.util.find_spec("sentence_transformers") is not None:
            retrieval_backends.append("semantic")
        retrieval_backend = st.selectbox("FAQ retrieval", retrieval_backends)
        min_confidence = st.slider("Minimum classifier confidence (demo threshold)", 0.0, 1.0, 0.0, 0.05)
        st.caption("Classifier probabilities are not calibrated. Set 0 to disable threshold-based abstention.")
        nvidia_api_key = get_setting("NVIDIA_API_KEY")
        has_key = bool(nvidia_api_key)
        use_llm = st.checkbox("Generate answer with an LLM", value=has_key, disabled=not has_key)
        if use_llm:
            llm_model = st.text_input(
                "NVIDIA model ID",
                value=get_setting("NVIDIA_MODEL", ""),
                placeholder="Enter a model enabled for your NVIDIA API key",
            )
            llm_api_key = nvidia_api_key
        else:
            llm_api_key = None
            llm_model = None
        if use_llm:
            st.caption("The question and retrieved demo policies are sent to NVIDIA.")
        else:
            st.caption("Set NVIDIA_API_KEY to enable LLM generation.")
        llm_select_tool = st.checkbox(
            "Use NVIDIA function calling for synthetic lookups",
            value=False,
            disabled=not use_llm,
            help="Requires a synthetic reference such as TX-CARD-001. The selected NVIDIA model must support tool calling.",
        )

    question = st.text_area(
        "Customer question",
        value="My card payment was declined. How can I fix it?",
        max_chars=2000,
    )
    record_reference = st.text_input(
        "Synthetic record reference (optional)",
        placeholder="TX-CARD-001, TX-TRF-001 or CARD-001",
        help="Read-only SQLite lookup for three supported workflows. You can also include the demo ID in the question.",
    )
    if llm_select_tool:
        st.caption("To trigger NVIDIA tool calling, enter a demo reference or include it in the question (for example, TX-CARD-001).")
    if not st.button("Get support", type="primary"):
        return

    try:
        retriever = load_retriever(retrieval_backend)
        result = SupportCopilot(classifier, retriever, load_demo_tools()).respond(
            question,
            min_confidence=min_confidence,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            record_reference=record_reference or None,
            llm_select_tool=llm_select_tool,
        )
    except (ImportError, ValueError, RuntimeError, OSError) as exc:
        st.error(f"Could not process this request: {exc}")
        return
    except Exception as exc:
        if use_llm:
            st.error(f"LLM request failed ({type(exc).__name__}): {exc}")
            return
        raise

    st.subheader("Classification")
    st.write(f"Intent: `{result.intent}`")
    st.write(f"Model confidence: {result.confidence:.1%} (uncalibrated)")
    st.subheader("Suggested response")
    st.write(result.answer)
    if result.status != "answered":
        st.info(f"Fallback reason: {result.status}")
    if result.grounding_issues:
        st.caption("NVIDIA generation details: " + "; ".join(result.grounding_issues))
    if llm_select_tool and result.tool_result is None:
        st.info("No synthetic lookup was performed. Add a matching demo reference to trigger NVIDIA tool calling.")

    if result.tool_result:
        st.subheader("Read-only demo tool call")
        st.write(f"Tool: `{result.tool_result.name}`")
        st.caption(f"Routing: {result.tool_routing}")
        st.write(result.tool_result.render())

    if result.retrieved:
        st.subheader("Retrieved demo FAQ sources")
        for hit in result.retrieved:
            with st.expander(f"[{hit.faq.id}] {hit.faq.title} (similarity {hit.score:.2f})"):
                st.write(hit.faq.content)
                st.caption(hit.faq.source)


if __name__ == "__main__":
    main()
