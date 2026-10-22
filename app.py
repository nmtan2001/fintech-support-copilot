"""Run the local demonstration using streamlit run app.py."""

import importlib.util
import os
from pathlib import Path

import streamlit as st

from fintech_support.copilot import IntentClassifier, SupportCopilot
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


def main():
    st.set_page_config(page_title="Fintech Support Copilot", layout="centered")
    st.title("Fintech Customer Support Copilot")
    st.warning("Academic prototype using synthetic support policies. No access to real bank accounts or transactions.")

    transformer_downloaded = (ROOT / "outputs/distilbert/model/config.json").is_file()
    model_id = get_setting("HF_MODEL_ID")
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
            "DistilBERT is configured. Inference requires: "
            + ", ".join(missing)
            + ". Install the project's inference dependencies to enable it."
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
            get_setting("HUGGINGFACE_HUB_TOKEN"),
            get_setting("HF_MODEL_REVISION"),
        )
    except (FileNotFoundError, ImportError, OSError, ValueError) as exc:
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
        openai_api_key = get_setting("OPENAI_API_KEY")
        has_key = bool(openai_api_key)
        use_llm = st.checkbox("Generate answer with OpenAI", value=False, disabled=not has_key)
        llm_model = st.text_input("LLM model", value="gpt-4.1-mini") if use_llm else None
        if use_llm:
            st.caption("The question and retrieved demo policies are sent to the selected LLM API.")
        elif not has_key:
            st.caption("Set OPENAI_API_KEY to enable optional LLM generation. Source-only mode works offline.")

    question = st.text_area(
        "Customer question",
        value="My card payment was declined. How can I fix it?",
        max_chars=2000,
    )
    if not st.button("Get support", type="primary"):
        return

    try:
        retriever = load_retriever(retrieval_backend)
        result = SupportCopilot(classifier, retriever).respond(
            question,
            min_confidence=min_confidence,
            llm_model=llm_model,
            llm_api_key=openai_api_key if use_llm else None,
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

    if result.retrieved:
        st.subheader("Retrieved demo FAQ sources")
        for hit in result.retrieved:
            with st.expander(f"[{hit.faq.id}] {hit.faq.title} (similarity {hit.score:.2f})"):
                st.write(hit.faq.content)
                st.caption(hit.faq.source)


if __name__ == "__main__":
    main()
