"""Streamlit demo for the synthetic fintech support copilot."""

from __future__ import annotations

import html
import importlib.util
import os
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
import streamlit as st

from fintech_support.copilot import (
    DEFAULT_HF_MODEL_ID,
    DEFAULT_NVIDIA_MODEL,
    DEFAULT_NVIDIA_TOOL_MODEL,
    IntentClassifier,
    SupportCopilot,
    SupportResult,
)
from fintech_support.demo_tools import DemoBankingTools, initialize_demo_database
from fintech_support.retrieval import FAQRetriever, load_faqs
from evaluate_retrieval import evaluate_cases as evaluate_retrieval_cases


ROOT = Path(__file__).resolve().parent
LOCAL_MODE = "Local model · evidence extract"
NIM_ANSWER_MODE = "Local route · NIM answer"
NIM_TOOL_MODE = "NIM function call · NIM answer"

SCENARIOS = {
    "Card payment declined · TX-CARD-001": {
        "intent": "declined_card_payment",
        "prompt": "Why was my TX-CARD-001 card payment declined?",
        "description": "Classify the issue, retrieve the card guidance, and check a synthetic transaction.",
    },
    "Transfer status · TX-TRF-001": {
        "intent": "pending_transfer",
        "prompt": "My transfer TX-TRF-001 is still pending. What should I know?",
        "description": "Retrieve transfer guidance and inspect the read-only transaction fixture.",
    },
    "Lost card · CARD-001": {
        "intent": "lost_or_stolen_card",
        "prompt": "I lost my card. Can you check the status of CARD-001?",
        "description": "Route a supported card-status request to the synthetic card fixture.",
    },
    "General support · no record": {
        "intent": "cash_withdrawal",
        "prompt": "I need help understanding a cash withdrawal on my card.",
        "description": "Show the safe fallback when no authored help article matches the request.",
    },
}

MODEL_METRICS = {
    "TF-IDF + Logistic Regression": {"accuracy": 85.49, "macro_f1": 85.36},
    "DistilBERT": {"accuracy": 92.44, "macro_f1": 92.44},
}
RETRIEVAL_LABELS = {
    "tfidf": "TF-IDF lexical ranking",
    "semantic": "MiniLM semantic ranking",
    "hybrid": "TF-IDF + MiniLM reciprocal-rank fusion",
}


def get_setting(name: str, default: str | None = None) -> str | None:
    """Read optional settings from Streamlit Secrets or the process environment."""
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
) -> tuple[Any, str]:
    """Load and cache the selected classifier for this app process."""
    return IntentClassifier.from_artifacts(
        transformer_dir=ROOT / "outputs/distilbert/model",
        baseline_path=ROOT / "outputs/baseline/model.joblib",
        prefer_transformer=model_choice == "DistilBERT",
        model_id=model_id,
        token=model_token,
        revision=model_revision,
    )


@st.cache_resource(show_spinner=False)
def load_retriever(backend: str) -> FAQRetriever:
    """Build and cache the selected local retrieval index."""
    return FAQRetriever(load_faqs(ROOT / "data/faq/demo_policies.json"), backend=backend)


@st.cache_data(show_spinner=False)
def retrieval_benchmark(backend: str) -> dict[str, Any]:
    """Score the selected retrieval method on the authored FAQ evaluation set."""
    import json

    cases = json.loads((ROOT / "data/evaluation/faq_cases.json").read_text(encoding="utf-8"))
    return evaluate_retrieval_cases(load_retriever(backend), cases)


@st.cache_resource(show_spinner=False)
def load_demo_tools() -> DemoBankingTools:
    """Initialize and cache the read-only synthetic record lookup."""
    database_path = Path(get_setting("DEMO_DATABASE_PATH", str(ROOT / "outputs/demo/demo.sqlite")))
    initialize_demo_database(database_path, ROOT / "data/transactions/demo_records.json")
    return DemoBankingTools(database_path)


def load_scenario() -> None:
    """Put the selected example into the message composer."""
    scenario = SCENARIOS[st.session_state.scenario]
    st.session_state.question = scenario["prompt"]
    st.session_state.pop("last_result", None)
    st.session_state.pop("last_error", None)


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600;700&family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&display=swap');

        :root {
            --paper: #f3f0e8;
            --paper-deep: #e9e4d8;
            --ink: #1d302b;
            --muted: #68766e;
            --line: #d9d5ca;
            --forest: #173630;
            --forest-soft: #254a40;
            --leaf: #477b63;
            --copper: #d77754;
            --gold: #d9ae6c;
            --white: #fffefa;
            --danger: #a8493d;
        }

        html, body, [class*="css"] {
            font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif;
            color: var(--ink);
        }
        .stApp, [data-testid="stAppViewContainer"] {
            background: var(--paper);
        }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stToolbar"] { right: 1rem; }
        [data-testid="stMainBlockContainer"] {
            max-width: 1500px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }
        [data-testid="stSidebar"] {
            background: var(--forest);
            border-right: 1px solid rgba(255,255,255,.08);
        }
        [data-testid="stSidebar"] * { color: #edf2eb; }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color: #b7c9bd; }
        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-baseweb="input"] > div {
            background: #203f37;
            border-color: #3d6255;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] *,
        [data-testid="stSidebar"] input { color: #f5f4ec !important; }
        [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.16); }

        h1, h2, h3 { color: var(--ink); letter-spacing: -.035em; }
        h1, .hero-title { font-family: 'Newsreader', Georgia, serif; font-weight: 500; }
        h2, h3 { font-weight: 600; }
        p, li { line-height: 1.55; }
        code, .mono { font-family: 'DM Mono', monospace; }

        .brand-lockup { display: flex; align-items: center; gap: .75rem; padding: .4rem 0 1.1rem; }
        .brand-mark {
            display: grid; place-items: center; width: 2.5rem; height: 2.5rem;
            border: 1px solid rgba(231,185,121,.75); border-radius: 9px;
            color: #edbd7f; font-family: 'Newsreader', Georgia, serif;
            font-size: 1.4rem; font-style: italic;
        }
        .brand-name { color: #fffaf0; font-size: 1rem; font-weight: 700; letter-spacing: .12em; }
        .brand-sub { color: #adc0b4; font-family: 'DM Mono', monospace; font-size: .62rem; letter-spacing: .11em; margin-top: .12rem; }
        .sidebar-section {
            color: #d4aa70; font-family: 'DM Mono', monospace; font-size: .65rem;
            letter-spacing: .14em; margin: 1rem 0 .5rem; text-transform: uppercase;
        }
        .runtime-card {
            background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.13);
            border-radius: 12px; padding: .8rem .9rem; margin: .6rem 0 1rem;
        }
        .runtime-row { display: flex; justify-content: space-between; gap: .5rem; padding: .24rem 0; }
        .runtime-label { color: #adc0b4; font-family: 'DM Mono', monospace; font-size: .68rem; }
        .runtime-value { color: #fffaf0; font-size: .76rem; font-weight: 600; text-align: right; }
        .sidebar-note {
            border-left: 2px solid var(--gold); color: #d9e2da;
            font-size: .78rem; line-height: 1.5; margin-top: .8rem; padding: .2rem 0 .2rem .75rem;
        }

        .hero-panel {
            align-items: center; background:
                radial-gradient(circle at 92% 12%, rgba(215,119,84,.23), transparent 25%),
                linear-gradient(125deg, #173630 0%, #21493e 100%);
            border-radius: 20px; color: #f8f3e9; display: flex; justify-content: space-between;
            min-height: 265px; overflow: hidden; padding: 2rem 2.5rem; position: relative;
        }
        .hero-panel:after {
            border: 1px solid rgba(232,213,173,.16); border-radius: 50%;
            content: ''; height: 330px; position: absolute; right: 6%; top: -130px; width: 330px;
        }
        .hero-panel:before {
            border: 1px solid rgba(232,213,173,.12); border-radius: 50%;
            content: ''; height: 430px; position: absolute; right: 1%; top: -180px; width: 430px;
        }
        .hero-copy { max-width: 790px; position: relative; z-index: 1; }
        .hero-kicker {
            color: #e8b47f; font-family: 'DM Mono', monospace; font-size: .68rem;
            letter-spacing: .17em; margin-bottom: .8rem; text-transform: uppercase;
        }
        .hero-title {
            color: #fffaf0; font-size: clamp(2.2rem, 4vw, 3.8rem); line-height: .98;
            margin: 0; max-width: 760px;
        }
        .hero-title em { color: #e6b17b; font-style: italic; }
        .hero-copy p { color: #d3e0d5; font-size: .97rem; margin: 1rem 0 0; max-width: 670px; }
        .hero-stamp {
            align-items: center; background: rgba(255,250,240,.08); border: 1px solid rgba(255,250,240,.25);
            border-radius: 50%; color: #fffaf0; display: flex; flex-direction: column; height: 132px;
            justify-content: center; margin-right: 1.3rem; position: relative; text-align: center;
            transform: rotate(8deg); width: 132px; z-index: 1;
        }
        .hero-stamp strong { color: #eab77f; font-family: 'Newsreader', Georgia, serif; font-size: 2rem; font-weight: 500; }
        .hero-stamp span { font-family: 'DM Mono', monospace; font-size: .58rem; letter-spacing: .12em; line-height: 1.45; text-transform: uppercase; }
        .eyebrow { color: var(--leaf); font-family: 'DM Mono', monospace; font-size: .65rem; letter-spacing: .13em; text-transform: uppercase; }
        .section-intro { color: var(--muted); font-size: .9rem; margin-top: -.4rem; }
        .trust-strip {
            background: #e7e4d9; border: 1px solid var(--line); border-radius: 10px;
            color: #44534c; font-size: .77rem; margin: .8rem 0 1.4rem; padding: .7rem 1rem;
        }
        .trust-strip strong { color: var(--forest); }
        div[data-testid="stMetric"] {
            background: var(--white); border: 1px solid var(--line); border-radius: 13px;
            padding: .85rem 1rem; box-shadow: 0 3px 12px rgba(30,45,38,.035);
        }
        div[data-testid="stMetricLabel"] p { color: var(--muted); font-family: 'DM Mono', monospace; font-size: .66rem; letter-spacing: .06em; text-transform: uppercase; }
        div[data-testid="stMetricValue"] { color: var(--forest); font-family: 'Newsreader', Georgia, serif; }
        .stTabs [data-baseweb="tab-list"] { gap: 1.3rem; border-bottom: 1px solid var(--line); }
        .stTabs [data-baseweb="tab"] { color: var(--muted); font-weight: 600; }
        .stTabs [aria-selected="true"] { color: var(--forest) !important; }
        .stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
            background: var(--copper); border: 1px solid var(--copper); border-radius: 9px;
            color: white; font-weight: 700; min-height: 2.7rem;
        }
        .stButton > button[kind="primary"]:hover, .stFormSubmitButton > button[kind="primary"]:hover {
            background: #bd6044; border-color: #bd6044; color: white;
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255,254,250,.72); border-color: var(--line) !important; border-radius: 15px;
        }
        .trace-card {
            background: var(--white); border: 1px solid var(--line); border-radius: 14px;
            height: 150px; padding: 1rem; position: relative;
        }
        .trace-card:before { background: var(--leaf); border-radius: 5px; content: ''; height: 3px; left: 1rem; position: absolute; right: 1rem; top: 0; }
        .trace-card.review:before { background: var(--copper); }
        .trace-index { color: #a0a69c; font-family: 'DM Mono', monospace; font-size: .62rem; letter-spacing: .12em; }
        .trace-title { color: var(--forest); font-size: .85rem; font-weight: 700; margin: .5rem 0 .3rem; }
        .trace-state { color: var(--leaf); font-family: 'DM Mono', monospace; font-size: .62rem; letter-spacing: .08em; text-transform: uppercase; }
        .trace-card.review .trace-state { color: var(--copper); }
        .trace-detail { color: var(--muted); font-size: .74rem; line-height: 1.4; margin-top: .42rem; }
        .response-heading { align-items: center; display: flex; gap: .6rem; }
        .response-dot { background: var(--leaf); border-radius: 50%; display: inline-block; height: .55rem; width: .55rem; }
        .response-dot.review { background: var(--copper); }
        .status-pill {
            background: #e4eee6; border: 1px solid #c7dccb; border-radius: 99px;
            color: #2e684d; display: inline-block; font-family: 'DM Mono', monospace;
            font-size: .64rem; letter-spacing: .06em; padding: .25rem .6rem; text-transform: uppercase;
        }
        .status-pill.review { background: #f5e8dc; border-color: #e5c7ae; color: #9b5839; }
        .answer-card { background: #fffefa; border-left: 3px solid var(--copper); border-radius: 0 12px 12px 0; padding: 1rem 1.2rem; white-space: pre-wrap; }
        .request-quote { background: #eef0e8; border-left: 2px solid var(--forest); border-radius: 0 8px 8px 0; color: var(--ink); font-size: .88rem; line-height: 1.55; margin: .65rem 0 1rem; padding: .7rem .9rem; white-space: pre-wrap; }
        .answer-label { color: var(--muted); font-family: 'DM Mono', monospace; font-size: .64rem; letter-spacing: .1em; text-transform: uppercase; }
        .data-disclaimer { border-top: 1px solid rgba(255,255,255,.14); color: #b7c9bd; font-size: .68rem; line-height: 1.5; margin-top: 1.3rem; padding-top: .9rem; }
        @media (max-width: 850px) {
            .hero-panel { align-items: flex-start; padding: 1.5rem; }
            .hero-stamp { display: none; }
            [data-testid="stMainBlockContainer"] { padding-left: 1rem; padding-right: 1rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def stage_card(index: int, title: str, state: str, detail: str, *, review: bool = False) -> None:
    """Render one fixed, escaped node in the request trace."""
    css_class = "trace-card review" if review else "trace-card"
    safe_title = html.escape(title)
    safe_state = html.escape(state)
    safe_detail = html.escape(detail)
    st.markdown(
        f"""
        <div class="{css_class}">
            <div class="trace-index">STEP {index:02d}</div>
            <div class="trace-title">{safe_title}</div>
            <div class="trace-state">{safe_state}</div>
            <div class="trace-detail">{safe_detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pipeline_stages(
    result: SupportResult | None,
    model_name: str,
    retrieval_backend: str,
) -> list[dict[str, Any]]:
    """Describe the real stages used by the current implementation."""
    retrieval_description = RETRIEVAL_LABELS[retrieval_backend]
    if result is None:
        return [
            {"title": "Intent classifier", "state": "Ready", "detail": model_name, "review": False},
            {
                "title": "Evidence retrieval",
                "state": "Ready",
                "detail": f"{retrieval_description} · intent-scoped over 9 authored articles",
                "review": False,
            },
            {"title": "Read-only tool", "state": "Ready", "detail": "30 synthetic card and transfer records", "review": False},
            {"title": "Grounding guard", "state": "Ready", "detail": "Exact FAQ sentence checks with safe fallback", "review": False},
        ]

    intent = result.intent.replace("_", " ")
    stages = [
        {
            "title": "Intent classifier",
            "state": "Complete",
            "detail": f"{model_name} · {intent} · {result.confidence:.1%} uncalibrated confidence",
            "review": False,
        },
        {
            "title": "Evidence retrieval",
            "state": "Complete" if result.retrieved else "No evidence",
            "detail": (
                f"{len(result.retrieved)} article(s) ranked by {retrieval_description}"
                if result.retrieved else "No relevant help article passed the retrieval step"
            ),
            "review": not bool(result.retrieved),
        },
    ]

    if result.tool_result:
        tool_state = "Record found" if result.tool_result.found else "No matching fixture"
        tool_detail = f"{result.tool_routing or 'read-only route'} · {result.tool_result.reference}"
        tool_review = not result.tool_result.found
    else:
        tool_state = "Not invoked"
        tool_detail = "No supported reference was present in this request"
        tool_review = False
    stages.append({"title": "Read-only tool", "state": tool_state, "detail": tool_detail, "review": tool_review})

    generation_labels = {
        "accepted": ("Grounded", "NVIDIA response passed exact-sentence citation checks", False),
        "disabled": ("Extractive fallback", "Local FAQ sentence returned without external generation", False),
        "skipped_no_record": ("Generation skipped", "No matching fixture; generated status claims were withheld", True),
        "rejected": ("Blocked", "Grounding check failed; the local extractive fallback was used", True),
        "unavailable": ("Fallback", "NVIDIA was unavailable; the local extractive fallback was used", True),
    }
    state, detail, review = generation_labels.get(
        result.generation_status,
        ("Safe response", result.status.replace("_", " "), False),
    )
    stages.append({"title": "Grounding guard", "state": state, "detail": detail, "review": review})
    return stages


def render_pipeline(result: SupportResult | None, model_name: str, retrieval_backend: str) -> None:
    stages = pipeline_stages(result, model_name, retrieval_backend)
    columns = st.columns(4, gap="small")
    for index, (column, stage) in enumerate(zip(columns, stages), start=1):
        with column:
            stage_card(index, **stage)


def render_support_result(
    result: SupportResult,
    model_name: str,
    elapsed_ms: float,
    question: str,
    retrieval_backend: str,
) -> None:
    st.markdown("<div class='eyebrow'>LATEST ANALYSIS</div>", unsafe_allow_html=True)
    heading_col, intent_col, confidence_col, latency_col = st.columns([2, 1, 1, 1])
    with heading_col:
        st.markdown("### Support recommendation")
        st.caption("Review the answer, evidence, and any synthetic lookup before presenting the case.")
    with intent_col:
        st.metric("Predicted intent", result.intent.replace("_", " ").title())
    with confidence_col:
        st.metric(
            "Confidence",
            f"{result.confidence:.1%}",
            help="Model score only. Confidence is not calibrated and does not measure factual correctness.",
        )
    with latency_col:
        st.metric("Processing time", f"{elapsed_ms:.0f} ms")

    st.markdown(
        f"<div class='request-quote'>{html.escape(question)}</div>",
        unsafe_allow_html=True,
    )

    response_text = (
        result.generated_answer
        if result.generation_status == "accepted" and result.generated_answer
        else result.answer
    )
    if result.tool_result:
        lookup_text = result.tool_result.render()
        if response_text.startswith(lookup_text):
            response_text = response_text[len(lookup_text):].lstrip()

    left, right = st.columns([1.45, 1], gap="large")
    with left:
        with st.container(border=True):
            status_label = {
                "answered": "Response ready",
                "low_confidence": "Confidence below threshold",
                "unsupported_intent": "Human support recommended",
                "tool_only": "Fixture result only",
                "grounding_fallback": "Safe fallback used",
                "generation_fallback": "Safe fallback used",
            }.get(result.status, result.status.replace("_", " ").title())
            status_class = "review" if result.status != "answered" else ""
            st.markdown(
                f"<div class='response-heading'><span class='response-dot {'review' if status_class else ''}'></span>"
                f"<span class='status-pill {status_class}'>{html.escape(status_label)}</span></div>",
                unsafe_allow_html=True,
            )
            st.markdown("<div class='answer-label'>Customer-facing response</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='answer-card'>{html.escape(response_text)}</div>", unsafe_allow_html=True)
            st.caption(f"Classifier: {model_name} · confidence {result.confidence:.1%} (uncalibrated)")
            if result.status in {"generation_fallback", "grounding_fallback"}:
                st.warning("The generated answer was not used. The app returned its local evidence-based fallback.")
            elif result.generation_status == "skipped_no_record":
                st.info("No matching synthetic record was found, so generated status claims were withheld.")

    with right:
        with st.container(border=True):
            st.markdown("#### Synthetic record check")
            st.caption("Read-only fixture data · no real accounts")
            if result.tool_result:
                lookup_state = "Fixture match" if result.tool_result.found else "No matching fixture"
                st.markdown(f"**{lookup_state}** · `{result.tool_result.reference}`")
                st.write(result.tool_result.render())
                st.caption(f"Route: {result.tool_routing or 'intent router'} · result is read-only")
            else:
                st.markdown("**No lookup performed**")
                st.write("A supported synthetic reference was not present or the classified intent does not allow a lookup.")

    if result.retrieved:
        st.markdown("#### Evidence used")
        if retrieval_backend == "hybrid":
            st.caption(
                "Hybrid score is reciprocal-rank fusion (RRF), not a probability. "
                "The candidate gate accepts lexical matches or semantic cosine ≥ 0.35."
            )
        else:
            score_name = "TF-IDF" if retrieval_backend == "tfidf" else "MiniLM semantic"
            st.caption(f"Similarity is the {score_name} cosine score, not a probability or a factuality score.")
        for hit in result.retrieved:
            score_label = "RRF" if retrieval_backend == "hybrid" else "cosine similarity"
            with st.expander(f"{hit.faq.title}  ·  {score_label} {hit.score:.3f}"):
                st.write(hit.faq.content)
                st.caption(f"{hit.faq.id} · {hit.faq.source}")
                if retrieval_backend == "hybrid":
                    st.caption(
                        f"TF-IDF cosine {hit.tfidf_score:.3f} · "
                        f"MiniLM cosine {hit.semantic_score:.3f}"
                    )
    elif result.grounding_issues:
        st.warning("No FAQ evidence was retrieved for this request.")


def render_scorecard(selected_model: str, retrieval_backend: str) -> None:
    st.markdown("<div class='eyebrow'>MODEL & EVALUATION</div>", unsafe_allow_html=True)
    st.markdown("## Compare the intent models")
    st.markdown(
        "The benchmark numbers use the untouched official BANKING77 test split. "
        "The separate support workflow and live NVIDIA response are not measured by this table."
    )

    selected_metrics = MODEL_METRICS.get(selected_model, MODEL_METRICS["DistilBERT"])
    metric_cols = st.columns(3)
    metric_cols[0].metric("Selected model", selected_model)
    metric_cols[1].metric("Test accuracy", f"{selected_metrics['accuracy']:.2f}%")
    metric_cols[2].metric("Test macro-F1", f"{selected_metrics['macro_f1']:.2f}%")

    chart_col, detail_col = st.columns([1.3, 1], gap="large")
    with chart_col:
        st.markdown("#### Official test performance")
        chart = pd.DataFrame(
            {
                "Accuracy (%)": [85.49, 92.44],
                "Macro-F1 (%)": [85.36, 92.44],
            },
            index=["TF-IDF + Logistic Regression", "Fine-tuned DistilBERT"],
        )
        st.bar_chart(chart, height=270)
    with detail_col:
        st.markdown("#### Stack evaluation")
        try:
            with st.spinner("Scoring this retriever on the authored FAQ cases…"):
                retrieval_results = retrieval_benchmark(retrieval_backend)
        except Exception as exc:
            retrieval_results = None
            st.warning(f"Could not score the selected retriever: {exc}")
        if retrieval_results:
            retrieval_row = st.columns(2)
            retrieval_row[0].metric(
                "FAQ Recall@1",
                f"{retrieval_results['recall_at_1']:.1%}",
                help=f"{retrieval_results['relevant_count']} relevant queries · {RETRIEVAL_LABELS[retrieval_backend]}",
            )
            retrieval_row[1].metric(
                "FAQ Recall@3",
                f"{retrieval_results['recall_at_3']:.1%}",
                help=f"{retrieval_results['relevant_count']} relevant queries · {RETRIEVAL_LABELS[retrieval_backend]}",
            )
            evaluation_row = st.columns(2)
            abstention = retrieval_results["negative_abstention_rate"]
            evaluation_row[0].metric(
                "No-match abstain",
                f"{abstention:.1%}" if abstention is not None else "N/A",
                help=f"{retrieval_results['negative_count']} authored out-of-scope queries",
            )
            evaluation_row[1].metric(
                "Workflow cases",
                "37 / 48",
                help="Deterministic baseline-classifier checks",
            )
        st.markdown(
            f"**Active retriever:** {RETRIEVAL_LABELS[retrieval_backend]} across a nine-article authored corpus.  "
            "**Fixture lookups** use 30 synthetic records and are read-only."
        )
        st.info(
            "The FAQ evaluation uses 22 relevant queries and four out-of-scope queries. "
            "The 37/48 workflow result is a prototype measure, not production customer-support quality. "
            "The optional live LLM path has no benchmark score in this project."
        )
        if retrieval_backend == "hybrid":
            st.caption(
                "The hybrid semantic floor (0.35) is an initial heuristic. "
                "Four out-of-scope examples are too few to calibrate a production relevance threshold."
            )


def main() -> None:
    st.set_page_config(
        page_title="Fintech Support Copilot · Service Desk",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_styles()

    model_id = get_setting("HF_MODEL_ID", DEFAULT_HF_MODEL_ID)
    hf_token = get_setting("HF_TOKEN") or get_setting("HUGGINGFACE_HUB_TOKEN")
    model_revision = get_setting("HF_MODEL_REVISION")
    transformer_downloaded = (ROOT / "outputs/distilbert/model/config.json").is_file()
    transformer_installed = bool(
        importlib.util.find_spec("transformers") and importlib.util.find_spec("torch")
    )
    transformer_available = transformer_installed and (transformer_downloaded or bool(model_id))
    baseline_available = (ROOT / "outputs/baseline/model.joblib").is_file()

    model_options = []
    if transformer_available:
        model_options.append("DistilBERT")
    if baseline_available:
        model_options.append("TF-IDF + Logistic Regression")

    retrieval_options = {"TF-IDF · lexical search": "tfidf"}
    semantic_available = bool(importlib.util.find_spec("sentence_transformers"))
    if semantic_available:
        retrieval_options["MiniLM · semantic search"] = "semantic"
        retrieval_options["Hybrid · TF-IDF + MiniLM"] = "hybrid"

    st.session_state.setdefault("scenario", next(iter(SCENARIOS)))
    st.session_state.setdefault("question", SCENARIOS[st.session_state.scenario]["prompt"])
    st.session_state.setdefault("record_reference", "")
    st.session_state.setdefault("minimum_confidence", 0.0)

    nvidia_api_key = get_setting("NVIDIA_API_KEY")
    with st.sidebar:
        st.markdown(
            """
            <div class="brand-lockup">
                <div class="brand-mark">F</div>
                <div><div class="brand-name">FINTECH</div>
                <div class="brand-sub">SUPPORT INTELLIGENCE</div></div>
            </div>
            <div class="sidebar-section">Runtime controls</div>
            """,
            unsafe_allow_html=True,
        )

        if not model_options:
            st.error("No usable model is available. Install the inference dependencies or provide a local model artifact.")
            st.stop()

        selected_model = st.selectbox("Intent classifier", model_options, key="selected_model")
        performance = MODEL_METRICS.get(selected_model, MODEL_METRICS["DistilBERT"])
        st.caption(f"BANKING77 test accuracy · {performance['accuracy']:.2f}%")

        retrieval_label = st.selectbox(
            "FAQ retrieval strategy",
            list(retrieval_options),
            help="Hybrid uses intent-gated TF-IDF and MiniLM candidates, fused with reciprocal-rank fusion.",
        )
        retrieval_backend = retrieval_options[retrieval_label]
        if retrieval_backend in {"semantic", "hybrid"}:
            st.caption("MiniLM runs locally; its model files download from Hugging Face the first time it is used.")
        elif not semantic_available:
            st.caption("Install the optional semantic extra to compare MiniLM and hybrid retrieval.")

        inference_modes = [LOCAL_MODE, NIM_ANSWER_MODE, NIM_TOOL_MODE]
        inference_mode = st.selectbox(
            "Response path",
            inference_modes,
            index=0,
            disabled=not bool(nvidia_api_key),
            help="NVIDIA modes send the selected support request to NVIDIA NIM.",
        )
        if nvidia_api_key:
            if inference_mode == LOCAL_MODE:
                st.caption("Local classifier, local retrieval, and extractive response. No NIM request.")
            else:
                st.warning("This mode sends the request to NVIDIA NIM and may incur provider usage charges.")
        else:
            st.caption("NVIDIA NIM is optional. Add NVIDIA_API_KEY to Streamlit Secrets to enable it.")

        st.session_state.minimum_confidence = st.slider(
            "Minimum intent confidence",
            min_value=0.0,
            max_value=0.8,
            value=float(st.session_state.minimum_confidence),
            step=0.05,
            help="Below this threshold, the copilot returns a safe fallback instead of routing a lookup.",
        )

        model_revision_label = model_revision[:10] if model_revision else "model default"
        st.markdown(
            f"""
            <div class="runtime-card">
                <div class="runtime-row"><span class="runtime-label">FAQ retrieval</span><span class="runtime-value">{html.escape(RETRIEVAL_LABELS[retrieval_backend])}</span></div>
                <div class="runtime-row"><span class="runtime-label">Model revision</span><span class="runtime-value">{html.escape(model_revision_label)}</span></div>
                <div class="runtime-row"><span class="runtime-label">Data mode</span><span class="runtime-value">Synthetic only</span></div>
            </div>
            <div class="sidebar-section">Safety boundary</div>
            <div class="sidebar-note">The lookup tool is read-only. It cannot freeze cards, move money, or access a real bank account.</div>
            <div class="data-disclaimer">Demo fixture data only · 30 fictional records · no account connection</div>
            """,
            unsafe_allow_html=True,
        )

    try:
        with st.spinner(f"Loading {selected_model}…"):
            classifier, model_name = load_classifier(
                selected_model,
                model_id,
                hf_token,
                model_revision,
            )
    except (FileNotFoundError, ImportError, OSError, ValueError, RuntimeError) as exc:
        st.error(f"Could not load the selected classifier: {exc}")
        st.stop()

    st.markdown(
        """
        <section class="hero-panel">
            <div class="hero-copy">
                <div class="hero-kicker">AI support workbench · synthetic demo</div>
                <h1 class="hero-title">Support that shows<br><em>its work.</em></h1>
                <p>Follow one request from intent classification through evidence retrieval, a guarded read-only lookup, and a grounded response.</p>
            </div>
            <div class="hero-stamp"><strong>AI</strong><span>evidence<br>first</span></div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='trust-strip'><strong>Demo boundary:</strong> fictional help content and records only. "
        "No real accounts, live bank policies, payments, or card actions.</div>",
        unsafe_allow_html=True,
    )

    selected_metrics = MODEL_METRICS.get(model_name, MODEL_METRICS["DistilBERT"])
    metric_cols = st.columns(4)
    metric_cols[0].metric("Intent labels", "77", "BANKING77")
    metric_cols[1].metric("Selected model", f"{selected_metrics['accuracy']:.2f}%", "official test accuracy")
    metric_cols[2].metric("Help articles", "9", "authored demo corpus")
    metric_cols[3].metric("Synthetic records", "30", "read-only fixtures")

    workspace_tab, trace_tab, scorecard_tab = st.tabs(
        ["Support workspace", "AI stack trace", "Model scorecard"]
    )

    with workspace_tab:
        st.markdown("<div class='eyebrow'>01 / COMPOSE A CASE</div>", unsafe_allow_html=True)
        st.markdown("## Start with a customer message")
        st.markdown(
            "Choose a prepared scenario or write a request. The trace will show which model, evidence, "
            "tool route, and guardrail handled it."
        )

        st.selectbox(
            "Load a demo scenario",
            list(SCENARIOS),
            key="scenario",
            on_change=load_scenario,
        )
        st.caption(SCENARIOS[st.session_state.scenario]["description"])

        with st.form("support_request", border=False):
            st.text_area(
                "Customer message",
                key="question",
                height=115,
                max_chars=2000,
                placeholder="Describe the issue. Include a supported demo reference to request a fixture lookup.",
            )
            ref_col, submit_col = st.columns([1, 1])
            with ref_col:
                st.text_input(
                    "Optional synthetic reference",
                    key="record_reference",
                    placeholder="TX-CARD-001, TX-TRF-001, or CARD-001",
                )
            with submit_col:
                st.caption("References are validated against the classified intent before lookup.")
                submitted = st.form_submit_button(
                    "Run support analysis  →",
                    type="primary",
                    use_container_width=True,
                )

        if submitted:
            question = st.session_state.question.strip()
            if not question:
                st.session_state.last_error = "Enter a customer message before running the analysis."
                st.session_state.pop("last_result", None)
            else:
                use_nim = inference_mode != LOCAL_MODE and bool(nvidia_api_key)
                use_nim_tools = inference_mode == NIM_TOOL_MODE and bool(nvidia_api_key)
                try:
                    started = perf_counter()
                    with st.spinner("Classifying · retrieving evidence · checking the allowed tool route…"):
                        result = SupportCopilot(
                            classifier,
                            load_retriever(retrieval_backend),
                            load_demo_tools(),
                        ).respond(
                            question,
                            min_confidence=st.session_state.minimum_confidence,
                            llm_model=DEFAULT_NVIDIA_MODEL if use_nim else None,
                            llm_api_key=nvidia_api_key if use_nim else None,
                            record_reference=st.session_state.record_reference.strip() or None,
                            llm_select_tool=use_nim_tools,
                            llm_tool_model=DEFAULT_NVIDIA_TOOL_MODEL if use_nim_tools else None,
                        )
                    elapsed_ms = (perf_counter() - started) * 1000
                    st.session_state.last_result = result
                    st.session_state.last_question = question
                    st.session_state.last_model = model_name
                    st.session_state.last_retrieval_backend = retrieval_backend
                    st.session_state.last_elapsed_ms = elapsed_ms
                    st.session_state.pop("last_error", None)
                except (ImportError, ValueError, RuntimeError, OSError) as exc:
                    st.session_state.last_error = str(exc)
                    st.session_state.pop("last_result", None)
                except Exception:
                    st.session_state.last_error = "The request could not be completed. Review the application log for details."
                    st.session_state.pop("last_result", None)

        if st.session_state.get("last_error"):
            st.error(st.session_state.last_error)
        latest = st.session_state.get("last_result")
        if latest:
            render_support_result(
                latest,
                st.session_state.get("last_model", model_name),
                st.session_state.get("last_elapsed_ms", 0.0),
                st.session_state.get("last_question", ""),
                st.session_state.get("last_retrieval_backend", retrieval_backend),
            )
        else:
            st.markdown(
                "<div class='trust-strip'><strong>What you will see:</strong> predicted intent and confidence, "
                "retrieved help evidence with similarity scores, an optional synthetic lookup, and the final grounding outcome.</div>",
                unsafe_allow_html=True,
            )

    latest = st.session_state.get("last_result")
    latest_model = st.session_state.get("last_model", model_name)
    latest_retrieval_backend = st.session_state.get("last_retrieval_backend", retrieval_backend)
    with trace_tab:
        st.markdown("<div class='eyebrow'>02 / OBSERVABILITY</div>", unsafe_allow_html=True)
        st.markdown("## Follow the request through the stack")
        st.markdown(
            "This demo makes each AI boundary visible. A model prediction never authorizes an arbitrary tool call; "
            "the application checks the intent and reference before touching the synthetic fixture."
        )
        render_pipeline(latest, latest_model, latest_retrieval_backend)
        if latest:
            st.markdown("#### Latest request")
            st.code(st.session_state.get("last_question", ""), language=None)
            trace_data = {
                "classifier": latest_model,
                "retrieval_backend": latest_retrieval_backend,
                "hybrid_semantic_floor": (
                    FAQRetriever.HYBRID_SEMANTIC_FLOOR
                    if latest_retrieval_backend == "hybrid" else None
                ),
                "hybrid_fusion": "reciprocal_rank_fusion_k_60" if latest_retrieval_backend == "hybrid" else None,
                "intent": latest.intent,
                "confidence_uncalibrated": latest.confidence,
                "retrieved_faqs": [
                    {
                        "id": hit.faq.id,
                        "title": hit.faq.title,
                        "ranking_score": hit.score,
                        "tfidf_cosine_score": hit.tfidf_score,
                        "semantic_cosine_score": hit.semantic_score,
                    }
                    for hit in latest.retrieved
                ],
                "tool_route": latest.tool_routing,
                "tool_result": (
                    {
                        "reference": latest.tool_result.reference,
                        "found": latest.tool_result.found,
                        "status": latest.tool_result.status,
                        "read_only": True,
                        "synthetic_only": True,
                    }
                    if latest.tool_result else None
                ),
                "generation_status": latest.generation_status,
                "grounding_issues": list(latest.grounding_issues),
                "overall_status": latest.status,
                "elapsed_ms": round(st.session_state.get("last_elapsed_ms", 0.0), 1),
            }
            with st.expander("Inspect structured request trace"):
                st.json(trace_data)
        else:
            st.info("Run a scenario in the Support workspace to populate a live trace.")

        st.markdown("#### Why the boundaries matter")
        boundary_cols = st.columns(3)
        boundary_cols[0].markdown("**Intent-scoped retrieval**  \nOnly FAQ articles mapped to the predicted intent are ranked.")
        boundary_cols[1].markdown("**Allowlisted tool use**  \nReference format and intent are validated in Python; SQL uses a read-only connection.")
        boundary_cols[2].markdown("**Extractive fallback**  \nNIM output must match complete evidence sentences or the local answer is used.")

    with scorecard_tab:
        render_scorecard(model_name, retrieval_backend)

    st.markdown(
        "<div class='data-disclaimer' style='color:#68766e;border-color:#d9d5ca;margin-top:2rem'>"
        "FINTECH SUPPORT COPILOT · COURSE DEMONSTRATION · SYNTHETIC DATA ONLY</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
