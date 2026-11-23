"""Conversation workspace for the API-backed and standalone Streamlit demos."""

from pathlib import Path

import streamlit as st

from fintech_support.conversations import create_service

ROOT = Path(__file__).resolve().parents[1]


@st.cache_resource(show_spinner=False)
def local_conversations():
    return create_service(ROOT)


def render_reply(response):
    answer = response["answer"]
    for index, citation in enumerate(response["citations"], start=1):
        answer = answer.replace(f"[{citation['id']}]", f"[{index}]")
    st.markdown(answer)
    status = response["generation_status"]
    if status not in {"accepted", "deterministic"}:
        if response["tool_results"]:
            st.caption("Record facts verified. Additional AI guidance could not be verified, so it is omitted.")
        else:
            detail = "Here are the relevant handbook excerpts." if response["citations"] else "Try rephrasing your question."
            st.info(f"I couldn't verify a generated answer. {detail}")
    if response["citations"]:
        with st.expander(f"Sources · {len(response['citations'])}"):
            for index, citation in enumerate(response["citations"], start=1):
                st.markdown(f"**[{index}] {citation['title']}**")
                st.write(citation["text"])
                st.caption(citation["source"])
    if response["tool_results"]:
        with st.expander("Record details"):
            st.json(response["tool_results"])
    if response["handoff"]:
        st.caption("Queued for local demo review. No request was sent to a real bank.")
        with st.expander("Review request details"):
            st.json(response["handoff"])


def render_message(message, key):
    role = message["role"]
    with st.container(key=key):
        with st.chat_message(role, avatar=":material/support_agent:" if role == "assistant" else ":material/person:"):
            st.caption("FablePay assistant" if role == "assistant" else "You")
            if role == "assistant":
                render_reply(message["response"])
            else:
                st.markdown(message["content"])


def render_chat(remote_client=None):
    with st.sidebar:
        st.markdown("## FablePay")
        st.caption("Banking support assistant")
        if st.button("New conversation", icon=":material/edit_square:", width="stretch"):
            session = st.session_state.get("conversation")
            try:
                if session:
                    if remote_client:
                        remote_client.delete_conversation(session)
                    else:
                        local_conversations().delete(session["id"], session["token"])
            except RuntimeError:
                st.warning("The previous conversation could not be removed or has expired; it will expire automatically.")
            st.session_state.pop("conversation", None)
            st.session_state["chat_messages"] = []
            st.session_state.pop("chat_error", None)
            st.session_state.pop("chat_request_id", None)
    try:
        profiles = remote_client.demo_profiles() if remote_client else local_conversations().profiles()
    except (RuntimeError, ValueError, OSError) as exc:
        st.error(str(exc))
        return
    with st.sidebar:
        st.divider()
        st.markdown("### Demo customer")
        names = {row["id"]: row["name"] for row in profiles}
        customer_id = st.selectbox("Fictional customer profile", list(names), format_func=names.get,
                                   key="demo_customer_id", disabled="conversation" in st.session_state,
                                   label_visibility="collapsed")
        selected = next(row for row in profiles if row["id"] == customer_id)
        if "conversation" in st.session_state:
            st.caption("Start a new conversation to switch customers.")
        st.markdown("### Sample records")
        st.caption("Choose a record to ask about its status.")

        def record_label(reference):
            kind = "Card payment" if reference.startswith("TX-CARD-") else (
                "Bank transfer" if reference.startswith("TX-TRF-") else "Card")
            return f"{reference} · {kind}"

        reference = st.selectbox("Choose a sample record", selected["references"],
                                 format_func=record_label, key=f"sample_record_{customer_id}",
                                 label_visibility="collapsed")
        sample_message = None
        if st.button("Ask about this record", disabled=reference is None, width="stretch"):
            sample_message = f"What is the status of {reference}?"
        st.divider()

    st.markdown('<div class="chat-heading"><h1>Support assistant</h1><span>FablePay demo</span></div>',
                unsafe_allow_html=True)
    messages = st.session_state.get("chat_messages", [])
    if not messages:
        with st.container(key="chat_welcome"):
            st.markdown("## How can I help?")
            st.markdown("Ask about a payment, track a transfer, or get help with your card.")
        with st.container(key="starter_prompts"):
            suggestions = [
                ("Check a card payment", "TX-CARD-", "Why was my card payment {reference} declined?"),
                ("Track a transfer", "TX-TRF-", "What is the status of my transfer {reference}?"),
                ("Report a lost card", "CARD-", "I lost my card {reference}. What should I do?"),
            ]
            for column, (label, prefix, prompt) in zip(st.columns(3), suggestions):
                ref = next((ref for ref in selected["references"] if ref.startswith(prefix)), None)
                if column.button(label, disabled=ref is None, width="stretch"):
                    sample_message = prompt.format(reference=ref)
        st.caption("You can also browse sample records in the sidebar.")
    for index, message in enumerate(messages):
        render_message(message, key=f"{message['role']}_message_{index}")
    if st.session_state.get("chat_messages"):
        latest = st.session_state.chat_messages[-1]["response"]
        with st.sidebar, st.expander("Conversation details"):
            st.json({"execution": "kubernetes_api" if remote_client else "local_process",
                     "request_id": st.session_state.get("chat_request_id"),
                     "model": latest["model"], "prompt_version": latest["prompt_version"],
                     "knowledge_version": latest["knowledge_version"],
                     "generation_status": latest["generation_status"],
                     "fallback_reason": latest.get("fallback_reason"),
                     "verification_supported": latest.get("verification_supported"),
                     "rejected_claim_indexes": latest.get("rejected_claim_indexes", []),
                     "model_call_count": latest.get("model_call_count"),
                     "route": latest.get("route"), "case_state": latest.get("case_state"),
                     "customer_id": latest.get("customer_id"),
                     "citations": [row["id"] for row in latest["citations"]],
                     "read_only_lookups": [row["reference"] for row in latest["tool_results"]]})
    session = st.session_state.get("conversation")
    if session:
        try:
            cases = remote_client.conversation_cases(session) if remote_client else local_conversations().cases(
                session["id"], session["token"])
            with st.sidebar, st.expander("Review queue"):
                if cases:
                    for case in cases:
                        st.markdown(f"**{case['case_id']}** · {case['status'].replace('_', ' ')}")
                        st.json(case)
                else:
                    st.caption("No review requests yet. Demo requests expire with this conversation.")
        except RuntimeError as exc:
            st.warning(str(exc))
    if st.session_state.get("chat_error"):
        st.error(st.session_state.chat_error)
    # Explicit placement avoids app-level autoscroll when details change height.
    typed_message = st.bottom.chat_input("Message FablePay support", max_chars=2000)
    if message := sample_message or typed_message:
        render_message({"role": "user", "content": message}, key="user_message_pending")
        try:
            with st.spinner("Checking your question…"):
                session = st.session_state.get("conversation")
                if session is None:
                    session = remote_client.create_conversation(customer_id) if remote_client else local_conversations().create(customer_id)
                    st.session_state.conversation = session
                if remote_client:
                    response = remote_client.conversation_message(session, message)
                    request_id = remote_client.request_id
                else:
                    response = local_conversations().respond(session["id"], session["token"], message)
                    request_id = None
            st.session_state.setdefault("chat_messages", []).extend([
                {"role": "user", "content": message},
                {"role": "assistant", "response": response},
            ])
            st.session_state.chat_request_id = request_id
            st.session_state.pop("chat_error", None)
            st.rerun()
        except (RuntimeError, ValueError, OSError) as exc:
            st.session_state.chat_error = str(exc)
            st.rerun()
