"""Streamlit conversation workspace for the synthetic fintech support copilot."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from fintech_support.api_client import SupportAPIClient
from fintech_support.chat_ui import render_chat


def get_setting(name: str, default: str | None = None) -> str | None:
    """Read optional settings from Streamlit Secrets or the process environment."""
    try:
        value = st.secrets.get(name)
    except st.errors.StreamlitSecretNotFoundError:
        value = None
    return value or os.environ.get(name, default)


def apply_styles() -> None:
    css = (Path(__file__).resolve().parent / "fintech_support/chat.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(
        page_title="Fintech Support Copilot",
        layout="wide",
    )
    apply_styles()
    api_url = get_setting("SUPPORT_API_URL")
    remote_client = SupportAPIClient(api_url) if api_url else None
    render_chat(remote_client)


if __name__ == "__main__":
    main()
