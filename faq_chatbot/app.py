"""Streamlit front-end for the FAQ retrieval chatbot.

This module contains presentation logic only. All retrieval behaviour is
delegated to :class:`core.engine.FAQEngine`.

Run with::

    streamlit run app.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from core import EngineConfig, FAQChatbotError, FAQEngine, RetrievalResult, TextPreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("faq_chatbot.app")

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "faqs.json"
CACHE_DIR = BASE_DIR / ".cache"

DEFAULT_THRESHOLD = 0.25
WELCOME_MESSAGE = (
    "Hello! I'm the Northfield University help assistant. "
    "Ask me about IT services, admissions, fees, registration, or housing."
)
GENERIC_ERROR_MESSAGE = "Something went wrong while processing your question. Please try again."

Message = Dict[str, Any]


@st.cache_resource(show_spinner="Building FAQ vector space...")
def load_engine(dataset_path: str, cache_dir: str, threshold: float) -> FAQEngine:
    """Build (once per process) and cache the FAQ engine.

    Args:
        dataset_path: Path to the FAQ JSON file.
        cache_dir: Directory for the persisted TF-IDF vector space.
        threshold: Default confidence threshold.

    Returns:
        A ready-to-query engine shared across sessions.
    """
    config = EngineConfig(
        dataset_path=Path(dataset_path),
        cache_dir=Path(cache_dir),
        confidence_threshold=threshold,
    )
    return FAQEngine(config, TextPreprocessor())


def init_session_state() -> None:
    """Initialise per-session chat history and settings."""
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": WELCOME_MESSAGE, "meta": None}]
    if "threshold" not in st.session_state:
        st.session_state.threshold = DEFAULT_THRESHOLD


def clear_chat() -> None:
    """Reset the conversation to the welcome message."""
    st.session_state.messages = [{"role": "assistant", "content": WELCOME_MESSAGE, "meta": None}]


def build_meta(engine: FAQEngine, result: RetrievalResult) -> Dict[str, Any]:
    """Serialise a retrieval result for storage in session state."""
    return engine.explain(result)


def render_meta(meta: Dict[str, Any]) -> None:
    """Render the retrieval observability panel inside an expander."""
    label = "Retrieval details"
    with st.expander(f"{label} - confidence {meta['confidence']:.0%} ({meta['confidence_level']})"):
        cols = st.columns(4)
        cols[0].metric("Confidence", f"{meta['confidence']:.3f}")
        cols[1].metric("Threshold", f"{meta['threshold']:.2f}")
        cols[2].metric("Margin", f"{meta['margin']:.3f}")
        cols[3].metric("Latency", f"{meta['latency_ms']:.2f} ms")

        st.progress(min(max(float(meta["confidence"]), 0.0), 1.0))
        st.markdown(f"**Status:** `{meta['status']}`  |  **Cache hit:** `{meta['cache_hit']}`")
        if meta.get("matched_question"):
            st.markdown(f"**Closest FAQ:** {meta['matched_question']}  \n**Category:** {meta['matched_category']}")
        if meta.get("processed_query"):
            st.markdown(f"**Processed query:** `{meta['processed_query']}`")
        if meta.get("candidates"):
            st.caption("Top candidates")
            st.table(
                [
                    {"Question": c["question"], "Category": c["category"], "Score": f"{c['score']:.3f}"}
                    for c in meta["candidates"]
                ]
            )


def render_message(message: Message) -> None:
    """Render one chat message (and its observability panel if present)."""
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("meta"):
            render_meta(message["meta"])


def render_sidebar(engine: FAQEngine) -> None:
    """Render sidebar controls and engine statistics."""
    with st.sidebar:
        st.header("Controls")
        st.button("Clear Chat", on_click=clear_chat, use_container_width=True)

        st.slider(
            "Confidence threshold",
            min_value=0.05,
            max_value=0.95,
            step=0.05,
            key="threshold",
            help="Matches scoring below this cosine similarity trigger the fallback reply.",
        )

        st.divider()
        st.subheader("Knowledge base")
        st.caption(
            f"{engine.faq_count} FAQs | {engine.vocabulary_size} features | "
            f"{len(engine.categories)} categories"
        )
        with st.expander("Browse questions"):
            for question in engine.list_questions():
                st.markdown(f"- {question}")


def handle_user_input(engine: FAQEngine, prompt: str) -> None:
    """Process a user prompt, render the exchange, and persist it to history."""
    user_message: Message = {"role": "user", "content": prompt, "meta": None}
    st.session_state.messages.append(user_message)
    render_message(user_message)

    try:
        result = engine.answer(prompt, threshold=float(st.session_state.threshold))
        assistant_message: Message = {
            "role": "assistant",
            "content": result.answer,
            "meta": build_meta(engine, result),
        }
    except FAQChatbotError as exc:
        logger.error("Engine error: %s", exc)
        assistant_message = {"role": "assistant", "content": f"{GENERIC_ERROR_MESSAGE} ({exc})", "meta": None}
    except Exception:  # noqa: BLE001 - UI boundary: never crash the chat loop
        logger.exception("Unexpected error while answering query.")
        assistant_message = {"role": "assistant", "content": GENERIC_ERROR_MESSAGE, "meta": None}

    st.session_state.messages.append(assistant_message)
    render_message(assistant_message)


def main() -> None:
    """Application entry point."""
    st.set_page_config(page_title="University FAQ Assistant", page_icon="🎓", layout="centered")
    st.title("🎓 University FAQ Assistant")
    st.caption("Classical NLP retrieval: NLTK preprocessing, TF-IDF vector space, cosine similarity.")

    init_session_state()

    try:
        engine = load_engine(str(DATASET_PATH), str(CACHE_DIR), DEFAULT_THRESHOLD)
    except FAQChatbotError as exc:
        st.error(f"The assistant could not start: {exc}")
        st.stop()
    except Exception as exc:  # noqa: BLE001 - surface any start-up failure to the operator
        logger.exception("Unexpected start-up failure.")
        st.error(f"Unexpected start-up failure: {exc}")
        st.stop()

    render_sidebar(engine)

    history: List[Message] = st.session_state.messages
    for message in history:
        render_message(message)

    prompt = st.chat_input("Ask a question about IT, admissions, fees, or housing...")
    if prompt is not None:
        handle_user_input(engine, prompt)


if __name__ == "__main__":
    main()
