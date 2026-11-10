"""Streamlit front end for the translation tool.

This module is the only place that imports Streamlit. It reaches the service
layer exclusively through the abstract ``TranslationService`` and
``SpeechService`` interfaces; the concrete backends are chosen in the two
``get_*_service`` factories below (the composition root). Presentation lives in
``styles.py``.

Run with::

    streamlit run app.py
"""

from __future__ import annotations

import logging
from typing import Final

import streamlit as st

from services import (
    AUTO_DETECT,
    FallbackTranslationService,
    GoogleTranslationService,
    GTTSSpeechService,
    InvalidInputError,
    Language,
    MyMemoryTranslationService,
    ServiceError,
    SpeechService,
    TranslationResult,
    TranslationService,
)
from styles import BASE_CSS, INPUT_SELECTOR, OUTPUT_SELECTOR, script_css

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Translate", page_icon="🔤", layout="centered")

DEFAULT_TARGET: Final[str] = "en"

SOURCE_KEY: Final[str] = "source_code"
TARGET_KEY: Final[str] = "target_code"
INPUT_KEY: Final[str] = "input_text"
RESULT_KEY: Final[str] = "result"
AUDIO_KEY: Final[str] = "audio"

Notice = tuple[str, str]
"""A message for the user: ``("warning" | "error", text)``."""


# --- Composition root -------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_translation_service() -> TranslationService:
    """Create the translation backend once per server process.

    Google is tried first without retries; when it is throttled or unreachable
    the request goes to MyMemory instead.
    """
    try:
        secondary: TranslationService | None = MyMemoryTranslationService(
            timeout=20.0, max_retries=1
        )
    except ServiceError:
        logger.exception("MyMemory fallback unavailable; using Google only")
        secondary = None
    if secondary is None:
        return GoogleTranslationService()
    return FallbackTranslationService(
        primary=GoogleTranslationService(max_retries=0),
        secondary=secondary,
    )


@st.cache_resource(show_spinner=False)
def get_speech_service() -> SpeechService:
    """Create the speech backend once per server process."""
    return GTTSSpeechService()


@st.cache_data(show_spinner=False, ttl=3600, max_entries=32)
def synthesize_cached(_service: SpeechService, text: str, language_code: str) -> bytes:
    """Synthesize audio, caching by ``(text, language_code)``.

    The leading underscore tells Streamlit not to hash the service object.
    """
    return _service.synthesize(text, language_code)


# --- State and callbacks ----------------------------------------------------


def _init_state() -> None:
    """Seed ``st.session_state`` on first load."""
    st.session_state.setdefault(SOURCE_KEY, AUTO_DETECT)
    st.session_state.setdefault(TARGET_KEY, DEFAULT_TARGET)
    st.session_state.setdefault(INPUT_KEY, "")
    st.session_state.setdefault(RESULT_KEY, None)
    st.session_state.setdefault(AUDIO_KEY, None)


def _swap_languages() -> None:
    """Swap source and target; reuse the translation as the new input."""
    source = st.session_state[SOURCE_KEY]
    if source == AUTO_DETECT:
        return
    st.session_state[SOURCE_KEY] = st.session_state[TARGET_KEY]
    st.session_state[TARGET_KEY] = source
    result: TranslationResult | None = st.session_state.get(RESULT_KEY)
    if result is not None:
        st.session_state[INPUT_KEY] = result.translated_text
    st.session_state[RESULT_KEY] = None
    st.session_state[AUDIO_KEY] = None


# --- Rendering --------------------------------------------------------------


def _render_toolbar(languages: tuple[Language, ...]) -> tuple[str, str]:
    """Render the language bar.

    Returns:
        ``(source_code, target_code)``.
    """
    names = {language.code: language.name for language in languages}
    names[AUTO_DETECT] = "Auto-detect"
    codes = [language.code for language in languages]

    with st.container(key="toolbar"):
        left, middle, right = st.columns([10, 1.4, 10], vertical_alignment="center")
        with left:
            source = st.selectbox(
                "Translate from",
                [AUTO_DETECT, *codes],
                key=SOURCE_KEY,
                format_func=lambda code: names.get(code, code),
                label_visibility="collapsed",
            )
        with middle:
            st.button(
                "⇄",
                key="swap",
                help="Swap languages",
                disabled=source == AUTO_DETECT,
                on_click=_swap_languages,
            )
        with right:
            target = st.selectbox(
                "Translate to",
                codes,
                key=TARGET_KEY,
                format_func=lambda code: names.get(code, code),
                label_visibility="collapsed",
            )
    return source, target


def _render_char_count(length: int, limit: int) -> None:
    """Show a live character counter below the input."""
    css_class = "char-count near-limit" if length >= 0.9 * limit else "char-count"
    st.markdown(
        f'<div class="{css_class}">{length:,} / {limit:,}</div>',
        unsafe_allow_html=True,
    )


def _run_translation(
    translator: TranslationService, text: str, source: str, target: str
) -> Notice | None:
    """Translate, store the result in session state and report any problem.

    Returns:
        A notice to show the user when the translation failed, else ``None``.
    """
    st.session_state[AUDIO_KEY] = None
    try:
        with st.spinner("Translating…"):
            result = translator.translate(text, target=target, source=source)
    except InvalidInputError as exc:
        st.session_state[RESULT_KEY] = None
        return ("warning", str(exc))
    except ServiceError as exc:
        st.session_state[RESULT_KEY] = None
        return ("error", str(exc))
    except Exception:  # noqa: BLE001 - never show a stack trace to users
        logger.exception("Unhandled error during translation")
        st.session_state[RESULT_KEY] = None
        return ("error", "Something went wrong. Try again in a moment.")
    st.session_state[RESULT_KEY] = result
    return None


def _render_result(
    result: TranslationResult, language_names: dict[str, str], speech: SpeechService
) -> Notice | None:
    """Show the translation, its copy button and audio playback."""
    # st.code supplies a built-in one-click copy button and renders text verbatim.
    st.code(result.translated_text, language=None, wrap_lines=True)

    notice: Notice | None = None
    target_name = language_names.get(result.target_language, result.target_language)
    can_speak = speech.supports(result.target_language)
    autoplay = False
    if st.button(
        "Listen",
        key="listen",
        disabled=not can_speak,
        help=None if can_speak else f"Audio is not available for {target_name}.",
    ):
        try:
            with st.spinner("Preparing audio…"):
                st.session_state[AUDIO_KEY] = synthesize_cached(
                    speech, result.translated_text, result.target_language
                )
            autoplay = True
        except ServiceError as exc:
            st.session_state[AUDIO_KEY] = None
            notice = ("error", str(exc))

    audio: bytes | None = st.session_state.get(AUDIO_KEY)
    if audio:
        st.audio(audio, format="audio/mp3", autoplay=autoplay)
    return notice


def _show_notice(notice: Notice | None) -> None:
    """Display a warning or error beneath the sheet."""
    if notice is None:
        return
    kind, message = notice
    (st.warning if kind == "warning" else st.error)(message)


def main() -> None:
    """Render the page."""
    st.markdown(f"<style>{BASE_CSS}</style>", unsafe_allow_html=True)

    translator = get_translation_service()
    speech = get_speech_service()
    languages = translator.get_languages()
    language_names = {language.code: language.name for language in languages}

    _init_state()
    notice: Notice | None = None

    st.markdown('<h1 class="wordmark">Translate</h1>', unsafe_allow_html=True)

    with st.container(key="sheet"):
        source, target = _render_toolbar(languages)
        left, right = st.columns(2, gap="small")

        with left, st.container(key="pane_in"):
            text: str = st.text_area(
                "Text to translate",
                key=INPUT_KEY,
                height=240,
                max_chars=translator.MAX_CHARS,
                placeholder="Type or paste text",
                label_visibility="collapsed",
            )
            count_col, action_col = st.columns([1, 1], vertical_alignment="center")
            with count_col:
                _render_char_count(len(text), translator.MAX_CHARS)
            with action_col:
                clicked = st.button("Translate", key="translate", type="primary")
            if clicked:
                notice = _run_translation(translator, text, source, target)

        with right, st.container(key="pane_out"):
            result: TranslationResult | None = st.session_state.get(RESULT_KEY)
            if result is None:
                st.markdown(
                    '<p class="placeholder">Your translation appears here.</p>',
                    unsafe_allow_html=True,
                )
            else:
                notice = _render_result(result, language_names, speech) or notice

    _show_notice(notice)

    # Direction and fonts depend on the languages chosen in this run, so this
    # block is emitted last. It produces no visible output.
    dynamic_css = script_css(INPUT_SELECTOR, source if source != AUTO_DETECT else "")
    if result is not None:
        dynamic_css += script_css(OUTPUT_SELECTOR, result.target_language)
    if dynamic_css:
        st.markdown(f"<style>{dynamic_css}</style>", unsafe_allow_html=True)


main()
