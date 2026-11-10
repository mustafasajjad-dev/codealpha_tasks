"""Visual design for the Streamlit UI: tokens, stylesheet and script handling.

Kept apart from ``app.py`` so layout logic and presentation can change
independently. The layout follows the two-pane pattern used by Google Translate
and DeepL: one sheet, a language bar on top, source on the left, result on the
right.

Palette
    mist      #EEF1F5  page background
    paper     #FFFFFF  the sheet
    wash      #F6F8FB  result pane
    ink       #1B2433  text
    slate     #5B6678  secondary text
    rule      #DDE2EA  hairlines
    cobalt    #2547D0  the single accent (primary button, focus)
    amber     #B45309  character-limit warning

Type
    Hanken Grotesk for the interface, Noto Nastaliq Urdu and Noto Naskh Arabic
    for right-to-left scripts.
"""

from __future__ import annotations

from typing import Final

_RTL_SUBTAGS: Final[frozenset[str]] = frozenset(
    {"ar", "ckb", "dv", "fa", "he", "iw", "pnb", "ps", "sd", "ug", "ur", "yi"}
)
_ARABIC_SCRIPT_SUBTAGS: Final[frozenset[str]] = frozenset(
    {"ar", "ckb", "fa", "pnb", "ps", "sd", "ug"}
)

INPUT_SELECTOR: Final[str] = ".st-key-input_text textarea"
OUTPUT_SELECTOR: Final[str] = (
    '.st-key-pane_out [data-testid="stCode"] pre, '
    '.st-key-pane_out [data-testid="stCode"] code'
)


def _subtag(code: str) -> str:
    return (code or "").lower().split("-")[0]


def is_rtl(code: str) -> bool:
    """Return ``True`` if the language is written right to left."""
    return _subtag(code) in _RTL_SUBTAGS


def script_css(selector: str, code: str) -> str:
    """Return CSS that sets direction and font for a right-to-left language.

    Args:
        selector: CSS selector for the text to style.
        code: Language code of that text (``"ur"``, ``"ar"``, ``"en-GB"``...).

    Returns:
        A CSS rule, or an empty string for left-to-right languages.
    """
    if not is_rtl(code):
        return ""
    subtag = _subtag(code)
    if subtag == "ur":
        font = '"Noto Nastaliq Urdu", "Jameel Noori Nastaleeq", serif'
        line_height = "2.1"
    elif subtag in _ARABIC_SCRIPT_SUBTAGS:
        font = '"Noto Naskh Arabic", serif'
        line_height = "1.9"
    else:
        font = "inherit"
        line_height = "1.7"
    return (
        f"{selector} {{ direction: rtl; text-align: right; "
        f"font-family: {font} !important; line-height: {line_height} !important; }}"
    )


BASE_CSS: Final[str] = """
@import url('https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@400;500;600;700&family=Noto+Naskh+Arabic:wght@400;500&family=Noto+Nastaliq+Urdu:wght@400;500&display=swap');

:root {
  --mist: #EEF1F5;
  --paper: #FFFFFF;
  --wash: #F6F8FB;
  --ink: #1B2433;
  --slate: #5B6678;
  --faint: #9AA3B2;
  --rule: #DDE2EA;
  --cobalt: #2547D0;
  --cobalt-deep: #1C3AAE;
  --amber: #B45309;
  --ui: "Hanken Grotesk", system-ui, -apple-system, "Segoe UI", sans-serif;
}

/* The style tag itself must not take up space in the page flow. */
.stElementContainer:has(style), .element-container:has(style) { display: none; }

html, body, .stApp, textarea, button, input, [data-baseweb="select"] {
  font-family: var(--ui);
}
.stApp { background: var(--mist); color: var(--ink); }
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { max-width: 1040px; padding: 3.2rem 1.5rem 4rem; }

.wordmark {
  font-size: 1.45rem; font-weight: 650; letter-spacing: -0.015em;
  margin: 0 0 1.4rem 0.25rem; color: var(--ink);
}

/* --- Sheet ------------------------------------------------------------ */
.st-key-sheet {
  background: var(--paper);
  border: 1px solid var(--rule);
  border-radius: 18px;
  overflow: hidden;
  gap: 0;
  box-shadow: 0 1px 2px rgba(27, 36, 51, 0.04), 0 14px 34px -14px rgba(27, 36, 51, 0.16);
}
.st-key-sheet [data-testid="stHorizontalBlock"] { gap: 0; align-items: stretch; }

/* Language bar */
.st-key-toolbar { border-bottom: 1px solid var(--rule); padding: 0.35rem 0.75rem; }
.st-key-toolbar [data-baseweb="select"] > div {
  background: transparent; border: 0; box-shadow: none;
  min-height: 2.6rem; font-weight: 600; font-size: 1rem; color: var(--ink);
  transition: background 0.15s ease;
}
.st-key-toolbar [data-baseweb="select"] > div:hover { background: var(--wash); }
.st-key-swap { display: flex; justify-content: center; }
.st-key-swap button {
  width: 2.3rem; height: 2.3rem; min-height: 0; padding: 0;
  border-radius: 999px; border: 1px solid var(--rule);
  background: var(--paper); color: var(--slate);
  transition: border-color 0.15s ease, color 0.15s ease;
}
.st-key-swap button:hover:not(:disabled) { border-color: var(--cobalt); color: var(--cobalt); }
.st-key-swap button:disabled { opacity: 0.4; }

/* Panes */
.st-key-pane_in, .st-key-pane_out { padding: 1.1rem 1.5rem 1rem; min-height: 22rem; }
[data-testid="stColumn"]:has(.st-key-pane_out) {
  background: var(--wash); border-left: 1px solid var(--rule);
}

.st-key-input_text [data-baseweb="textarea"],
.st-key-input_text [data-baseweb="base-input"] {
  background: transparent !important; border: 0 !important; box-shadow: none !important;
}
.st-key-input_text textarea {
  font-size: 1.2rem; line-height: 1.6; padding: 0; color: var(--ink);
}
.st-key-input_text textarea::placeholder { color: var(--faint); }

.char-count { font-size: 0.82rem; color: var(--slate); font-variant-numeric: tabular-nums; }
.char-count.near-limit { color: var(--amber); font-weight: 600; }

.st-key-translate { display: flex; justify-content: flex-end; }
.st-key-translate button {
  background: var(--cobalt); color: #fff; border: 0; border-radius: 10px;
  padding: 0.5rem 1.4rem; font-weight: 600;
  transition: background 0.15s ease;
}
.st-key-translate button:hover { background: var(--cobalt-deep); color: #fff; }
.st-key-translate button:focus-visible,
.st-key-swap button:focus-visible,
.st-key-listen button:focus-visible { outline: 2px solid var(--cobalt); outline-offset: 2px; }

/* Result */
.placeholder { color: var(--faint); font-size: 1.2rem; line-height: 1.6; margin: 0; }
.st-key-pane_out [data-testid="stCode"] { background: transparent; border: 0; }
.st-key-pane_out [data-testid="stCode"] pre {
  background: transparent !important; padding: 0 2.4rem 0 0; min-height: 15rem;
}
.st-key-pane_out [data-testid="stCode"] pre,
.st-key-pane_out [data-testid="stCode"] code,
.st-key-pane_out [data-testid="stCode"] code span {
  font-family: var(--ui) !important; font-size: 1.2rem; line-height: 1.6;
  color: var(--ink) !important; white-space: pre-wrap !important;
}
.st-key-pane_out [data-testid="stCodeCopyButton"] { opacity: 1 !important; color: var(--slate); }

.st-key-listen button {
  background: transparent; border: 0; border-radius: 8px;
  color: var(--slate); font-weight: 500; padding: 0.35rem 0.7rem;
  transition: background 0.15s ease, color 0.15s ease;
}
.st-key-listen button:hover:not(:disabled) { background: var(--mist); color: var(--ink); }
.st-key-pane_out audio { width: 100%; margin-top: 0.5rem; }

[data-testid="stAlert"] { border-radius: 12px; margin-top: 1rem; }

@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
"""
