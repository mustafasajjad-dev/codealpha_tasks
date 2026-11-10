# Translate

A minimalist Streamlit translation tool with text-to-speech and one-click copy.
Translation uses `deep-translator` (Google Translate); audio uses `gTTS`.

## Structure

```
translation-tool/
├── app.py                  # Streamlit UI (only file that imports Streamlit)
├── styles.py               # Design tokens and CSS, including RTL script handling
├── services/
│   ├── __init__.py         # Public API of the service layer
│   ├── exceptions.py       # ServiceError hierarchy with user-safe messages
│   ├── translator.py       # TranslationService (ABC), Google, MyMemory, Fallback
│   └── speech.py           # SpeechService (ABC) + GTTSSpeechService
├── .streamlit/config.toml  # Theme
├── requirements.txt
└── README.md
```

## Setup

Requires Python 3.10+ and an internet connection.

```bash
cd translation-tool
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Open http://localhost:8501.

## Usage

1. Pick the source language (default: Auto-detect) and the target language.
2. Type or paste text, then press **Translate**. The text box applies on
   Ctrl+Enter or when it loses focus, which is when the counter updates.
3. Use the copy icon at the top right of the result to copy it.
4. Press **Listen** to hear the translation. Audio is unavailable for
   languages gTTS has no voice for; the button is disabled with a note.
5. **⇄** swaps the languages and moves the translation into the input.

## Design notes

- The UI depends only on `TranslationService` and `SpeechService`. To change
  provider, subclass the ABC, implement `get_languages()` and `_translate()`,
  and return it from `get_translation_service()` in `app.py`.
  Validation, language-code normalisation and the result type are inherited.
- Every service error derives from `ServiceError` and carries a message safe to
  show users. Handled cases: empty input, text over 5,000 characters, unknown
  language codes, timeouts, rate limits and network failures. Timeouts, rate
  limits and network errors are retried twice with exponential backoff.
- `FallbackTranslationService` tries Google first and switches to MyMemory on
  a rate limit, timeout or network error. After a rate limit, Google is skipped
  for 60 seconds.
- `deep-translator` makes requests without a timeout, so calls run in a worker
  thread and are abandoned after 10 seconds.
- User text is never logged.

## Limitations

- This uses Google's unofficial free endpoint. It can throttle or change without
  notice; use an official paid backend for heavy or critical workloads.
- Multi-line input may lose some line breaks in the translation.
- The MyMemory fallback has a smaller free quota, splits long text into
  chunks (paragraph breaks are not kept), and its Auto-detect support depends
  on the installed `deep-translator` version. Pick the source language
  explicitly if Auto-detect fails while Google is blocked.
- Audio is capped at 2,000 characters.
