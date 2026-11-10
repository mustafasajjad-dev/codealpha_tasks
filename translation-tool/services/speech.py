"""Text-to-speech service layer.

Mirrors the translation layer: :class:`SpeechService` is the interface the UI
depends on, and :class:`GTTSSpeechService` is the default ``gTTS`` backend.
"""

from __future__ import annotations

import io
import logging
from abc import ABC, abstractmethod
from typing import ClassVar

import requests
from gtts import gTTS, gTTSError
from gtts.lang import tts_langs

from .exceptions import SpeechError

logger = logging.getLogger(__name__)


class SpeechService(ABC):
    """Backend-agnostic text-to-speech interface."""

    @abstractmethod
    def supports(self, language_code: str) -> bool:
        """Return ``True`` if audio can be generated for ``language_code``."""

    @abstractmethod
    def synthesize(self, text: str, language_code: str) -> bytes:
        """Convert ``text`` to spoken audio.

        Args:
            text: Text to speak.
            language_code: Language of ``text``.

        Returns:
            Encoded audio (MP3) as bytes.

        Raises:
            SpeechError: If the text is empty or too long, the language is not
                supported, or the backend fails.
        """


class GTTSSpeechService(SpeechService):
    """Speech synthesis through Google Text-to-Speech (``gTTS``).

    Args:
        timeout: Seconds to wait for each request to Google.
        slow: Speak more slowly when ``True``.
    """

    MAX_CHARS: ClassVar[int] = 2000
    """gTTS sends long text in many sequential requests, so input is capped."""

    def __init__(self, timeout: float = 10.0, slow: bool = False) -> None:
        self._timeout = timeout
        self._slow = slow
        self._languages: dict[str, str] = {
            code.lower(): code for code in tts_langs()
        }

    def supports(self, language_code: str) -> bool:
        """Return ``True`` if gTTS has a voice for ``language_code``."""
        return (language_code or "").strip().lower() in self._languages

    def synthesize(self, text: str, language_code: str) -> bytes:
        """Generate MP3 audio for ``text``."""
        cleaned = (text or "").strip()
        if not cleaned:
            raise SpeechError("There is no text to read aloud.")
        if len(cleaned) > self.MAX_CHARS:
            raise SpeechError(
                f"Audio is limited to {self.MAX_CHARS:,} characters. "
                "Shorten the text and try again."
            )
        canonical = self._languages.get(language_code.strip().lower())
        if canonical is None:
            raise SpeechError("Audio is not available for this language.")

        buffer = io.BytesIO()
        try:
            gTTS(
                text=cleaned,
                lang=canonical,
                slow=self._slow,
                timeout=self._timeout,
            ).write_to_fp(buffer)
        except requests.exceptions.Timeout as exc:
            raise SpeechError("The audio service took too long. Try again.") from exc
        except (gTTSError, requests.exceptions.RequestException) as exc:
            logger.warning("gTTS request failed: %s", type(exc).__name__)
            raise SpeechError(
                "Could not generate audio. Check your connection and try again."
            ) from exc
        except (AssertionError, ValueError) as exc:
            raise SpeechError("This text cannot be read aloud.") from exc
        return buffer.getvalue()
