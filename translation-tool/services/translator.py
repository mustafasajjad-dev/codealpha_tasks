"""Translation service layer.

Structure:

* :class:`TranslationService` is the abstract interface. It owns everything that
  is backend-independent: input validation, language-code normalisation and the
  shape of the result. Subclasses implement :meth:`get_languages` and
  :meth:`_translate`.
* :class:`DeepTranslatorService` is shared plumbing for ``deep-translator``
  backends: a hard timeout, retries with exponential backoff and mapping of
  library errors to :mod:`services.exceptions`.
* :class:`GoogleTranslationService` and :class:`MyMemoryTranslationService` are
  the concrete backends.
* :class:`FallbackTranslationService` wraps two services and switches to the
  second when the first is throttled or unreachable.

Swapping to another provider means writing one new subclass; nothing in the UI
changes. User text is never written to logs.
"""

from __future__ import annotations

import logging
import re
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import ClassVar, Final

import requests
from deep_translator import GoogleTranslator, MyMemoryTranslator
from deep_translator.exceptions import (
    ElementNotFoundInGetRequest,
    LanguageNotSupportedException,
    NotValidLength,
    NotValidPayload,
    RequestError,
    TooManyRequests,
    TranslationNotFound,
)

from .exceptions import (
    BackendUnavailableError,
    EmptyTextError,
    RateLimitError,
    ServiceError,
    TextTooLongError,
    TranslationError,
    TranslationTimeoutError,
    UnsupportedLanguageError,
)

logger = logging.getLogger(__name__)

AUTO_DETECT: Final[str] = "auto"
"""Pseudo language code meaning "detect the source language automatically"."""


@dataclass(frozen=True, slots=True)
class Language:
    """A language supported by a translation backend.

    Attributes:
        code: Backend language code, e.g. ``"en"`` or ``"zh-CN"``.
        name: Human-readable name, e.g. ``"English"``.
    """

    code: str
    name: str


@dataclass(frozen=True, slots=True)
class TranslationResult:
    """Outcome of a successful translation.

    Attributes:
        original_text: The validated (stripped) input text.
        translated_text: The translated text.
        source_language: Resolved source code, or :data:`AUTO_DETECT`.
        target_language: Resolved target language code.
    """

    original_text: str
    translated_text: str
    source_language: str
    target_language: str


class TranslationService(ABC):
    """Backend-agnostic translation interface.

    Public callers use :meth:`translate`, which validates input and then
    delegates to the backend-specific :meth:`_translate` (template method).
    """

    MAX_CHARS: ClassVar[int] = 5000
    """Maximum accepted input length, in characters."""

    @abstractmethod
    def get_languages(self) -> tuple[Language, ...]:
        """Return the languages the backend can translate to and from.

        The pseudo language :data:`AUTO_DETECT` is *not* included.

        Returns:
            Languages sorted by display name.
        """

    @abstractmethod
    def _translate(self, text: str, source: str, target: str) -> str:
        """Translate already-validated text.

        Args:
            text: Non-empty, stripped text no longer than ``MAX_CHARS``.
            source: A resolved language code or :data:`AUTO_DETECT`.
            target: A resolved language code.

        Returns:
            The translated text.

        Raises:
            TranslationError: If the backend fails.
        """

    def translate(
        self, text: str, target: str, source: str = AUTO_DETECT
    ) -> TranslationResult:
        """Validate the request and translate ``text``.

        Args:
            text: Text to translate.
            target: Target language code (case-insensitive).
            source: Source language code, or :data:`AUTO_DETECT` (default).

        Returns:
            A :class:`TranslationResult`.

        Raises:
            EmptyTextError: If ``text`` is empty or whitespace-only.
            TextTooLongError: If ``text`` exceeds ``MAX_CHARS``.
            UnsupportedLanguageError: If a language code is unknown.
            TranslationError: If the backend fails (timeout, rate limit, ...).
        """
        cleaned = self._validate_text(text)
        resolved_source = self._resolve_language(source, allow_auto=True)
        resolved_target = self._resolve_language(target, allow_auto=False)
        translated = self._translate(cleaned, resolved_source, resolved_target)
        return TranslationResult(
            original_text=cleaned,
            translated_text=translated,
            source_language=resolved_source,
            target_language=resolved_target,
        )

    def _validate_text(self, text: str) -> str:
        """Return stripped ``text`` or raise if it is unusable."""
        if not isinstance(text, str) or not text.strip():
            raise EmptyTextError("Enter some text to translate.")
        cleaned = text.strip()
        if len(cleaned) > self.MAX_CHARS:
            raise TextTooLongError(
                f"The text is {len(cleaned):,} characters long. "
                f"The limit is {self.MAX_CHARS:,}."
            )
        return cleaned

    def _resolve_language(self, code: str, *, allow_auto: bool) -> str:
        """Normalise ``code`` to the backend's canonical spelling."""
        candidate = (code or "").strip().lower()
        if allow_auto and candidate == AUTO_DETECT:
            return AUTO_DETECT
        for language in self.get_languages():
            if language.code.lower() == candidate:
                return language.code
        raise UnsupportedLanguageError(f"'{code}' is not a supported language code.")


class DeepTranslatorService(TranslationService, ABC):
    """Shared plumbing for backends built on ``deep-translator``.

    ``deep-translator`` issues its HTTP requests without a timeout, so each call
    runs in a worker thread and is abandoned if it exceeds ``timeout``. A
    timed-out request may keep running in the background until its socket
    closes; it cannot be forcibly cancelled.

    Args:
        timeout: Seconds to wait for each attempt.
        max_retries: Extra attempts after a timeout, rate limit or network error.
        backoff_seconds: Base delay between attempts; doubles each retry.
        max_workers: Size of the worker thread pool.
    """

    _RETRYABLE: Final[tuple[type[TranslationError], ...]] = (
        RateLimitError,
        TranslationTimeoutError,
        BackendUnavailableError,
    )

    def __init__(
        self,
        timeout: float = 10.0,
        max_retries: int = 2,
        backoff_seconds: float = 1.0,
        max_workers: int = 4,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self._timeout = timeout
        self._max_attempts = max_retries + 1
        self._backoff = backoff_seconds
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="translate"
        )
        self._languages = self._load_languages()

    @abstractmethod
    def _load_languages(self) -> tuple[Language, ...]:
        """Build the sorted language table for this backend."""

    @abstractmethod
    def _call_backend(self, text: str, source: str, target: str) -> str:
        """Perform the blocking network call (runs in a worker thread)."""

    def get_languages(self) -> tuple[Language, ...]:
        """Return supported languages sorted by name."""
        return self._languages

    def close(self) -> None:
        """Release the worker threads."""
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _translate(self, text: str, source: str, target: str) -> str:
        """Translate with retries and exponential backoff."""
        for attempt in range(1, self._max_attempts + 1):
            try:
                return self._attempt(text, source, target)
            except self._RETRYABLE as exc:
                if attempt == self._max_attempts:
                    raise
                delay = self._backoff * 2 ** (attempt - 1)
                logger.warning(
                    "%s attempt %d/%d failed (%s); retrying in %.1fs",
                    type(self).__name__,
                    attempt,
                    self._max_attempts,
                    type(exc).__name__,
                    delay,
                )
                time.sleep(delay)
        raise TranslationError("Translation failed.")  # pragma: no cover

    def _attempt(self, text: str, source: str, target: str) -> str:
        """Run one backend call and map library errors to service errors."""
        future = self._executor.submit(self._call_backend, text, source, target)
        try:
            return future.result(timeout=self._timeout)
        except FutureTimeoutError as exc:
            future.cancel()
            raise TranslationTimeoutError(
                "The translation service took too long to respond. Try again."
            ) from exc
        except ServiceError:
            raise
        except TooManyRequests as exc:
            raise RateLimitError(
                "Too many requests. Wait a moment, then try again."
            ) from exc
        except NotValidPayload as exc:
            raise EmptyTextError("Enter some text to translate.") from exc
        except NotValidLength as exc:
            raise TextTooLongError(
                f"The text is too long. The limit is {self.MAX_CHARS:,} characters."
            ) from exc
        except LanguageNotSupportedException as exc:
            raise UnsupportedLanguageError(
                "That language is not supported by the translation service."
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise TranslationTimeoutError(
                "The translation service took too long to respond. Try again."
            ) from exc
        except (requests.exceptions.ConnectionError, RequestError) as exc:
            raise BackendUnavailableError(
                "Could not reach the translation service. Check your internet "
                "connection and try again."
            ) from exc
        except (TranslationNotFound, ElementNotFoundInGetRequest) as exc:
            raise TranslationError(
                "The service returned no translation for this text."
            ) from exc
        except Exception as exc:  # noqa: BLE001 - last line of defence
            logger.exception("Unexpected translation backend failure")
            raise TranslationError(
                "Something went wrong while translating. Try again."
            ) from exc


class GoogleTranslationService(DeepTranslatorService):
    """Google Translate backend (unofficial free endpoint)."""

    def _load_languages(self) -> tuple[Language, ...]:
        """Read the language table bundled with ``deep-translator``."""
        try:
            mapping: dict[str, str] = GoogleTranslator(
                source=AUTO_DETECT, target="en"
            ).get_supported_languages(as_dict=True)  # type: ignore[assignment]
        except Exception as exc:  # noqa: BLE001 - surface as a service error
            logger.exception("Could not load Google language table")
            raise TranslationError("Could not load the language list.") from exc
        return tuple(
            sorted(
                (Language(code=code, name=name.title()) for name, code in mapping.items()),
                key=lambda language: language.name,
            )
        )

    def _call_backend(self, text: str, source: str, target: str) -> str:
        translated = GoogleTranslator(source=source, target=target).translate(text)
        if not translated or not translated.strip():
            raise TranslationError("The service returned an empty translation.")
        return translated


def _chunk_text(text: str, limit: int) -> list[str]:
    """Split ``text`` into chunks of at most ``limit`` characters.

    Splits on sentence ends and line breaks first, then on spaces, and finally
    hard-cuts words longer than ``limit``.
    """
    pieces: list[str] = []
    for sentence in re.split(r"(?<=[.!?\u061f\u06d4\u3002])\s+|\n+", text):
        sentence = sentence.strip()
        while len(sentence) > limit:
            cut = sentence.rfind(" ", 0, limit)
            if cut <= 0:
                cut = limit
            piece = sentence[:cut].strip()
            if piece:
                pieces.append(piece)
            sentence = sentence[cut:].strip()
        if sentence:
            pieces.append(sentence)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + 1 + len(piece) > limit:
            chunks.append(current)
            current = piece
        else:
            current = f"{current} {piece}".strip()
    if current:
        chunks.append(current)
    return chunks


class MyMemoryTranslationService(DeepTranslatorService):
    """MyMemory backend: free, no API key, but a small per-request limit.

    MyMemory accepts roughly 500 characters per request, so longer input is
    split into chunks and the results are joined with spaces; paragraph breaks
    are not preserved. Google-style codes (``"ur"``) are mapped to MyMemory
    codes (``"ur-PK"``) automatically, so this service can stand in for
    :class:`GoogleTranslationService`.

    Args:
        email: Optional contact address; MyMemory raises its daily quota for
            requests that include one.
        **kwargs: Forwarded to :class:`DeepTranslatorService`.
    """

    _CHUNK_CHARS: Final[int] = 450
    _API_URL: Final[str] = "https://api.mymemory.translated.net/get"

    def __init__(self, email: str | None = None, **kwargs: float | int) -> None:
        self._email = email
        self._code_map: dict[str, str] = {}
        super().__init__(**kwargs)  # type: ignore[arg-type]

    @staticmethod
    def _read_language_table() -> dict[str, str]:
        """Return MyMemory's ``{name: code}`` table across library versions."""
        try:
            from deep_translator.constants import MY_MEMORY_LANGUAGES_TO_CODES

            return dict(MY_MEMORY_LANGUAGES_TO_CODES)
        except ImportError:
            # Older releases: the table is exposed by the translator itself.
            # MyMemory's own codes (e.g. "en-GB") are required to construct it.
            translator = MyMemoryTranslator(source="en-GB", target="fr-FR")
            return dict(translator.get_supported_languages(as_dict=True))

    def _load_languages(self) -> tuple[Language, ...]:
        """Read MyMemory's language table and build a short-code lookup."""
        try:
            table = self._read_language_table()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Could not load MyMemory language table")
            raise TranslationError("Could not load the language list.") from exc

        for code in table.values():
            lowered = code.lower()
            self._code_map.setdefault(lowered, code)
            self._code_map.setdefault(lowered.split("-")[0], code)
        return tuple(
            sorted(
                (Language(code=code, name=name.title()) for name, code in table.items()),
                key=lambda language: language.name,
            )
        )

    def _to_backend_code(self, code: str) -> str:
        """Map ``"ur"`` or ``"ur-PK"`` (any case) to MyMemory's own code."""
        lowered = (code or "").strip().lower()
        mapped = self._code_map.get(lowered) or self._code_map.get(
            lowered.split("-")[0]
        )
        if mapped is None:
            raise UnsupportedLanguageError(f"'{code}' is not a supported language code.")
        return mapped

    def _resolve_language(self, code: str, *, allow_auto: bool) -> str:
        if allow_auto and (code or "").strip().lower() == AUTO_DETECT:
            return AUTO_DETECT
        return self._to_backend_code(code)

    def _call_backend(self, text: str, source: str, target: str) -> str:
        """Call MyMemory's HTTP API directly, chunk by chunk.

        The REST API is used instead of ``deep-translator``'s wrapper because
        the wrapper's handling of auto-detection differs between releases.
        """
        langpair = f"{'Autodetect' if source == AUTO_DETECT else source}|{target}"
        parts = [
            self._request_chunk(chunk, langpair)
            for chunk in _chunk_text(text, self._CHUNK_CHARS)
        ]
        joined = " ".join(part for part in parts if part).strip()
        if not joined:
            raise TranslationError("The service returned an empty translation.")
        return joined

    def _request_chunk(self, chunk: str, langpair: str) -> str:
        """Translate one chunk of at most ``_CHUNK_CHARS`` characters."""
        params = {"q": chunk, "langpair": langpair}
        if self._email:
            params["de"] = self._email
        response = requests.get(self._API_URL, params=params, timeout=self._timeout)
        if response.status_code == 429:
            raise RateLimitError("The backup translation quota has been reached.")
        if response.status_code != 200:
            raise BackendUnavailableError(
                "The backup translation service is not responding. Try again."
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise TranslationError(
                "The backup translation service returned an unreadable response."
            ) from exc

        status = str(payload.get("responseStatus", ""))
        translated = str((payload.get("responseData") or {}).get("translatedText") or "")
        if status == "429" or translated.upper().startswith("MYMEMORY WARNING"):
            raise RateLimitError("The backup translation quota has been reached.")
        if status != "200":
            logger.warning(
                "MyMemory rejected the request: status=%s details=%s langpair=%s",
                status,
                payload.get("responseDetails"),
                langpair,
            )
            raise TranslationError(
                "The backup translation service could not translate this text."
            )
        return translated


class FallbackTranslationService(TranslationService):
    """Try ``primary``; switch to ``secondary`` when the primary is unavailable.

    After the primary reports a rate limit it is skipped for ``cooldown_seconds``
    so that every request does not pay for a call that is known to fail. Input
    errors (empty text, bad language) are never retried on the secondary.

    Args:
        primary: Preferred service. Its language list is what the UI shows.
        secondary: Service used when the primary fails with a backend error.
        cooldown_seconds: How long to skip the primary after a rate limit.
    """

    def __init__(
        self,
        primary: TranslationService,
        secondary: TranslationService,
        cooldown_seconds: float = 60.0,
    ) -> None:
        self._primary = primary
        self._secondary = secondary
        self._cooldown = cooldown_seconds
        self._primary_blocked_until = 0.0
        self._last_primary_error: TranslationError | None = None

    def get_languages(self) -> tuple[Language, ...]:
        """Return the primary service's languages."""
        return self._primary.get_languages()

    def _translate(self, text: str, source: str, target: str) -> str:
        if time.monotonic() >= self._primary_blocked_until:
            try:
                return self._primary.translate(
                    text, target=target, source=source
                ).translated_text
            except TranslationError as exc:
                self._last_primary_error = exc
                if isinstance(exc, RateLimitError):
                    self._primary_blocked_until = time.monotonic() + self._cooldown
                logger.warning(
                    "Primary translator failed (%s); using fallback",
                    type(exc).__name__,
                )

        try:
            return self._secondary.translate(
                text, target=target, source=source
            ).translated_text
        except ServiceError as exc:
            logger.warning("Fallback translator failed (%s): %s", type(exc).__name__, exc)
            if self._last_primary_error is not None and isinstance(exc, TranslationError):
                raise TranslationError(
                    "Google is blocking requests from this connection and the "
                    "backup service failed too. Try again in a few minutes."
                ) from exc
            raise (self._last_primary_error or exc) from exc
