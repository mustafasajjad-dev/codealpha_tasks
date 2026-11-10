"""Exception hierarchy for the service layer.

Every error that can leave the service layer derives from :class:`ServiceError`,
and every message is written for end users. The UI can therefore catch
``ServiceError`` and display ``str(exc)`` without knowing which backend failed.
"""

from __future__ import annotations


class ServiceError(Exception):
    """Base class for all service-layer errors. ``str(exc)`` is user-safe."""


# --- Input problems (the caller can fix these) ------------------------------


class InvalidInputError(ServiceError):
    """The request itself is invalid."""


class EmptyTextError(InvalidInputError):
    """The text to process was empty or whitespace-only."""


class TextTooLongError(InvalidInputError):
    """The text exceeds the maximum length the service accepts."""


class UnsupportedLanguageError(InvalidInputError):
    """A language code is not supported by the active backend."""


# --- Translation backend problems -------------------------------------------


class TranslationError(ServiceError):
    """The translation backend failed to produce a result."""


class TranslationTimeoutError(TranslationError):
    """The backend did not respond within the configured timeout."""


class RateLimitError(TranslationError):
    """The backend is throttling requests."""


class BackendUnavailableError(TranslationError):
    """The backend could not be reached (network failure, HTTP error)."""


# --- Text-to-speech problems ------------------------------------------------


class SpeechError(ServiceError):
    """Speech synthesis failed."""
