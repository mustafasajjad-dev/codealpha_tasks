"""Service layer: translation and speech, independent of any UI framework."""

from .exceptions import (
    BackendUnavailableError,
    EmptyTextError,
    InvalidInputError,
    RateLimitError,
    ServiceError,
    SpeechError,
    TextTooLongError,
    TranslationError,
    TranslationTimeoutError,
    UnsupportedLanguageError,
)
from .speech import GTTSSpeechService, SpeechService
from .translator import (
    AUTO_DETECT,
    DeepTranslatorService,
    FallbackTranslationService,
    GoogleTranslationService,
    Language,
    MyMemoryTranslationService,
    TranslationResult,
    TranslationService,
)

__all__ = [
    "AUTO_DETECT",
    "BackendUnavailableError",
    "DeepTranslatorService",
    "EmptyTextError",
    "FallbackTranslationService",
    "GTTSSpeechService",
    "GoogleTranslationService",
    "InvalidInputError",
    "Language",
    "MyMemoryTranslationService",
    "RateLimitError",
    "ServiceError",
    "SpeechError",
    "SpeechService",
    "TextTooLongError",
    "TranslationError",
    "TranslationResult",
    "TranslationService",
    "TranslationTimeoutError",
    "UnsupportedLanguageError",
]
