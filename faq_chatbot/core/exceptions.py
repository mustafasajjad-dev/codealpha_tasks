"""Domain-specific exception hierarchy for the FAQ retrieval chatbot.

All errors raised intentionally by this package derive from
:class:`FAQChatbotError`, allowing callers (e.g. the UI layer) to handle every
expected failure mode with a single ``except`` clause while still being able to
discriminate between failure categories when needed.
"""

from __future__ import annotations


class FAQChatbotError(Exception):
    """Base class for all errors raised by the FAQ chatbot package."""


class ConfigurationError(FAQChatbotError):
    """Raised when an engine or preprocessor configuration value is invalid."""


class DatasetError(FAQChatbotError):
    """Raised when the FAQ dataset is missing, unreadable, or malformed."""


class NLTKResourceError(FAQChatbotError):
    """Raised when required NLTK corpora cannot be located or downloaded."""


class EngineInitializationError(FAQChatbotError):
    """Raised when the TF-IDF vector space cannot be constructed."""
