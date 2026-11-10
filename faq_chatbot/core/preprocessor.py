"""NLTK-based text preprocessing pipeline.

The pipeline is: Unicode normalisation -> lowercasing -> URL/e-mail removal ->
tokenisation -> punctuation stripping -> stop-word removal -> lemmatisation.

All required NLTK corpora are resolved automatically on first use. The class is
deliberately defensive: ``None``, non-string, empty, or whitespace-only input
yields an empty result instead of raising.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import unicodedata
from functools import lru_cache
from typing import FrozenSet, Iterable, List, NamedTuple, Optional, Tuple

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from .exceptions import ConfigurationError, NLTKResourceError

logger = logging.getLogger(__name__)

_PIPELINE_VERSION = "1"
_LEMMA_CACHE_SIZE = 65_536

_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_ALNUM_RE = re.compile(r"[a-z0-9]+")

#: Words that carry semantic weight in FAQ queries despite being stop words.
DEFAULT_KEEP_WORDS: FrozenSet[str] = frozenset({"no", "not"})


class _ResourceSpec(NamedTuple):
    """Describes one NLTK resource and how to locate or download it."""

    name: str
    lookup_paths: Tuple[str, ...]
    packages: Tuple[str, ...]
    required: bool


# Lookup paths include the zipped form because NLTK may leave some corpora
# zipped after download. ``punkt_tab`` replaced ``punkt`` in NLTK >= 3.8.2, so
# either one satisfies the tokenizer requirement.
_RESOURCE_SPECS: Tuple[_ResourceSpec, ...] = (
    _ResourceSpec("punkt", ("tokenizers/punkt_tab", "tokenizers/punkt"), ("punkt_tab", "punkt"), True),
    _ResourceSpec("wordnet", ("corpora/wordnet", "corpora/wordnet.zip"), ("wordnet",), True),
    _ResourceSpec("stopwords", ("corpora/stopwords", "corpora/stopwords.zip"), ("stopwords",), True),
    _ResourceSpec("omw-1.4", ("corpora/omw-1.4", "corpora/omw-1.4.zip"), ("omw-1.4",), False),
)

_resource_lock = threading.Lock()
_resources_ready = False


def _is_available(lookup_paths: Iterable[str]) -> bool:
    """Return ``True`` if any NLTK lookup path resolves locally."""
    for path in lookup_paths:
        try:
            nltk.data.find(path)
            return True
        except (LookupError, OSError, ValueError):
            continue
    return False


def _download(packages: Iterable[str]) -> None:
    """Attempt to download NLTK packages, logging (not raising) on failure."""
    for package in packages:
        try:
            nltk.download(package, quiet=True, raise_on_error=False)
        except Exception as exc:  # noqa: BLE001 - network stack can raise anything
            logger.warning("NLTK download of '%s' failed: %s", package, exc)


def ensure_nltk_resources() -> None:
    """Ensure all NLTK resources needed by the preprocessor are installed.

    Missing resources are downloaded automatically. The check is idempotent and
    thread-safe; after the first success it is a no-op.

    Raises:
        NLTKResourceError: If a required resource is unavailable and could not
            be downloaded (e.g. no network access).
    """
    global _resources_ready
    if _resources_ready:
        return

    with _resource_lock:
        if _resources_ready:
            return

        missing_required: List[str] = []
        for spec in _RESOURCE_SPECS:
            if _is_available(spec.lookup_paths):
                continue
            logger.info("NLTK resource '%s' missing; attempting download.", spec.name)
            _download(spec.packages)
            if not _is_available(spec.lookup_paths):
                if spec.required:
                    missing_required.append(spec.name)
                else:
                    logger.warning("Optional NLTK resource '%s' is unavailable.", spec.name)

        if missing_required:
            raise NLTKResourceError(
                "Required NLTK resources could not be located or downloaded: "
                f"{', '.join(missing_required)}. Check network access or install them "
                "manually with: python -m nltk.downloader punkt punkt_tab wordnet "
                "omw-1.4 stopwords"
            )
        _resources_ready = True


class TextPreprocessor:
    """Normalises raw text into lemmatised, stop-word-free tokens.

    Instances are thread-safe for concurrent ``preprocess`` calls and are
    cheap to share across the application.

    Args:
        language: NLTK language name used for stop words and tokenisation.
        extra_stopwords: Additional words to discard.
        keep_words: Words to retain even if they appear in the stop-word list.
        min_token_length: Minimum length for alphabetic tokens. Purely numeric
            tokens are always retained.

    Raises:
        ConfigurationError: If ``min_token_length`` is invalid or the language
            has no stop-word list.
        NLTKResourceError: If required NLTK data is unavailable.
    """

    def __init__(
        self,
        language: str = "english",
        extra_stopwords: Optional[Iterable[str]] = None,
        keep_words: Optional[Iterable[str]] = None,
        min_token_length: int = 2,
    ) -> None:
        if min_token_length < 1:
            raise ConfigurationError("min_token_length must be >= 1.")

        ensure_nltk_resources()

        self._language = language
        self._min_token_length = min_token_length

        try:
            base_stopwords = set(stopwords.words(language))
        except (OSError, LookupError) as exc:
            raise ConfigurationError(f"No NLTK stop-word list available for language '{language}'.") from exc

        base_stopwords.update(w.lower() for w in (extra_stopwords or ()))
        retained = frozenset(w.lower() for w in (keep_words if keep_words is not None else DEFAULT_KEEP_WORDS))
        self._stopwords: FrozenSet[str] = frozenset(base_stopwords - retained)

        self._lemmatizer = WordNetLemmatizer()
        try:
            self._lemmatizer.lemmatize("probes", pos="n")  # force lazy corpus load now
        except (LookupError, OSError) as exc:
            raise NLTKResourceError(f"WordNet corpus is unusable: {exc}") from exc
        self._lemmatize_cached = lru_cache(maxsize=_LEMMA_CACHE_SIZE)(self._lemmatize_uncached)

        fingerprint = "|".join((_PIPELINE_VERSION, language, str(min_token_length), ",".join(sorted(self._stopwords))))
        self._signature = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()

    @property
    def signature(self) -> str:
        """Stable digest of this preprocessor's behaviour (used for cache keys)."""
        return self._signature

    @staticmethod
    def clean_text(text: str) -> str:
        """Normalise Unicode, lowercase, and strip e-mails, URLs and control chars.

        Args:
            text: Raw input string.

        Returns:
            Cleaned lowercase string (possibly empty).
        """
        normalised = unicodedata.normalize("NFKC", text).lower()
        normalised = _EMAIL_RE.sub(" ", normalised)
        normalised = _URL_RE.sub(" ", normalised)
        return "".join(ch if (ch.isprintable() or ch.isspace()) else " " for ch in normalised)

    def tokenize(self, text: str) -> List[str]:
        """Tokenise text and strip punctuation, without removing stop words.

        Args:
            text: Raw input string.

        Returns:
            Alphanumeric tokens in original order. Empty for invalid input.
        """
        if not isinstance(text, str) or not text.strip():
            return []

        cleaned = self.clean_text(text)
        if not cleaned.strip():
            return []

        try:
            raw_tokens = word_tokenize(cleaned, language=self._language)
        except (LookupError, OSError):
            logger.warning("NLTK tokenizer unavailable; falling back to whitespace split.")
            raw_tokens = cleaned.split()

        tokens: List[str] = []
        for raw in raw_tokens:
            # Splitting on non-alphanumerics handles hyphens, contractions
            # ("n't", "'s") and residual punctuation in a single pass.
            for piece in _ALNUM_RE.findall(raw):
                if piece.isdigit() or len(piece) >= self._min_token_length:
                    tokens.append(piece)
        return tokens

    def _lemmatize_uncached(self, token: str) -> str:
        """Lemmatise as a verb, then as a noun (e.g. 'resetting' -> 'reset')."""
        return self._lemmatizer.lemmatize(self._lemmatizer.lemmatize(token, pos="v"), pos="n")

    def preprocess_tokens(self, text: str) -> List[str]:
        """Run the full pipeline and return the resulting tokens.

        Args:
            text: Raw input string.

        Returns:
            Lemmatised tokens with stop words removed; empty for invalid or
            fully filtered input.
        """
        result: List[str] = []
        for token in self.tokenize(text):
            if token in self._stopwords:
                continue
            lemma = self._lemmatize_cached(token)
            if lemma in self._stopwords or not lemma:
                continue
            result.append(lemma)
        return result

    def preprocess(self, text: str) -> str:
        """Run the full pipeline and return a space-joined token string.

        Args:
            text: Raw input string.

        Returns:
            Normalised text suitable for vectorisation (may be empty).
        """
        return " ".join(self.preprocess_tokens(text))

    __call__ = preprocess
