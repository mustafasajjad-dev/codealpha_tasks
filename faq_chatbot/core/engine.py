"""TF-IDF / cosine-similarity FAQ retrieval engine.

The engine owns only vector-space concerns. Dataset parsing lives in
:mod:`core.ingestion` and text normalisation is injected through the
:class:`TextProcessor` protocol, so either can be replaced independently.

Performance model:
    * Document vectors are computed once at start-up (or restored from a
      fingerprinted on-disk cache) and held in a CSR matrix.
    * Each query costs one sparse transform plus one sparse-dense product over
      the FAQ matrix; an LRU cache makes repeated queries effectively free.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

import joblib
import numpy as np
import sklearn
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .exceptions import ConfigurationError, EngineInitializationError
from .ingestion import FAQEntry, dataset_fingerprint, load_faqs

logger = logging.getLogger(__name__)

_CACHE_SCHEMA_VERSION = "1"

DEFAULT_FALLBACK_MESSAGE = (
    "I'm sorry, I couldn't find a confident answer to that. "
    "Could you rephrase your question or add a few more details?"
)
DEFAULT_EMPTY_QUERY_MESSAGE = "Please type a question and I'll do my best to help."
DEFAULT_NO_TERMS_MESSAGE = (
    "I couldn't pick out any meaningful keywords in that message. "
    "Could you ask your question using a few more descriptive words?"
)


class TextProcessor(Protocol):
    """Structural interface the engine requires from a text preprocessor."""

    @property
    def signature(self) -> str:
        """Digest identifying the preprocessor's behaviour (for cache keys)."""
        ...

    def preprocess(self, text: str) -> str:
        """Return normalised, space-delimited tokens for ``text``."""
        ...


class ConfidenceLevel(str, Enum):
    """Human-readable confidence band for a retrieval result."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class ResultStatus(str, Enum):
    """Outcome category of a query."""

    MATCHED = "matched"
    LOW_CONFIDENCE = "low_confidence"
    EMPTY_QUERY = "empty_query"
    NO_USABLE_TERMS = "no_usable_terms"


@dataclass(frozen=True)
class EngineConfig:
    """Immutable configuration for :class:`FAQEngine`.

    Attributes:
        dataset_path: JSON dataset location.
        confidence_threshold: Minimum cosine similarity to accept a match.
        high_confidence_threshold: Similarity at/above which confidence is HIGH.
        suggestion_floor: Minimum similarity for a candidate to be suggested
            in a fallback response.
        ngram_range: Inclusive n-gram bounds for the vectoriser.
        top_k: Number of ranked candidates retained per query.
        max_query_chars: Longer queries are truncated defensively.
        query_cache_size: LRU capacity for ranked results (0 disables).
        cache_dir: Directory for the persisted vector space (None disables).
        fallback_message: Reply when similarity is below the threshold.
        empty_query_message: Reply for blank input.
        no_usable_terms_message: Reply when preprocessing removes every token.
    """

    dataset_path: Path
    confidence_threshold: float = 0.25
    high_confidence_threshold: float = 0.60
    suggestion_floor: float = 0.10
    ngram_range: Tuple[int, int] = (1, 2)
    top_k: int = 3
    max_query_chars: int = 1000
    query_cache_size: int = 512
    cache_dir: Optional[Path] = None
    fallback_message: str = DEFAULT_FALLBACK_MESSAGE
    empty_query_message: str = DEFAULT_EMPTY_QUERY_MESSAGE
    no_usable_terms_message: str = DEFAULT_NO_TERMS_MESSAGE

    def __post_init__(self) -> None:
        """Validate configuration values.

        Raises:
            ConfigurationError: If any value is out of range or inconsistent.
        """
        for name in ("confidence_threshold", "high_confidence_threshold", "suggestion_floor"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ConfigurationError(f"{name} must be within [0, 1]; got {value}.")
        if self.high_confidence_threshold < self.confidence_threshold:
            raise ConfigurationError("high_confidence_threshold must be >= confidence_threshold.")
        low, high = self.ngram_range
        if low < 1 or high < low:
            raise ConfigurationError(f"Invalid ngram_range {self.ngram_range}.")
        if self.top_k < 1:
            raise ConfigurationError("top_k must be >= 1.")
        if self.max_query_chars < 1:
            raise ConfigurationError("max_query_chars must be >= 1.")
        if self.query_cache_size < 0:
            raise ConfigurationError("query_cache_size must be >= 0.")


@dataclass(frozen=True)
class RankedCandidate:
    """A scored FAQ candidate."""

    faq_id: int
    question: str
    category: str
    score: float


@dataclass(frozen=True)
class RetrievalResult:
    """Full outcome of a query, including observability data.

    Attributes:
        status: Outcome category.
        answer: Text to show the user (FAQ answer or fallback message).
        confidence: Cosine similarity of the best candidate, in [0, 1].
        confidence_level: Banded confidence.
        matched_question: Best-scoring FAQ question, even for fallbacks.
        matched_category: Category of the best-scoring FAQ.
        margin: Score gap between the best and second-best candidates.
        threshold: Threshold applied to this query.
        candidates: Top-k ranked candidates.
        processed_query: Query after preprocessing.
        latency_ms: End-to-end retrieval latency in milliseconds.
        cache_hit: Whether the ranking was served from the query cache.
    """

    status: ResultStatus
    answer: str
    confidence: float = 0.0
    confidence_level: ConfidenceLevel = ConfidenceLevel.NONE
    matched_question: Optional[str] = None
    matched_category: Optional[str] = None
    margin: float = 0.0
    threshold: float = 0.0
    candidates: Tuple[RankedCandidate, ...] = field(default_factory=tuple)
    processed_query: str = ""
    latency_ms: float = 0.0
    cache_hit: bool = False

    @property
    def is_fallback(self) -> bool:
        """``True`` when no FAQ answer was returned."""
        return self.status is not ResultStatus.MATCHED


class FAQEngine:
    """Retrieves the best-matching FAQ answer using TF-IDF and cosine similarity.

    Args:
        config: Engine configuration.
        preprocessor: Text preprocessor. Defaults to a freshly built
            :class:`core.preprocessor.TextPreprocessor`.

    Raises:
        DatasetError: If the dataset is missing or invalid.
        NLTKResourceError: If the default preprocessor cannot obtain NLTK data.
        EngineInitializationError: If the vector space cannot be built.
    """

    def __init__(self, config: EngineConfig, preprocessor: Optional[TextProcessor] = None) -> None:
        if preprocessor is None:
            from .preprocessor import TextPreprocessor  # deferred: keeps NLTK optional for tests

            preprocessor = TextPreprocessor()

        self._config = config
        self._preprocessor: TextProcessor = preprocessor
        self._faqs: List[FAQEntry] = load_faqs(config.dataset_path)
        self._query_cache: "OrderedDict[str, Tuple[Tuple[int, float], ...]]" = OrderedDict()
        self._cache_lock = threading.Lock()

        self._vectorizer, self._matrix = self._load_or_build_index()
        self._warm_up()
        logger.info(
            "FAQ engine ready: %d documents, %d features.",
            self._matrix.shape[0],
            self._matrix.shape[1],
        )

    # ------------------------------------------------------------------ props
    @property
    def config(self) -> EngineConfig:
        """The active engine configuration."""
        return self._config

    @property
    def faq_count(self) -> int:
        """Number of indexed FAQ entries."""
        return len(self._faqs)

    @property
    def vocabulary_size(self) -> int:
        """Number of unigram/bigram features in the vector space."""
        return int(self._matrix.shape[1])

    @property
    def categories(self) -> List[str]:
        """Sorted distinct FAQ categories."""
        return sorted({faq.category for faq in self._faqs})

    def list_questions(self) -> List[str]:
        """Return all canonical FAQ questions in dataset order."""
        return [faq.question for faq in self._faqs]

    # ------------------------------------------------------------- index build
    def _fingerprint(self) -> str:
        """Digest of everything that influences the fitted vector space."""
        digest = hashlib.sha256()
        parts = (
            dataset_fingerprint(self._config.dataset_path),
            self._preprocessor.signature,
            repr(self._config.ngram_range),
            sklearn.__version__,
            _CACHE_SCHEMA_VERSION,
        )
        for part in parts:
            digest.update(part.encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()

    def _cache_path(self, fingerprint: str) -> Optional[Path]:
        """Return the cache file path, or ``None`` if caching is disabled."""
        if self._config.cache_dir is None:
            return None
        return Path(self._config.cache_dir) / f"tfidf_{fingerprint[:16]}.joblib"

    def _build_index(self) -> Tuple[TfidfVectorizer, sparse.csr_matrix]:
        """Fit a TF-IDF vectoriser over the preprocessed FAQ index texts.

        Documents are preprocessed *before* vectorisation, so the fitted
        vectoriser contains no custom callables and remains picklable.
        """
        documents = [self._preprocessor.preprocess(faq.index_text) for faq in self._faqs]
        if not any(doc.strip() for doc in documents):
            raise EngineInitializationError("Preprocessing removed all tokens from every FAQ entry.")

        vectorizer = TfidfVectorizer(
            ngram_range=self._config.ngram_range,
            token_pattern=r"(?u)\b\w+\b",
            sublinear_tf=True,
            norm="l2",
        )
        try:
            matrix = vectorizer.fit_transform(documents)
        except ValueError as exc:
            raise EngineInitializationError(f"Unable to fit TF-IDF vector space: {exc}") from exc
        return vectorizer, matrix.tocsr()

    def _load_cached_index(self, path: Path, fingerprint: str) -> Optional[Tuple[TfidfVectorizer, sparse.csr_matrix]]:
        """Restore a persisted vector space, validating it thoroughly.

        Returns:
            ``(vectorizer, matrix)`` when the cache is present and valid,
            otherwise ``None``.
        """
        if not path.is_file():
            return None
        try:
            payload: Any = joblib.load(path)
        except Exception as exc:  # noqa: BLE001 - unpickling can raise arbitrary errors
            logger.warning("Ignoring unreadable vector cache %s: %s", path, exc)
            return None

        if (
            not isinstance(payload, dict)
            or payload.get("fingerprint") != fingerprint
            or not isinstance(payload.get("vectorizer"), TfidfVectorizer)
            or not sparse.issparse(payload.get("matrix"))
            or payload["matrix"].shape[0] != len(self._faqs)
        ):
            logger.warning("Vector cache %s is stale or malformed; rebuilding.", path)
            return None

        logger.info("Restored vector space from cache %s.", path)
        return payload["vectorizer"], payload["matrix"].tocsr()

    def _persist_index(self, path: Path, fingerprint: str, vectorizer: TfidfVectorizer, matrix: sparse.csr_matrix) -> None:
        """Atomically write the vector space to disk; failures are non-fatal."""
        tmp_name: Optional[str] = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as tmp:
                tmp_name = tmp.name
            joblib.dump({"fingerprint": fingerprint, "vectorizer": vectorizer, "matrix": matrix}, tmp_name)
            os.replace(tmp_name, path)
            tmp_name = None
            logger.info("Persisted vector space to %s.", path)
        except Exception as exc:  # noqa: BLE001 - pickling/IO failures must not break startup
            logger.warning("Could not persist vector cache to %s: %s", path, exc)
        finally:
            if tmp_name is not None:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass

    def _load_or_build_index(self) -> Tuple[TfidfVectorizer, sparse.csr_matrix]:
        """Return the vector space from cache when valid, otherwise build it."""
        fingerprint = self._fingerprint()
        cache_path = self._cache_path(fingerprint)

        if cache_path is not None:
            restored = self._load_cached_index(cache_path, fingerprint)
            if restored is not None:
                return restored

        vectorizer, matrix = self._build_index()
        if cache_path is not None:
            self._persist_index(cache_path, fingerprint, vectorizer, matrix)
        return vectorizer, matrix

    def _warm_up(self) -> None:
        """Execute one throw-away scoring pass so first user query is fast."""
        probe = self._faqs[0].index_text
        self._score(self._preprocessor.preprocess(probe) or probe)

    # --------------------------------------------------------------- retrieval
    def _score(self, processed_query: str) -> Tuple[Tuple[int, float], ...]:
        """Compute the top-k (faq_index, cosine) pairs for a processed query."""
        query_vector = self._vectorizer.transform([processed_query])
        scores = cosine_similarity(query_vector, self._matrix).ravel()
        scores = np.clip(scores, 0.0, 1.0)
        order = np.argsort(-scores, kind="stable")[: self._config.top_k]
        return tuple((int(i), float(scores[i])) for i in order)

    def _rank(self, processed_query: str) -> Tuple[Tuple[Tuple[int, float], ...], bool]:
        """Return ranked scores, consulting the LRU cache first."""
        capacity = self._config.query_cache_size
        if capacity > 0:
            with self._cache_lock:
                cached = self._query_cache.get(processed_query)
                if cached is not None:
                    self._query_cache.move_to_end(processed_query)
                    return cached, True

        ranked = self._score(processed_query)

        if capacity > 0:
            with self._cache_lock:
                self._query_cache[processed_query] = ranked
                while len(self._query_cache) > capacity:
                    self._query_cache.popitem(last=False)
        return ranked, False

    def _level_for(self, score: float, threshold: float) -> ConfidenceLevel:
        """Map a similarity score to a confidence band."""
        if score >= self._config.high_confidence_threshold:
            return ConfidenceLevel.HIGH
        if score >= threshold:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW

    def _compose_fallback(self, candidates: Tuple[RankedCandidate, ...]) -> str:
        """Build the fallback reply, appending close-but-unconfident topics."""
        suggestions = [c for c in candidates if c.score >= self._config.suggestion_floor]
        if not suggestions:
            return self._config.fallback_message
        lines = [self._config.fallback_message, "", "Here are the closest topics I can help with:"]
        lines.extend(f"- {c.question}" for c in suggestions)
        return "\n".join(lines)

    def _terminal_result(self, status: ResultStatus, message: str, threshold: float, started: float) -> RetrievalResult:
        """Create a result for inputs that never reach similarity scoring."""
        return RetrievalResult(
            status=status,
            answer=message,
            threshold=threshold,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    def answer(self, query: str, threshold: Optional[float] = None) -> RetrievalResult:
        """Answer a user query.

        Args:
            query: Raw user text.
            threshold: Optional per-call override of the confidence threshold.

        Returns:
            A :class:`RetrievalResult`. Never raises for bad *input*; blank,
            non-string, or fully-filtered queries produce a fallback result.

        Raises:
            ConfigurationError: If ``threshold`` is outside ``[0, 1]``.
        """
        started = time.perf_counter()

        effective = self._config.confidence_threshold if threshold is None else threshold
        if not 0.0 <= effective <= 1.0:
            raise ConfigurationError(f"threshold must be within [0, 1]; got {effective}.")

        if not isinstance(query, str) or not query.strip():
            return self._terminal_result(ResultStatus.EMPTY_QUERY, self._config.empty_query_message, effective, started)

        processed = self._preprocessor.preprocess(query[: self._config.max_query_chars])
        if not processed:
            return self._terminal_result(ResultStatus.NO_USABLE_TERMS, self._config.no_usable_terms_message, effective, started)

        ranked, cache_hit = self._rank(processed)
        candidates = tuple(
            RankedCandidate(
                faq_id=self._faqs[idx].faq_id,
                question=self._faqs[idx].question,
                category=self._faqs[idx].category,
                score=score,
            )
            for idx, score in ranked
        )

        best = candidates[0]
        margin = best.score - candidates[1].score if len(candidates) > 1 else best.score
        matched = best.score >= effective and best.score > 0.0

        if matched:
            status, answer = ResultStatus.MATCHED, self._faqs[best.faq_id].answer
        else:
            status, answer = ResultStatus.LOW_CONFIDENCE, self._compose_fallback(candidates)

        return RetrievalResult(
            status=status,
            answer=answer,
            confidence=best.score,
            confidence_level=self._level_for(best.score, effective) if best.score > 0.0 else ConfidenceLevel.NONE,
            matched_question=best.question,
            matched_category=best.category,
            margin=margin,
            threshold=effective,
            candidates=candidates,
            processed_query=processed,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            cache_hit=cache_hit,
        )

    def explain(self, result: RetrievalResult) -> Dict[str, Any]:
        """Flatten a result into a JSON-friendly dict for logging/UI."""
        return {
            "status": result.status.value,
            "confidence": round(result.confidence, 4),
            "confidence_level": result.confidence_level.value,
            "matched_question": result.matched_question,
            "matched_category": result.matched_category,
            "margin": round(result.margin, 4),
            "threshold": result.threshold,
            "processed_query": result.processed_query,
            "latency_ms": round(result.latency_ms, 3),
            "cache_hit": result.cache_hit,
            "candidates": [
                {"question": c.question, "category": c.category, "score": round(c.score, 4)}
                for c in result.candidates
            ],
        }
