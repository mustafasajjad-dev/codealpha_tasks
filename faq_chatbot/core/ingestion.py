"""Dataset ingestion layer.

This module is intentionally free of any NLP or vectorisation logic. Its sole
responsibility is to read a JSON file from disk, validate its structure, and
return immutable :class:`FAQEntry` records for downstream consumers.

Supported JSON layouts::

    [ {"question": "...", "answer": "..."}, ... ]

    {"metadata": {...}, "faqs": [ {"question": "...", "answer": "..."}, ... ]}

Optional per-record fields are ``category`` (str) and ``keywords`` (list of
str). Keywords are appended to the indexed text to widen vocabulary coverage
(synonyms, abbreviations, common misspellings) without altering the displayed
question.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Set, Tuple, Union

from .exceptions import DatasetError

logger = logging.getLogger(__name__)

_HASH_CHUNK_BYTES = 1 << 16


@dataclass(frozen=True, slots=True)
class FAQEntry:
    """A single validated FAQ record.

    Attributes:
        faq_id: Zero-based position of the record in the validated dataset.
        question: Canonical question text shown to the user.
        answer: Answer text returned when this entry is matched.
        category: Free-form topical label used for display and analytics.
        keywords: Extra terms indexed alongside the question.
    """

    faq_id: int
    question: str
    answer: str
    category: str = "General"
    keywords: Tuple[str, ...] = ()

    @property
    def index_text(self) -> str:
        """Return the text that is vectorised for this entry."""
        return " ".join((self.question, *self.keywords))


def _parse_record(record: Any, faq_id: int) -> Optional[FAQEntry]:
    """Validate one raw JSON record.

    Args:
        record: Arbitrary decoded JSON value.
        faq_id: Identifier to assign if the record is valid.

    Returns:
        A :class:`FAQEntry`, or ``None`` when the record is malformed.
    """
    if not isinstance(record, dict):
        return None

    question = record.get("question")
    answer = record.get("answer")
    if not isinstance(question, str) or not isinstance(answer, str):
        return None

    question, answer = question.strip(), answer.strip()
    if not question or not answer:
        return None

    category = record.get("category")
    category = category.strip() if isinstance(category, str) and category.strip() else "General"

    raw_keywords = record.get("keywords", [])
    keywords: Tuple[str, ...] = ()
    if isinstance(raw_keywords, list):
        keywords = tuple(k.strip() for k in raw_keywords if isinstance(k, str) and k.strip())

    return FAQEntry(
        faq_id=faq_id,
        question=question,
        answer=answer,
        category=category,
        keywords=keywords,
    )


def load_faqs(path: Union[str, Path]) -> List[FAQEntry]:
    """Load and validate the FAQ dataset from a JSON file.

    Malformed individual records and duplicate questions are skipped with a
    warning; the load only fails when the file itself is unusable or no valid
    record remains.

    Args:
        path: Location of the JSON dataset.

    Returns:
        A non-empty list of validated entries.

    Raises:
        DatasetError: If the file is missing, unreadable, not valid JSON, has
            an unsupported structure, or contains no valid records.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise DatasetError(f"FAQ dataset not found: {file_path}")

    try:
        raw_text = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise DatasetError(f"Unable to read FAQ dataset '{file_path}': {exc}") from exc

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise DatasetError(f"FAQ dataset '{file_path}' is not valid JSON: {exc}") from exc

    records = payload.get("faqs") if isinstance(payload, dict) else payload
    if not isinstance(records, list) or not records:
        raise DatasetError(
            f"FAQ dataset '{file_path}' must be a non-empty JSON array or an object "
            "with a non-empty 'faqs' array."
        )

    entries: List[FAQEntry] = []
    seen_questions: Set[str] = set()
    for position, record in enumerate(records):
        entry = _parse_record(record, faq_id=len(entries))
        if entry is None:
            logger.warning("Skipping malformed FAQ record at index %d.", position)
            continue
        normalised = " ".join(entry.question.lower().split())
        if normalised in seen_questions:
            logger.warning("Skipping duplicate FAQ question at index %d: %r", position, entry.question)
            continue
        seen_questions.add(normalised)
        entries.append(entry)

    if not entries:
        raise DatasetError(f"FAQ dataset '{file_path}' contains no valid records.")

    logger.info("Loaded %d FAQ entries from %s.", len(entries), file_path)
    return entries


def dataset_fingerprint(path: Union[str, Path]) -> str:
    """Compute a SHA-256 digest of the dataset file contents.

    Args:
        path: Location of the dataset file.

    Returns:
        Hex digest identifying the exact file contents.

    Raises:
        DatasetError: If the file cannot be read.
    """
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
                digest.update(chunk)
    except OSError as exc:
        raise DatasetError(f"Unable to fingerprint dataset '{path}': {exc}") from exc
    return digest.hexdigest()
