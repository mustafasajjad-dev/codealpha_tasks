"""FAQ retrieval chatbot core package.

Layers (each independently replaceable):
    * :mod:`core.ingestion`    - dataset loading and validation
    * :mod:`core.preprocessor` - NLTK text normalisation
    * :mod:`core.engine`       - TF-IDF vector space and cosine retrieval
"""

from .engine import (
    ConfidenceLevel,
    EngineConfig,
    FAQEngine,
    RankedCandidate,
    ResultStatus,
    RetrievalResult,
    TextProcessor,
)
from .exceptions import (
    ConfigurationError,
    DatasetError,
    EngineInitializationError,
    FAQChatbotError,
    NLTKResourceError,
)
from .ingestion import FAQEntry, dataset_fingerprint, load_faqs
from .preprocessor import TextPreprocessor, ensure_nltk_resources

__all__ = [
    "ConfidenceLevel",
    "ConfigurationError",
    "DatasetError",
    "EngineConfig",
    "EngineInitializationError",
    "FAQChatbotError",
    "FAQEngine",
    "FAQEntry",
    "NLTKResourceError",
    "RankedCandidate",
    "ResultStatus",
    "RetrievalResult",
    "TextPreprocessor",
    "TextProcessor",
    "dataset_fingerprint",
    "ensure_nltk_resources",
    "load_faqs",
]

__version__ = "1.0.0"
