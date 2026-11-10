# University FAQ Retrieval Chatbot

A production-style FAQ chatbot built on **classical NLP and vector space modelling**:
NLTK preprocessing -> TF-IDF (unigrams + bigrams) -> cosine similarity -> confidence-gated
answers, served through a Streamlit chat UI. No neural models, no external APIs.

## Folder structure

```
faq_chatbot/
├── app.py                  # Streamlit UI (presentation only)
├── requirements.txt
├── README.md
├── data/
│   └── faqs.json           # 23 University IT / Admissions FAQs
└── core/
    ├── __init__.py         # Public API
    ├── exceptions.py       # Typed exception hierarchy
    ├── ingestion.py        # JSON loading, validation, dataset fingerprinting
    ├── preprocessor.py     # NLTK cleaning, tokenisation, stop words, lemmatisation
    └── engine.py           # TF-IDF index, caching, cosine retrieval, fallback logic
```

## Setup

Requires **Python 3.10+**.

```bash
cd faq_chatbot
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

NLTK data (`punkt`/`punkt_tab`, `wordnet`, `omw-1.4`, `stopwords`) is **downloaded automatically**
on first run. If your environment is offline, install it manually:

```bash
python -m nltk.downloader punkt punkt_tab wordnet omw-1.4 stopwords
```

## Programmatic use

```python
from pathlib import Path
from core import EngineConfig, FAQEngine, TextPreprocessor

engine = FAQEngine(
    EngineConfig(dataset_path=Path("data/faqs.json"), confidence_threshold=0.25),
    TextPreprocessor(),
)
result = engine.answer("i forgot my password")
print(result.answer, result.confidence, result.matched_question)
```

## Design notes

| Concern | Decision |
| --- | --- |
| **Separation of concerns** | Ingestion, preprocessing, retrieval and UI are separate modules. The engine depends only on a `TextProcessor` protocol, so the preprocessor can be swapped or mocked. |
| **Picklable vector space** | Documents are preprocessed *before* vectorisation, so `TfidfVectorizer` holds no custom callables and serialises cleanly. |
| **Caching** | The fitted vectoriser and document matrix are persisted to `.cache/` under a SHA-256 key covering the dataset bytes, preprocessor signature, n-gram range and scikit-learn version. Stale, corrupt or unpicklable caches are detected and rebuilt; write failures never break start-up. |
| **Latency** | Document vectors are precomputed; a query is one sparse transform plus one sparse product. An LRU cache (default 512 entries) serves repeated queries without recomputation. Expect well under a millisecond for datasets of this size once warm. |
| **Confidence** | Score is the cosine similarity of the top candidate. Bands: HIGH (>= 0.60), MEDIUM (>= threshold), LOW (below). The result also exposes the margin between the top two candidates, which helps flag ambiguous matches. |
| **Fallback** | Below the threshold the bot asks the user to rephrase and lists the closest topics. Blank input and queries that reduce to zero tokens each get their own message. |
| **Defensive handling** | Missing/invalid dataset, malformed records, duplicate questions, `None`/non-string/blank queries, over-long queries, missing NLTK data, unwritable or corrupt cache - all handled with typed exceptions or graceful degradation. |

## Configuration

All options live in `core.engine.EngineConfig`:

| Option | Default | Purpose |
| --- | --- | --- |
| `confidence_threshold` | `0.25` | Minimum cosine similarity to answer (also adjustable live in the sidebar) |
| `high_confidence_threshold` | `0.60` | Boundary for the HIGH band |
| `suggestion_floor` | `0.10` | Minimum score for "closest topics" suggestions in fallbacks |
| `ngram_range` | `(1, 2)` | Unigrams + bigrams |
| `top_k` | `3` | Candidates retained and shown for observability |
| `query_cache_size` | `512` | LRU capacity (0 disables) |
| `cache_dir` | `None` | Where to persist the vector space (`app.py` uses `.cache/`) |
| `fallback_message` etc. | see code | Customisable reply texts |

## Adding or changing FAQs

Edit `data/faqs.json` (a list, or an object with a `faqs` list). Each record needs `question` and
`answer`; `category` and `keywords` are optional. **Keywords** are indexed with the question but never
displayed - use them for synonyms and abbreviations ("wifi", "2fa", "dorm") since classical TF-IDF
cannot bridge vocabulary gaps on its own. The cache invalidates automatically when the file changes.

## Tuning tips

* Use the sidebar slider and the *Retrieval details* expander to inspect scores on real queries.
* Raise the threshold for precision (fewer wrong answers), lower it for recall.
* A small margin between the top two candidates usually means overlapping FAQs - consider merging
  them or adding discriminating keywords.

## Limitations

TF-IDF matches on shared vocabulary, not meaning. Paraphrases with no word overlap ("can't get online" vs
"Wi-Fi") need keyword coverage in the dataset. The sample dataset describes a fictional university.

## Threshold calibration

The default threshold is `0.25`, not `0.35`. Short user queries compared against a keyword-augmented index
yield modest cosine scores even for correct matches (for example, "application deadline" scores about 0.3),
while off-topic queries score near 0.0. A value of 0.35 or higher is a stricter setting that favours
precision. Calibrate against real traffic with the sidebar slider.
