# Information Retrieval Pipeline

A modular Python pipeline for web crawling, text preprocessing, and TF-IDF based information retrieval.

## Project Structure

```
.
├── main.py                 # Unified CLI entry point
├── .gitignore             # Ignore cache and generated data
├── README.md              # This file
│
├── crawler/
│   ├── crawler.py         # Web crawler with state persistence
│   └── README.md
│
├── preprocessing/
│   ├── main.py            # CLI entry point
│   ├── cli.py             # Argument parsing
│   ├── config.py          # Pipeline definitions & constants
│   ├── dataset.py         # I/O, document handling, text key detection
│   ├── orchestration.py   # Pipeline execution & vocabulary building
│   ├── preprocess.py      # Preprocessing filters (stopwords, stemming, etc.)
│   ├── tokenizer.py       # Tokenization & token classification
│   └── README.md
│
├── indexing/
│   ├── __init__.py
│   └── main.py            # Builds persisted inverted index from preprocessed docs
│
├── retrieval/
│   ├── __init__.py        # Package marker
│   ├── dataset.py         # Shared collection parsing & tokenization
│   ├── reporting.py       # Shared ASCII table rendering helpers
│   ├── tfidf.py           # TF-IDF index/scorer implementation
│   ├── scoring.py         # Backward-compatible TF-IDF exports
│   ├── workflow.py        # TF-IDF workflow orchestration
│   ├── tfidf_search.py    # TF-IDF search entry point
│   ├── boolean.py         # Boolean index/scorer implementation
│   ├── boolean_search.py  # Boolean search entry point
│   ├── test1.txt          # Example collection
│   ├── test2.txt          # Example collection
│   └── README.md
│
├── data/
│   ├── crawler/
│   │   ├── crawled_pages.json
│   │   └── crawler_state.json
│   └── preprocessed/
│       ├── vocab_baseline.txt
│       ├── docs_baseline.jsonl
│       ├── vocab_stemming.txt
│       └── ...
│   └── index/
│       └── inverted_index_baseline.json
│
└── app_config.py          # Root-level pipeline config
```

## Quick Start

### Installation

No external setup required beyond Python 3.10+ and pip dependencies:

```bash
pip install beautifulsoup4 requests simplemma
```

### 1. Crawl the Web

Start the web crawler:

```bash
python crawler/crawler.py
```

Or use through main dispatcher:

```bash
python main.py crawler
```

**Options:**
- The crawler loads state from `data/crawler/crawler_state.json` if it exists
- First run fetches `robots.txt` and sitemap
- Pages are saved to `data/crawler/crawled_pages.json` (JSONL format)
- Configurable: `REQUEST_DELAY`, `MAX_URLS`, `DISALLOWED_PATHS` in `crawler/crawler.py`

### 2. Preprocess Text

Run preprocessing pipelines interactively:

```bash
python preprocessing/main.py
```

Or specify options:

```bash
python preprocessing/main.py \
  --input data/crawler/crawled_pages.json \
  --text-key article_text \
  --pipelines baseline stemming lemmatization \
  --output-dir data/preprocessed
```

**Available pipelines:**
- `baseline` — lowercase, remove punctuation/stopwords, min length 2
- `stemming` — + Czech stemming
- `lemmatization` — + Czech lemmatization (via simplemma)
- `stemming_no_diacritics` — + remove accents
- `lemmatization_no_diacritics` — + remove accents

**Detect available text keys from data:**

```bash
python preprocessing/main.py --list-text-keys
```

**Output:**
- Vocabulary files: `data/preprocessed/vocab_<pipeline>.txt` (term frequency sorted)
- Compatibility vocab: `data/preprocessed/vocab.txt` (baseline)
- Normalized docs per pipeline: `data/preprocessed/docs_<pipeline>.jsonl`

### 3. Indexing

Build an inverted index from normalized preprocessed docs:

```bash
python indexing/main.py --pipeline baseline
```

Or through the main dispatcher:

```bash
python main.py indexing --index-pipeline baseline
```

**Output:**
- Persisted index: `data/index/inverted_index_<pipeline>.json`

### 4. Information Retrieval Search

Run retrieval against persisted index:

```bash
python retrieval/tfidf_search.py --pipeline baseline --query "turistika tatry"
```

Or via the root CLI:

```bash
python main.py retrieval --index-pipeline baseline --query "turistika tatry" --top-k 10
```

Legacy compatibility mode (collection files with `d*/q*`) is still available:

```bash
python retrieval/tfidf_search.py retrieval/test1.txt retrieval/test2.txt
```

**Import format for collections:**
```
d1: Document 1 text here
d2: Document 2 text here
q1: Query 1 text here
q2: Query 2 text here
```

**Output:**
- Term DF/IDF table
- Document vectors with per-term TF-IDF breakdown
- Query vectors and ranked document results

## Module Execution

All entry points support **both script and module execution**:

```bash
# Script mode
python pipeline.py --help

# Module mode
python -m pipeline.main --help
```

### Main Entry Point

Unified dispatcher for the full pipeline:

```bash
python main.py -h
```

**Stages:**
- `interactive` (default) — menu-driven pipeline
- `crawler` — run web crawler
- `preprocessing` — run text preprocessing
- `indexing` — build persisted inverted index
- `retrieval` — query persisted index (or run legacy file mode)
- `all` — crawler → preprocessing → indexing → retrieval

Example:

```bash
python main.py all --continue-on-error
```

### Crawler

```bash
python crawler/crawler.py
python -m crawler.crawler
```

### Preprocessing

```bash
python preprocessing/main.py
python -m preprocessing.main

# With options:
python preprocessing/main.py --input data/crawler/crawled_pages.json \
                              --text-key article_text \
                              --pipelines all \
                              --output-dir data/preprocessed
```

### Information Retrieval

```bash
python retrieval/tfidf_search.py
python -m retrieval.tfidf_search

# With custom collection:
python retrieval/tfidf_search.py my_collection.txt
```

## Configuration

### Global Config ([app_config.py](app_config.py))

```python
ROOT = Path(__file__).resolve().parent
CRAWLER_SCRIPT = ROOT / "crawler" / "crawler.py"
PREPROCESSING_SCRIPT = ROOT / "preprocessing" / "main.py"
RETRIEVAL_SCRIPT = ROOT / "retrieval" / "tfidf_search.py"

DEFAULT_INPUT = ROOT / "data" / "crawler" / "crawled_pages.json"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "preprocessed"
```

### Crawler Config

Edit `crawler/crawler.py`:

```python
INIT_URL = 'https://www.interez.sk/'
ROBOTS_URL = 'https://www.interez.sk/robots.txt'
SITEMAP_URL = 'https://www.interez.sk/sitemap.xml'
MAX_URLS = 1000
REQUEST_DELAY = 3  # seconds
DISALLOWED_PATHS = ['/profil/', '/kontakt/', '/o-nas/']
```

### Preprocessing Config

Edit `preprocessing/config.py`:

```python
PIPELINE_NAMES = [
    "baseline",
    "stemming",
    "lemmatization",
    "stemming_no_diacritics",
    "lemmatization_no_diacritics",
]
```

Stopwords: `preprocessing/preprocess.py` → `CZECH_STOPWORDS` set

## Workflow Examples

### Complete Pipeline (Crawler → Preprocessing → TF-IDF)

```bash
# Option 1: Interactive menu
python main.py interactive

# Option 2: Run all stages in sequence
python main.py all

# Option 3: With error tolerance
python main.py all --continue-on-error
```

### Preprocess Existing Crawled Data

```bash
python preprocessing/main.py \
  --input data/crawler/crawled_pages.json \
  --text-key article_text \
  --pipelines baseline stemming \
  --output-dir data/preprocessed
```

### Information Retrieval Search

1. Create a collection file (JSONL-style):

```
d1: First article about Python
d2: Second article about JavaScript
q1: What is Python?
q2: JavaScript frameworks
```

2. Run search:

```bash
python retrieval/tfidf_search.py my_collection.txt
```

3. Review output:
   - Term statistics (DF, IDF)
   - Document vector breakdowns
   - Query rankings

## Development

### Adding a New Preprocessing Step

1. Create a class in `preprocessing/preprocess.py`:

```python
class MyPreprocessor(TokenPreprocessor):
    def preprocess(self, token: Token, document: str) -> Token:
        token.processed_form = token.processed_form.upper()
        return token
```

2. Add to pipeline in `preprocessing/config.py`:

```python
BASE_STEPS.append(MyPreprocessor())
```

### Extending Extraction Logic

Edit `retrieval/dataset.py` (`CollectionParser`) or crawler extraction functions in `crawler/crawler.py`.

### Inverted Index Optimization (TF-IDF Retrieval)

The `retrieval/scoring.py` module uses a **dictionary-based inverted index** (`postings`) to optimize search performance. Instead of scoring every document in the collection, the `CosineScorer.search()` method:

1. **Extracts query terms** from the query text
2. **Uses the inverted index** to find candidate documents that contain at least one query term
3. **Scores only candidates** rather than all documents

This significantly improves efficiency for large collections by reducing unnecessary similarity computations.

**Key data structures in `InvertedIndex`:**
- `postings: Dict[str, Dict[str, int]]` — maps each term to documents containing it with term frequencies
- `doc_vectors, doc_norms` — precomputed TF-IDF vectors for cosine similarity
- `idf` — precomputed inverse document frequency for each term

### Running Tests

No test suite yet — consider adding pytest for:
- URL validation
- Tokenization
- Extraction accuracy
- Vector math

## Troubleshooting

**"No input files found for TF-IDF run"**
- Ensure `retrieval/test1.txt` and `retrieval/test2.txt` exist, or provide file paths
- Example: `python retrieval/tfidf_search.py my_file.txt`

**"Unable to determine text key automatically"**
- Your crawler output doesn't have `article_text`, `text`, or detectable keys
- Use `--list-text-keys` to see available keys
- Specify manually: `--text-key your_key`

**Crawler hangs or times out**
- Increase timeout in `crawler/crawler.py`: `timeout=10` → `timeout=30`
- Reduce `REQUEST_DELAY` (but be respectful to the server)

**"ModuleNotFoundError: No module named 'simplemma'"**
- Install: `pip install simplemma`

## License & Notes

This is an educational IR pipeline for coursework. 
- Respectful crawling: configurable delays, robots.txt support
- Stateful: resume crawls on crash
- Modular: each stage can run independently

---

**Questions?** Check individual README files in each module directory.
