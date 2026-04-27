# Information Retrieval Pipeline

A modular Python project for:
- crawling web pages,
- preprocessing text with multiple pipelines,
- building persisted inverted indexes,
- running TF-IDF or Boolean retrieval.

The project now runs primarily through one interactive entry point.

## Quick Start

Install dependencies:

```bash
pip install beautifulsoup4 requests simplemma
```

Run the app:

```bash
python main.py
```

This opens an interactive menu where you can:
1. run crawler (foreground or background),
2. run preprocessing + indexing,
3. run indexing from existing preprocessed docs,
4. run retrieval.

## Current Workflow

### 1. Crawler

- Crawls site content and stores JSONL output in `data/crawler/crawled_pages.json`.
- Background mode writes logs to `data/crawler/crawler.log`.

### 2. Preprocessing + Indexing

In interactive mode, preprocessing now:
- lets you select input source file from `data/crawler` or type a custom path,
- lets you choose document language (`cs`, `sk`, `en`),
- lets you choose one or more pipelines,
- automatically builds matching index files after preprocessing.

Generated outputs:
- docs: `data/preprocessed/docs_<pipeline>[ _<lang> ].jsonl`
- vocab: `data/preprocessed/vocab_<pipeline>[ _<lang> ].txt`
- index: `data/index/inverted_index_<pipeline>[ _<lang> ].json`

Notes:
- Czech (`cs`) remains default.
- For Czech baseline, compatibility `data/preprocessed/vocab.txt` is still produced.

### 3. Indexing Only

Indexing-only mode expects already preprocessed docs (`docs_*.jsonl`).

### 4. Retrieval

Interactive retrieval lets you choose a persisted index and then a method:
- TF-IDF (ranked results with scores)
- Boolean (`AND`, `OR`, `NOT`)

Boolean mode now uses the real indexed term space (from persisted index postings), not placeholder text.

Standalone Boolean CLI also supports reading queries from a plain text file (one query per line):

```bash
python -m retrieval.boolean_search path/to/collection.txt --query-file path/to/boolean_queries.txt
```

## Inverted Index Format (Example)

Persisted index files are stored in:
- `data/index/inverted_index_<pipeline>[ _<lang> ].json`

Current JSON structure is:

```json
{
    "meta": {
        "pipeline": "baseline",
        "source_file": ".../data/preprocessed/docs_baseline.jsonl",
        "doc_count": 81730,
        "documents_sha256": "..."
    },
    "index": {
        "postings": {
            "term_a": {
                "doc_1": 3,
                "doc_25": 1
            }
        },
        "doc_term_freqs": {
            "doc_1": {
                "term_a": 3,
                "term_b": 2
            }
        },
        "doc_freq": {
            "term_a": 120
        },
        "idf": {
            "term_a": 2.34567
        },
        "doc_vectors": {
            "doc_1": {
                "term_a": 4.12345,
                "term_b": 2.00000
            }
        },
        "doc_norms": {
            "doc_1": 6.78901
        },
        "num_docs": 81730
    }
}
```

Note: New indexes are persisted in a compact format (`index_format=compact-v2`) to reduce file size.
In compact mode, only core structures are stored (`postings`, `doc_freq`, `idf`, `num_docs`),
and `doc_term_freqs` / `doc_vectors` / `doc_norms` are reconstructed on load.
Older full-format indexes remain supported.

Field types:
- `meta`: `dict`
- `meta.pipeline`: `str`
- `meta.source_file`: `str`
- `meta.doc_count`: `int`
- `meta.documents_sha256`: `str`
- `index.postings`: `dict[str, dict[str, int]]` (term -> doc_id -> raw TF)
- `index.doc_term_freqs`: `dict[str, dict[str, int]]` (doc_id -> term -> raw TF)
- `index.doc_freq`: `dict[str, int]` (term -> DF)
- `index.idf`: `dict[str, float]` (term -> IDF)
- `index.doc_vectors`: `dict[str, dict[str, float]]` (doc_id -> term -> TF-IDF weight)
- `index.doc_norms`: `dict[str, float]` (doc_id -> L2 norm)
- `index.num_docs`: `int`

Size notes:
- Index files can be large because both raw frequencies and TF-IDF vectors are persisted.
- In this repository, `data/index/inverted_index_baseline.json` is currently around 664 MB on disk.
- Interactive retrieval shows the selected index file size before query mode starts.

## Query Preprocessing Consistency

Query preprocessing is aligned with the index pipeline:
- pipeline type is read from index metadata (or selected file name),
- language suffix (`_cs`, `_sk`, `_en`) is resolved,
- the same preprocessing pipeline/language combination is applied to queries.

This applies to both TF-IDF and Boolean query paths.

## Project Structure

```text
.
├── main.py
├── app_config.py
├── interactive.py
├── runner.py
├── crawler/
│   ├── crawler.py
│   └── README.md
├── preprocessing/
│   ├── config.py
│   ├── language_config.py
│   ├── dataset.py
│   ├── orchestration.py
│   ├── preprocess.py
│   ├── tokenizer.py
│   ├── main.py
│   └── README.md
├── indexing/
│   └── main.py
├── retrieval/
│   ├── tfidf.py
│   ├── tfidf_search.py
│   ├── boolean.py
│   ├── query_interface.py
│   ├── dataset.py
│   ├── workflow.py
│   ├── reporting.py
│   ├── scoring.py
│   └── boolean_search.py
└── data/
    ├── crawler/
    ├── preprocessed/
    └── index/
```

## Configuration

### Global config

See `app_config.py`:
- `ROOT`
- `CRAWLER_SCRIPT`
- `CRAWLER_DATA_FILE`
- `CRAWLER_LOG_FILE`
- `PREPROCESSED_DIR`
- `INDEX_DIR`
- `PIPELINE_OPTIONS`

### Crawler config

Tune crawler settings directly in `crawler/crawler.py` (request delay, max URLs, robots/sitemap handling, URL filtering).

### Preprocessing config

See `preprocessing/config.py` for pipeline definitions.

Language resources are centralized in:
- `preprocessing/language_config.py`

## Legacy/Standalone Module Notes

Some submodules still contain standalone `main()` functions (for direct module execution), but the intended user flow is interactive via `python main.py`.

## Troubleshooting

- If retrieval finds no index files, run preprocessing + indexing first.
- If Boolean returns no matches for expected terms, verify you selected the intended index file and language variant.
- If a query has terms unseen in the selected index, both TF-IDF and Boolean may return no results.
