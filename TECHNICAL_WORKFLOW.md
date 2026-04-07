# Technical Workflow Documentation

## 1. Runtime Entry

- File: `main.py`
- Entrypoint function: `main()`
- Runtime mode: always interactive
- Behavior: calls `interactive_mode()` from `interactive.py`

Execution chain:
1. `main.main()`
2. `interactive.interactive_mode()`

---

## 2. Interactive Orchestration Layer

- File: `interactive.py`

Core orchestrator helpers:
- `run_stage(name, fn)`
- `ask_input()`, `ask_yes_no()`, `ask_yes_no_or_nav()`
- `choose_from_list()`

Top-level menu dispatcher:
- `interactive_mode()`

Menu handlers:
- Crawler: `run_crawler_interactive()`
- Preprocessing + Indexing: `run_preprocessing_interactive()`
- Indexing only: `run_indexing_interactive()`
- Retrieval: `run_retrieval_interactive()`

Selection and navigation helpers:
- Input source selection: `choose_preprocessing_input_path()`
- Language selection: `choose_language()`
- Pipeline selection: `choose_pipelines(language)`
- Preprocessed docs selection: `choose_preprocessed_docs_file()`
- Index selection: `list_index_files()` + `choose_from_list()`

---

## 3. Crawler Stage

Primary crawl implementation:
- File: `crawler/crawler.py`
- Main callable: `run_crawler()`
- Crawl loop: `crawl()`

State and persistence:
- `save_state()`, `load_state()`
- Output sink: `store_data(url, content)`
- Output file: `data/crawler/crawled_pages.json` (JSONL)
- State file: `data/crawler/crawler_state.json`

Background execution:
- File: `runner.py`
- Function: `run_crawler_background()`
- Uses `subprocess.Popen` to spawn crawler and append logs to `data/crawler/crawler.log`

---

## 4. Preprocessing Stage

Stage implementation:
- File: `preprocessing/main.py`
- Main stage function: `run_preprocessing_stage(...)`

Input processing:
- `load_records(input_path)` from `preprocessing/dataset.py`
- Text key detection: `detect_text_keys(records)`
- Document normalization: `normalize_docs(records, selected_key)`

Pipeline building:
- File: `preprocessing/config.py`
- `build_pipelines(language)`
- Pipeline names: `PIPELINE_NAMES`

Language resources:
- File: `preprocessing/language_config.py`
- Functions:
  - `normalize_language_code(language)`
  - `get_stopwords(language)`

Pipeline execution:
- File: `preprocessing/orchestration.py`
- `parse_pipeline_selection(raw_selection)`
- `process_pipeline(pipeline_name, pipeline, raw_docs, tokenizer, progress_every)`

Tokenization and preprocessing operators:
- Tokenizer: `RegexMatchTokenizer` from `preprocessing/tokenizer.py`
- Preprocessors in `preprocessing/preprocess.py`:
  - `LowercasePreprocessor`
  - `RemoveTokenTypesPreprocessor`
  - `StopwordPreprocessor`
  - `MinLengthPreprocessor`
  - `StemmingPreprocessor`
  - `LemmatizationPreprocessor`
  - `RemoveDiacriticsPreprocessor`

Preprocessing artifacts:
- Docs: `data/preprocessed/docs_<pipeline>[ _<lang> ].jsonl`
- Vocab: `data/preprocessed/vocab_<pipeline>[ _<lang> ].txt`
- Czech baseline compat vocab: `data/preprocessed/vocab.txt`

Auto-index coupling:
- In `run_preprocessing_stage()`, indexing is invoked unless `skip_index=True`
- Calls `indexing.main.run_indexing_stage(...)` for each selected pipeline variant

---

## 5. Indexing Stage

Index builder:
- File: `indexing/main.py`
- Stage function: `run_indexing_stage(pipeline, input_path, output_path)`

Input:
- Preprocessed docs JSONL
- Loader: `load_preprocessed_docs(input_path)`

Index model:
- File: `retrieval/tfidf.py`
- Class: `InvertedIndex`
- Build call: `index.build(documents, WhitespacePreprocessor())`

Output payload:
- `meta` section:
  - `pipeline`
  - `source_file`
  - `doc_count`
  - `documents_sha256`
- `index` section:
  - serialized inverted index data

Persisted file:
- `data/index/inverted_index_<pipeline>.json`

---

## 6. Retrieval Stage (Interactive)

Interactive retrieval loop:
- File: `retrieval/query_interface.py`
- Entrypoint used by interactive app: `run_interactive_query_loop(index_file, pipeline, doc_texts=None)`

Index loading:
- Loads selected persisted index JSON
- Resolves pipeline from index metadata: `meta.pipeline` fallback to passed `pipeline`

Search method selection:
- `_ask_search_method()`
- Modes:
  - TF-IDF
  - Boolean

### 6.1 Query Preprocessing Alignment

Shared query preprocessor class:
- `PipelineQueryPreprocessor`

Important methods:
- `_resolve_pipeline_and_language(pipeline_name)`
  - Handles base names and suffixed names (`*_cs`, `*_sk`, `*_en`)
- `tokenize(text)`
  - Tokenizes and preprocesses queries with matching pipeline and language

This ensures query normalization follows the same pipeline family as the indexed docs.

### 6.2 TF-IDF Path

- Scorer: `CosineScorer` from `retrieval/tfidf.py`
- Search call: `tfidf_scorer.search(query)`
- Ranking output: scored `(doc_id, similarity)` list

### 6.3 Boolean Path

- Boolean structures from `retrieval/boolean.py`
  - `BooleanIndex`
  - `BooleanScorer`

Workflow in interactive mode:
1. Build Boolean index from persisted TF-IDF postings:
   - `_build_boolean_index_from_tfidf_docs(tfidf_index)`
2. Use `BooleanScorer(boolean_index, query_preprocessor)`
3. Evaluate query tokens/operators (`AND`, `OR`, `NOT`)

Important detail:
- Boolean mode uses the same query preprocessing pipeline class as TF-IDF mode.

---

## 7. Retrieval Stage (Programmatic / CLI-compatible module)

- File: `retrieval/tfidf_search.py`
- Main function: `run_retrieval_stage(...)`

Index-query mode:
- `run_index_query_mode(index_path, pipeline_name, query, top_k)`

Legacy collection-file mode:
- Uses `run_collection(file_path)` from `retrieval/workflow.py`
- Triggered when file arguments are provided to `run_retrieval_stage`

---

## 8. Data Contracts

Crawler output record (`crawled_pages.json` JSONL):
- Typical fields:
  - `url`
  - `title`
  - `author`
  - `topic`
  - `publication_date`
  - `hashed_content`
  - `article_text`
  - `scraped_at`

Preprocessed docs record (`docs_*.jsonl`):
- `doc_id`
- `url`
- `tokens` (list of normalized tokens)
- `normalized_text` (space-joined tokens)

Index JSON (`inverted_index_*.json`):
- `meta` + `index`

---

## 9. File/Function Map (Quick Reference)

- `main.py`
  - `main`
- `interactive.py`
  - `interactive_mode`
  - `run_crawler_interactive`
  - `run_preprocessing_interactive`
  - `run_indexing_interactive`
  - `run_retrieval_interactive`
- `runner.py`
  - `run_crawler_background`
- `crawler/crawler.py`
  - `run_crawler`
  - `crawl`
- `preprocessing/main.py`
  - `run_preprocessing_stage`
- `preprocessing/config.py`
  - `build_pipelines`
- `preprocessing/language_config.py`
  - `normalize_language_code`
  - `get_stopwords`
- `indexing/main.py`
  - `run_indexing_stage`
- `retrieval/query_interface.py`
  - `run_interactive_query_loop`
  - `PipelineQueryPreprocessor`
- `retrieval/tfidf.py`
  - `InvertedIndex`
  - `CosineScorer`
- `retrieval/boolean.py`
  - `BooleanIndex`
  - `BooleanScorer`
