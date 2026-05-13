# Information Retrieval System

**Author:** Kevin Varchola

---

This document provides an overview of the Information Retrieval System project, including its features, performance metrics, and instructions on how to run it.

## Features

The project implements a complete Information Retrieval pipeline from scratch, completely avoiding external search platforms like Lucene, Solr, or Elasticsearch.

### Mandatory Features
- **Custom In-Memory Indexer:** A custom two-pass algorithm builds an inverted index, calculating TF and IDF distributions.
- **Vector Space Model (TF-IDF):** Implements log-normalized Term Frequency and Inverse Document Frequency with Cosine Similarity scoring.
- **Boolean Model:** Supports `AND`, `OR`, `NOT` operators.
- **Parentheses & Precedence:** Fully supports overriding operator precedence using parentheses `( )`.
- **Batch Evaluation:** Implements automated generation of `trec_eval` compatible 6-column run files (`qid iter docno rank sim run_id`).
- **Independent Indexing:** Supports independently indexing crawled web data and evaluation datasets.

### Extra Credit Features
- **Okapi BM25 Scoring Model:** Implemented alongside TF-IDF, utilizing probabilistic term frequency saturation ($k_1$) and document length normalization ($b$). This improved the Czech MAP score by ~80% over TF-IDF.
- **Custom Boolean Query Parser:** Built an entirely custom parser using the **Shunting-yard algorithm** to convert infix query notation (e.g., `A AND (B OR NOT C)`) to postfix notation, evaluated using native Python set mathematics.
- **File-based Indexing & Persistence:** The in-memory index is serialized to disk using a custom `compact-v2` JSON format. Redundant data (like heavy TF-IDF document vectors) are dropped during save and rapidly recomputed upon loading, saving hundreds of megabytes of disk space.
- **Multi-language NLP Support:** Language-aware pipelines for Czech, Slovak, and English, utilizing dedicated stopwords and morphological processing.
- **Stemming & Lemmatization:** Employs custom suffix-stripping stemmers and integrates `simplemma` for robust lemmatization.
- **HTML Tag Handling:** Safely sanitizes crawled markup utilizing a custom `RegexMatchTokenizer` before building the index.
- **Keyword-In-Context (KWIC) Highlighting:** Dynamic retrieval UI displaying snippets of matched documents with query terms distinctly highlighted (`[[term]]`).
- **Interactive CLI UI:** A highly polished, menu-driven command-line interface that glues the crawler, preprocessor, indexer, and search engines together.

---

## Performance Report

The system's performance was evaluated using standard `trec_eval` metrics on the provided datasets. The implementation of the **Okapi BM25** probabilistic model yielded a massive improvement over the baseline TF-IDF model, particularly for the highly inflected Czech language, bringing the **MAP score to ~0.19**. English language evaluation achieved a **MAP of ~0.54**.

### Highest Achieved Scores
- **Czech (BM25 + Stemming):** MAP = **0.1914** | P@10 = 0.2220
- **English (BM25 + Stemming):** MAP = **0.5446** | P@10 = 0.1900

### Full TREC Metrics Overview

| Dataset | Language | Pipeline | Scorer | MAP | P@10 | NDCG | MRR |
|---------|----------|----------|--------|-----|------|------|-----|
| eval_data_cs | cs | `baseline` | **TFIDF** | 0.0925 | 0.1060 | 0.2978 | 0.2052 |
| eval_data_cs | cs | `baseline` | **BM25** | 0.1677 | 0.1980 | 0.3815 | 0.4213 |
| eval_data_cs | cs | `stemming` | **TFIDF** | 0.1088 | 0.1180 | 0.3277 | 0.2276 |
| eval_data_cs | cs | `stemming` | **BM25** | **0.1914** | **0.2220** | 0.4185 | 0.4531 |
| eval_data_cs | cs | `lemmatization` | **TFIDF** | 0.1227 | 0.1440 | 0.3406 | 0.2532 |
| eval_data_cs | cs | `lemmatization` | **BM25** | 0.1903 | 0.2260 | **0.4246** | **0.4801** |
| eval_data_en | en | `baseline` | **TFIDF** | 0.5433 | 0.1860 | 0.6760 | 0.7011 |
| eval_data_en | en | `stemming` | **BM25** | **0.5446** | **0.1900** | **0.6793** | **0.7113** |

*Note: Indexing time for 81,734 Czech documents is ~5 minutes. Ranked searches evaluate in ~1.6 seconds across the entire batch.*

---

## How to Run

### Requirements
- **Python:** Version 3.10 or higher.
- **Dependencies:** Install required libraries using `pip`:
  ```bash
  pip install beautifulsoup4 requests simplemma
  ```

### Data Setup
- Evaluation data (`documents.json`, `full_text_queries.json`, `czech_stopwords.txt`, etc.) should be placed in the root directory under `data/eval_data_cs/` or `data/eval_data_en/`.
- Downloaded pre-processed datasets (if any) can be placed in `data/`.
- **Dataset Link:** Download from Google Drive. After downloading, extract the contents into the `data/` directory.
- https://drive.google.com/file/d/1oYKwR-O_po-GtoacoVF3typPUZABlURP/view?usp=sharing

### Execution
The entire pipeline is glued together via a single entry point. Start the interactive console by running:
```bash
python main.py
```

This launches the main orchestrator menu:
1. **Run Crawler:** Start the web scraper (foreground or background).
2. **Preprocess & Index:** Select a raw data file (like `documents.json` or `crawled_pages.json`). The system will auto-detect text fields, ask for the language, process the text, and generate `inverted_index_*.json` files.
3. **Index preprocessed docs:** Build an index from an already normalized `.jsonl` document stream.
4. **Run Retrieval:** Select an existing index from the menu, choose your scoring model (TF-IDF, BM25, or Boolean), and type your queries interactively.