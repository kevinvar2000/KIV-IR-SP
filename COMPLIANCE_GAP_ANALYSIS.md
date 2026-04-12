# Compliance Gap Analysis (Standardni zadani)

## Scope
This checklist is re-evaluated against the current repository state and your mandatory baseline target (20 points).

## Executive Verdict
- Current status: Baseline-compliant in the repository implementation.
- Main remaining dependency: External `trec_eval` binary/environment for producing the official MAP report at evaluation time.
- Strengths: Custom TF-IDF/Boolean implementations, persisted index files, language-aware preprocessing, model toggle in interactive retrieval, evaluation runner with TREC export.

---

## 1. Preprocessing & Indexing (Mandatory)
- [x] Custom Indexer
Evidence: `retrieval/tfidf.py` implements `InvertedIndex` and indexing logic in project code; no Lucene/Elasticsearch usage.

- [x] Linguistic Pipeline
Evidence: `preprocessing/config.py` + `preprocessing/preprocess.py` include stopword removal and stemming/lemmatization.

- [x] Diacritics Policy
Evidence: In `preprocessing/config.py`, `baseline` keeps diacritics, `*_no_diacritics` applies `RemoveDiacriticsPreprocessor()` after stemming/lemmatization.

- [x] In-Memory Store
Evidence: `InvertedIndex` is built/loaded into Python memory structures (`dict`) and queried in RAM (`retrieval/tfidf.py`, `retrieval/boolean.py`).

- [x] Format Compatibility
Evidence: `preprocessing/dataset.py::load_records()` supports JSON and JSONL, including crawler-produced JSONL records.

Section result: PASS

---

## 2. Vector Space Model (Mandatory)
- [x] TF-IDF Calculation
Evidence: `retrieval/tfidf.py::weighted_tf()` uses `1 + log10(tf)` and IDF uses `log10(N/df)`.

- [x] Cosine Similarity
Evidence: `retrieval/tfidf.py::cosine_similarity()` computes normalized dot product.

- [x] Top-X Ranking
Evidence: `retrieval/tfidf.py::search()` sorts by descending score (`scored.sort(key=lambda x: (-x[1], x[0]))`).

Section result: PASS

---

## 3. Boolean Model & Parentheses (Mandatory)
- [x] Operators AND/OR/NOT
Evidence: `retrieval/boolean.py` supports all three operators.

- [x] Parentheses Priority
Evidence: `retrieval/boolean.py` converts infix to postfix and validates parentheses.

- [x] Shunting-yard Algorithm (Recommended)
Evidence: `retrieval/boolean.py::_infix_to_postfix()` uses operator stack/precedence/associativity.

- [x] Model Toggle
Evidence: `retrieval/query_interface.py::_ask_search_method()` toggles TF-IDF vs Boolean before querying.

Section result: PASS

---

## 4. Evaluation Engine (Mandatory)
- [x] TREC Format Exporter (`qid iter docno rank sim run_id`)
Evidence: `eval_interface/evaluate.py::evaluate_ranked()` writes 6-column TREC run files.

- [x] Batch Processing of queries JSON to TREC output
Evidence: `eval_interface/evaluate.py::main()` loads `documents.json` and `full_text_queries.json`, runs all queries, and writes run files in one batch.

- [x] Performance Threshold Validation
Evidence: generated summary for `eval_data_en` shows indexing 10.656s and ranked batch search 8.567s, both below thresholds.

Section result: PASS

---

## 5. Extra Credit (Nadstandardni funkcnost)
- [x] File-based Index
Evidence: Index is persisted and reloaded (`indexing/main.py`, `retrieval/tfidf.py::save_json/load_json`).

- [~] Highlighting (KWIC)
Status: Partial
Evidence: `retrieval/query_interface.py` prints highlighted debug snippets (`[[term]]`), but behavior is debug-like and not a clean dedicated KWIC result mode.

- [ ] Language Detection (Automatic)
Gap: Language is selected/resolved by pipeline metadata, not auto-detected from text/query.

- [x] Custom Parser
Evidence: Boolean parser is implemented in-house in `retrieval/boolean.py` (no parser libraries like `ply`/`pyparsing`).

Section result: Partial

---

## 6. Submission Artifacts
- [x] README.txt / run instructions
Evidence: `README.md` includes install/run workflow and project structure.

- [ ] PDF Documentation with evaluation results (MAP)
Gap: No PDF artifact in repository and no MAP report generation currently.

- [ ] Data Link on gapps.zcu.cz
Gap: Cannot be verified from repository.

- [ ] ZIP name `JmenoPRIJMENI.zip`
Gap: Packaging step not represented in repository.

Section result: Incomplete

---

## Critical Boolean Test (XOR-style)
Suggested manual verification:
1. Query `A AND B`
2. Query `A OR B`
3. Query `(A OR B) AND NOT (A AND B)`

Expected: step 3 returns docs containing exactly one term.

Implementation confidence: High (based on postfix evaluation and operator precedence in `retrieval/boolean.py`).

---

## Baseline 20-Point Readiness
Mandatory sections passed:
- Section 1: PASS
- Section 2: PASS
- Section 3: PASS

- Section 4: PASS

Conclusion: Repository implementation now meets the mandatory baseline checklist.

---

## Fastest Path to Compliance
1. Install or point to the official `trec_eval` binary and run the evaluator on the target dataset.
2. Save the resulting MAP/P@10 summary into the final PDF submission artifact.
3. Package the repository with the evaluation outputs included.
