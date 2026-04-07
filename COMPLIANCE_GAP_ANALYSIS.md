# Compliance Gap Analysis and Improvement Plan

## Scope
This document compares the current implementation against the final compliance checklist (retrieval correctness, UI behavior, and performance/readiness).

## Executive Summary
Current system is functional, but there are still important compliance gaps that can cost significant points:
1. Boolean query parsing should be documented more clearly for maintainability.
2. Debug output still needs a final polish pass for clean submission runs.
3. The performance/readiness story still needs a concise benchmark note for indexing and search.

High-risk gaps are now centered on output polish and documentation clarity.

---

## Stage 2: Vector Space Model

### Current State
- Log TF implemented as `1 + log10(tf)`.
- Cosine similarity implemented with normalized vectors.
- Document norms are precomputed during indexing and reused in search.

### Flaws / Missing Items
1. No explicit regression test that checks score reproducibility against expected values.
2. No quality gate for target MAP (for example MAP >= 0.2) tied to evaluation run.

### Improvement Actions
1. Add deterministic scoring tests for a tiny synthetic corpus.
2. Add evaluation summary step that computes MAP and fails if threshold not met.
3. Freeze index/query preprocessing params in evaluation mode to avoid accidental drift.

Priority: Medium

---

## Stage 4: UI/UX and Output Contract

### Current State
- Interactive index selection exists.
- Interactive model selection exists.

### Flaws / Missing Items
1. The query UI still has a compact terminal-oriented presentation and could be made more explicit for evaluation mode.
2. Boolean debug output is still developer-oriented and should be behind a debug flag for polished submission runs.

### Improvement Actions
1. Keep the current per-query model toggle.
2. Gate debug output behind a flag so the console stays readable.
3. Preserve the current table output for interactive exploration.

Priority: High

---

## Stage 5: Documentation and Technical Explainability

### Current State
- User-facing README exists.
- Technical workflow document exists.

### Flaws / Missing Items
1. No programmer-facing parser design notes for the Boolean query parser.

### Improvement Actions
1. Add a short parser design section describing operator precedence and postfix evaluation.
2. Keep the evaluation runbook in sync with the exporter if the output contract changes.

Priority: Medium

---

## Stage 6: Performance and Readiness Checks

### Current State
- TF-IDF implementation precomputes document norms during indexing.
- Candidate filtering via postings is in place for search.

### Flaws / Missing Items
1. No explicit benchmark threshold has been committed for indexing/search timing.
2. No automated check for full corpus search completion time.

### Improvement Actions
1. Add benchmark utility:
   - indexing duration
   - average query latency
   - p95 query latency
2. Add CI/local guard for indexing/search runtime budget.

Priority: Medium

---

## Recommended Implementation Order

1. Boolean parser correctness (parentheses + precedence + postfix evaluator).
2. Exact output contract (`Total documents found: X`, per-query model toggle).
3. Documentation update with final command references.

---

## Definition of Done (Submission-Oriented)

The system is submission-ready when all are true:
1. Boolean queries with parentheses evaluate correctly.
2. Every query run prints `Total documents found: X`.
3. Retrieval results are reproducible for the selected index and pipeline.
4. README + technical docs include exact usage commands for crawler, preprocessing, indexing, and retrieval.
