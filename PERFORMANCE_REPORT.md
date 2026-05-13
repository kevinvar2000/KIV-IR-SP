# Evaluation & Performance Report

**Generated on:** 2026-05-11 18:21:44

## Overview

This report aggregates the performance and accuracy metrics across different preprocessing pipelines and languages, evaluated against `trec_eval`.

## Retrieval Effectiveness (TREC Metrics)

| Dataset | Language | Pipeline | Scorer | MAP | P@10 | NDCG | MRR |
|---------|----------|----------|--------|-----|------|------|-----|
| eval_data_cs | cs | `baseline` | **TFIDF** | 0.0925 | 0.1060 | 0.2978 | 0.2052 |
| eval_data_cs | cs | `baseline` | **BM25** | 0.1677 | 0.1980 | 0.3815 | 0.4213 |
| eval_data_cs | cs | `stemming` | **TFIDF** | 0.1088 | 0.1180 | 0.3277 | 0.2276 |
| eval_data_cs | cs | `stemming` | **BM25** | 0.1914 | 0.2220 | 0.4185 | 0.4531 |
| eval_data_cs | cs | `lemmatization` | **TFIDF** | 0.1227 | 0.1440 | 0.3406 | 0.2532 |
| eval_data_cs | cs | `lemmatization` | **BM25** | 0.1903 | 0.2260 | 0.4246 | 0.4801 |
| eval_data_en | en | `baseline` | **TFIDF** | 0.5433 | 0.1860 | 0.6760 | 0.7011 |
| eval_data_en | en | `baseline` | **BM25** | 0.5401 | 0.1880 | 0.6747 | 0.6973 |
| eval_data_en | en | `stemming` | **TFIDF** | 0.5370 | 0.1860 | 0.6725 | 0.7032 |
| eval_data_en | en | `stemming` | **BM25** | 0.5446 | 0.1900 | 0.6793 | 0.7113 |
| eval_data_en | en | `lemmatization` | **TFIDF** | 0.5329 | 0.1850 | 0.6695 | 0.6987 |
| eval_data_en | en | `lemmatization` | **BM25** | 0.5420 | 0.1890 | 0.6762 | 0.7029 |

## Speed & Indexing Performance

| Dataset | Pipeline | Scorer | Docs | Queries | Indexing Time (s) | Ranked Search Time (s) | Boolean Time (s) |
|---------|----------|--------|------|---------|-------------------|------------------------|------------------|
| eval_data_cs | `baseline` | **TFIDF** | 81734 | 50 | 280.263 | 25.161 | 183.703 |
| eval_data_cs | `baseline` | **BM25** | 81734 | 50 | 302.118 | 1.307 | 167.087 |
| eval_data_cs | `stemming` | **TFIDF** | 81734 | 50 | 329.455 | 30.397 | 176.272 |
| eval_data_cs | `stemming` | **BM25** | 81734 | 50 | 320.023 | 1.632 | 190.752 |
| eval_data_cs | `lemmatization` | **TFIDF** | 81734 | 50 | 361.067 | 42.836 | 168.609 |
| eval_data_cs | `lemmatization` | **BM25** | 81734 | 50 | 362.528 | 2.089 | 147.998 |
| eval_data_en | `baseline` | **TFIDF** | 609 | 100 | 10.794 | 6.146 | 1.315 |
| eval_data_en | `baseline` | **BM25** | 609 | 100 | 10.949 | 0.300 | 1.302 |
| eval_data_en | `stemming` | **TFIDF** | 609 | 100 | 12.580 | 6.209 | 1.549 |
| eval_data_en | `stemming` | **BM25** | 609 | 100 | 12.575 | 0.368 | 1.471 |
| eval_data_en | `lemmatization` | **TFIDF** | 609 | 100 | 12.411 | 7.891 | 1.571 |
| eval_data_en | `lemmatization` | **BM25** | 609 | 100 | 12.368 | 0.364 | 1.466 |

---
*Note: Timings reflect local execution and may vary based on hardware.*