from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from preprocessing.config import PIPELINE_NAMES, build_pipelines
from preprocessing.language_config import normalize_language_code
from retrieval.boolean import BooleanIndex, BooleanScorer
from retrieval.tfidf import CosineScorer, InvertedIndex


DOCUMENT_ID_KEYS = ("doc_id", "document_id", "docno", "id", "url")
DOCUMENT_TEXT_KEYS = (
    "text",
    "article_text",
    "content",
    "body",
    "normalized_text",
    "document",
)
QUERY_ID_KEYS = ("qid", "query_id", "queryid", "id", "question_id")
QUERY_TEXT_KEYS = ("query", "text", "content", "body", "normalized_query", "title")
COLLECTION_KEYS = ("documents", "queries", "items", "data", "records", "results")
DEFAULT_TOP_K = 1000


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    def load_jsonl() -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as file_handle:
            for line in file_handle:
                stripped = line.strip()
                if not stripped:
                    continue
                loaded = json.loads(stripped)
                if isinstance(loaded, dict):
                    rows.append(loaded)
        return rows

    if path.suffix.lower() == ".jsonl":
        return load_jsonl()

    try:
        with path.open("r", encoding="utf-8") as file_handle:
            loaded = json.load(file_handle)
    except json.JSONDecodeError:
        return load_jsonl()

    if isinstance(loaded, list):
        return [row for row in loaded if isinstance(row, dict)]

    if isinstance(loaded, dict):
        for key in COLLECTION_KEYS:
            value = loaded.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]

        if all(not isinstance(value, (dict, list)) for value in loaded.values()):
            return [{"id": str(key), "text": value} for key, value in loaded.items()]

        if all(isinstance(value, dict) for value in loaded.values()):
            rows: list[dict[str, Any]] = []
            for key, value in loaded.items():
                row = dict(value)
                row.setdefault("id", str(key))
                rows.append(row)
            return rows

        return [loaded]

    raise ValueError(f"Unsupported JSON format in {path}")


def _detect_text_key(records: list[dict[str, Any]], preferred_keys: tuple[str, ...]) -> str | None:
    key_score: dict[str, int] = {}
    for row in records[:200]:
        for key, value in row.items():
            if isinstance(value, str) and value.strip():
                key_score[key] = key_score.get(key, 0) + 1

    for key in preferred_keys:
        if key_score.get(key, 0) > 0:
            return key

    if key_score:
        return max(key_score.items(), key=lambda item: (item[1], item[0]))[0]

    return None


def _resolve_pipeline_and_language(pipeline: str, language: str | None) -> tuple[str, str]:
    normalized_pipeline = pipeline.strip()
    resolved_language = normalize_language_code(language or "cs")

    for language_code in ("cs", "sk", "en"):
        suffix = f"_{language_code}"
        if normalized_pipeline.endswith(suffix):
            candidate = normalized_pipeline[: -len(suffix)]
            if candidate in PIPELINE_NAMES:
                return candidate, normalize_language_code(language_code)

    if normalized_pipeline not in PIPELINE_NAMES:
        raise ValueError(f"Unknown preprocessing pipeline: {pipeline}")

    return normalized_pipeline, resolved_language


def _normalize_collection(records: list[dict[str, Any]], *, id_keys: tuple[str, ...], text_keys: tuple[str, ...], fallback_prefix: str) -> dict[str, str]:
    text_key = _detect_text_key(records, text_keys)
    if text_key is None:
        raise ValueError(f"Could not detect a text field for {fallback_prefix} records")

    normalized: dict[str, str] = {}
    for index, row in enumerate(records, start=1):
        if not isinstance(row, dict):
            continue

        text_value = row.get(text_key)
        if not isinstance(text_value, str) or not text_value.strip():
            continue

        doc_id = None
        for key in id_keys:
            candidate = row.get(key)
            if isinstance(candidate, str) and candidate.strip():
                doc_id = candidate.strip()
                break
            if candidate is not None and not isinstance(candidate, (dict, list)):
                doc_id = str(candidate)
                break

        if doc_id is None:
            doc_id = f"{fallback_prefix}_{index}"

        normalized[doc_id] = text_value.strip()

    if not normalized:
        raise ValueError(f"No usable {fallback_prefix} records were found")

    return normalized


def _load_qrels(path: Path) -> dict[str, set[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Qrels file does not exist: {path}")

    qrels: dict[str, set[str]] = {}

    def add_qrels_row(qid: str, doc_id: str, relevance: int) -> None:
        if relevance > 0:
            qrels.setdefault(qid, set()).add(doc_id)

    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as file_handle:
            for line in file_handle:
                stripped = line.strip()
                if not stripped:
                    continue
                row = json.loads(stripped)
                if not isinstance(row, dict):
                    continue
                qid = row.get("qid") or row.get("query_id") or row.get("id")
                doc_id = row.get("doc_id") or row.get("document_id") or row.get("docno")
                rel = row.get("relevance") or row.get("rel") or row.get("score") or 0
                if qid is None or doc_id is None:
                    continue
                add_qrels_row(str(qid), str(doc_id), int(rel))
        return qrels

    with path.open("r", encoding="utf-8") as file_handle:
        for line in file_handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            parts = stripped.split()
            if len(parts) < 4:
                continue

            qid, _, doc_id, rel = parts[:4]
            try:
                add_qrels_row(qid, doc_id, int(rel))
            except ValueError:
                continue

    return qrels


def _average_precision(ranked_doc_ids: list[str], relevant_docs: set[str]) -> float:
    if not relevant_docs:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for rank, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in relevant_docs:
            hits += 1
            precision_sum += hits / rank

    return precision_sum / len(relevant_docs)


def _build_scorer(model: str, documents: dict[str, str], preprocessor):
    if model == "boolean":
        index = BooleanIndex()
        index.build(documents, preprocessor)
        return BooleanScorer(index, preprocessor)

    index = InvertedIndex()
    index.build(documents, preprocessor)
    return CosineScorer(index, preprocessor)


def run_trec_evaluation(
    *,
    documents_path: str | Path,
    queries_path: str | Path,
    output_path: str | Path,
    pipeline: str = "baseline",
    language: str | None = None,
    model: str = "tfidf",
    top_k: int = DEFAULT_TOP_K,
    run_id: str = "run",
    qrels_path: str | Path | None = None,
) -> int:
    documents_file = Path(documents_path)
    queries_file = Path(queries_path)
    output_file = Path(output_path)
    qrels_file = Path(qrels_path) if qrels_path else None

    resolved_pipeline, resolved_language = _resolve_pipeline_and_language(pipeline, language)

    documents_records = _load_json_records(documents_file)
    queries_records = _load_json_records(queries_file)

    documents = _normalize_collection(
        documents_records,
        id_keys=DOCUMENT_ID_KEYS,
        text_keys=DOCUMENT_TEXT_KEYS,
        fallback_prefix="doc",
    )
    queries = _normalize_collection(
        queries_records,
        id_keys=QUERY_ID_KEYS,
        text_keys=QUERY_TEXT_KEYS,
        fallback_prefix="q",
    )

    pipelines = build_pipelines(resolved_language)
    if resolved_pipeline not in pipelines:
        raise ValueError(f"Pipeline '{resolved_pipeline}' is not available for language '{resolved_language}'")

    preprocessor = pipelines[resolved_pipeline]
    scorer = _build_scorer(model, documents, preprocessor)
    qrels = _load_qrels(qrels_file) if qrels_file else None

    top_k = max(1, min(int(top_k), DEFAULT_TOP_K))

    output_file.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    query_rankings: dict[str, list[str]] = {}

    with output_file.open("w", encoding="utf-8") as file_handle:
        for qid, query_text in queries.items():
            ranked = scorer.search(query_text)
            limited = ranked[:top_k]
            query_rankings[qid] = [doc_id for doc_id, _ in limited]

            for rank, (doc_id, score) in enumerate(limited, start=1):
                file_handle.write(f"{qid} 0 {doc_id} {rank} {score:.6f} {run_id}\n")

    elapsed = time.perf_counter() - t0

    print(f"[trec] documents: {len(documents)}")
    print(f"[trec] queries: {len(queries)}")
    print(f"[trec] model: {model}")
    print(f"[trec] pipeline: {resolved_pipeline}")
    print(f"[trec] language: {resolved_language}")
    print(f"[trec] output: {output_file}")
    print(f"[trec] elapsed: {elapsed:.2f}s")

    if qrels:
        ap_scores: list[float] = []
        for qid, relevant_docs in qrels.items():
            ranked_doc_ids = query_rankings.get(qid, [])
            if not relevant_docs:
                continue
            ap_scores.append(_average_precision(ranked_doc_ids, relevant_docs))

        if ap_scores:
            map_score = sum(ap_scores) / len(ap_scores)
            print(f"[trec] MAP: {map_score:.6f}")
        else:
            print("[trec] MAP: N/A (no matching qrels queries)")

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch IR evaluation and export TREC results.")
    parser.add_argument("--documents", required=True, help="Documents JSON/JSONL file.")
    parser.add_argument("--queries", required=True, help="Queries JSON/JSONL file.")
    parser.add_argument("--output", default="data/evaluation/results.txt", help="Output TREC run file.")
    parser.add_argument("--pipeline", default="baseline", help="Preprocessing pipeline name.")
    parser.add_argument(
        "--language",
        default=None,
        choices=("cs", "sk", "en"),
        help="Language used by preprocessing. Defaults to Czech unless the pipeline suffix encodes it.",
    )
    parser.add_argument("--model", default="tfidf", choices=("tfidf", "boolean"), help="Retrieval model.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Maximum results per query.")
    parser.add_argument("--run-id", default="run", help="Run identifier written to the TREC file.")
    parser.add_argument("--qrels", default=None, help="Optional relevance file for MAP reporting.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_trec_evaluation(
        documents_path=args.documents,
        queries_path=args.queries,
        output_path=args.output,
        pipeline=args.pipeline,
        language=args.language,
        model=args.model,
        top_k=args.top_k,
        run_id=args.run_id,
        qrels_path=args.qrels,
    )


if __name__ == "__main__":
    raise SystemExit(main())
