"""Evaluation interface for assignment runs using local IR implementation and trec_eval."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app_config import ROOT

_interface_path = Path(__file__).with_name("interface.py")
_interface_spec = importlib.util.spec_from_file_location("eval_interface_interface", _interface_path)
if _interface_spec is None or _interface_spec.loader is None:
    raise ImportError(f"Unable to load interface module from {_interface_path}")
_interface_module = importlib.util.module_from_spec(_interface_spec)
_interface_spec.loader.exec_module(_interface_module)
Index = _interface_module.Index
SearchEngine = _interface_module.SearchEngine
from preprocessing.config import build_pipelines
from preprocessing.language_config import normalize_language_code
from preprocessing.tokenizer import RegexMatchTokenizer
from retrieval.boolean import BooleanIndex, BooleanScorer
from retrieval.tfidf import CosineScorer, InvertedIndex


TEXT_KEYS = (
    "text",
    "article_text",
    "normalized_text",
    "content",
    "body",
    "fact",
    "description",
    "title",
)
DOC_ID_KEYS = ("id", "doc_id", "document_id", "docno", "url")
QUERY_ID_KEYS = ("id", "qid", "query_id", "queryid")
QUERY_TEXT_KEYS = ("description", "query", "text", "title", "question")


class PipelineTokenizer:
    """Tokenize and normalize text using selected preprocessing pipeline and language."""

    def __init__(self, pipeline: str, language: str) -> None:
        self.tokenizer = RegexMatchTokenizer()
        self.pipeline = build_pipelines(normalize_language_code(language))[pipeline]

    def tokenize(self, text: str) -> list[str]:
        tokens = self.tokenizer.tokenize(text)
        tokens = self.pipeline.preprocess(tokens, text)
        return [token.processed_form for token in tokens if token.processed_form]


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def _first_text_value(row: dict) -> str:
    for key in TEXT_KEYS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _first_id_value(row: dict, id_keys: Iterable[str], fallback_prefix: str, idx: int) -> str:
    for key in id_keys:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            if value.strip():
                return value.strip()
            continue
        return str(value)
    return f"{fallback_prefix}{idx}"


def _normalize_documents(raw_documents: list[dict]) -> dict[str, str]:
    documents: dict[str, str] = {}
    for idx, row in enumerate(raw_documents, start=1):
        if not isinstance(row, dict):
            continue
        text = _first_text_value(row)
        if not text:
            continue
        doc_id = _first_id_value(row, DOC_ID_KEYS, "d", idx)
        documents[doc_id] = text
    if not documents:
        raise ValueError("No usable documents found in evaluation documents file.")
    return documents


def _normalize_queries(raw_queries: list[dict]) -> list[dict[str, str]]:
    queries: list[dict[str, str]] = []
    for idx, row in enumerate(raw_queries, start=1):
        if not isinstance(row, dict):
            continue
        qid = _first_id_value(row, QUERY_ID_KEYS, "q", idx)
        qtext = ""
        for key in QUERY_TEXT_KEYS:
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                qtext = value.strip()
                break
        if qtext:
            queries.append({"id": qid, "text": qtext})
    if not queries:
        raise ValueError("No usable queries found in evaluation query file.")
    return queries


class LocalIndex(Index):
    """Adapter implementing assignment interface over project TF-IDF and Boolean indexes."""

    def __init__(self, tokenizer: PipelineTokenizer):
        self.tokenizer = tokenizer
        self.documents: dict[str, str] = {}
        self.tfidf_index = InvertedIndex()
        self.boolean_index = BooleanIndex()

    def index_documents(self, documents: Iterable[dict[str, str]]):
        self.documents = _normalize_documents(list(documents))
        self.tfidf_index.build(self.documents, self.tokenizer)
        self.boolean_index.build(self.documents, self.tokenizer)

    def get_document(self, doc_id: str) -> dict[str, str]:
        return {"id": doc_id, "text": self.documents.get(doc_id, "")}


class LocalSearchEngine(SearchEngine):
    """Adapter implementing assignment search API over current project retrieval code."""

    def __init__(self, index: LocalIndex):
        super().__init__(index)
        self.ranked = CosineScorer(index.tfidf_index, index.tokenizer)
        self.boolean = BooleanScorer(index.boolean_index, index.tokenizer)

    def search(self, query: str) -> list[tuple[str, float]]:
        return self.ranked.search(query)

    def boolean_search(self, query: str) -> set[str]:
        return {doc_id for doc_id, _ in self.boolean.search(query)}


def run_trec_eval(trec_eval_bin: str | Path, gold_file: Path, results_file: Path) -> tuple[int, str, str]:
    """Run trec_eval if binary is available and return (returncode, stdout, stderr)."""
    bin_path = Path(trec_eval_bin)
    if not bin_path.exists():
        return 127, "", f"trec_eval binary not found: {bin_path}"

    proc = subprocess.run(
        [str(bin_path), "-m", "all_trec", str(gold_file), str(results_file)],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def evaluate_ranked(
    search_engine: LocalSearchEngine,
    queries: list[dict[str, str]],
    output_path: Path,
    run_id: str,
    top_k: int,
) -> None:
    """Write ranked retrieval results in 6-column TREC run format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_handle:
        for query in queries:
            ranked = search_engine.search(query["text"])
            for rank, (doc_id, score) in enumerate(ranked[:top_k], start=1):
                file_handle.write(f"{query['id']} Q0 {doc_id} {rank} {score:.6f} {run_id}\n")


def evaluate_boolean(search_engine: LocalSearchEngine, queries_file: Path) -> dict[str, int]:
    """Run optional boolean query set and report hit counts by line number."""
    hit_counts: dict[str, int] = {}
    with queries_file.open("r", encoding="utf-8") as file_handle:
        for idx, line in enumerate(file_handle, start=1):
            query = line.strip()
            if not query:
                continue
            results = search_engine.boolean_search(query)
            hit_counts[f"bq{idx}"] = len(results)
    return hit_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run assignment evaluation and optional trec_eval.")
    parser.add_argument("--dataset", choices=("eval_data_cs", "eval_data_en"), default="eval_data_cs")
    parser.add_argument("--pipeline", default="baseline", help="Preprocessing pipeline for index/query normalization.")
    parser.add_argument("--language", default="cs", choices=("cs", "sk", "en"))
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--run-id", default="runindex1")
    parser.add_argument("--trec-eval-bin", default="trec_eval-main/trec_eval")
    parser.add_argument("--skip-trec-eval", action="store_true")
    parser.add_argument("--run-boolean", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    dataset_dir = ROOT / "data" / args.dataset
    documents_file = dataset_dir / "documents.json"
    queries_file = dataset_dir / "full_text_queries.json"
    qrels_file = dataset_dir / "gold_relevancies.txt"
    boolean_file = dataset_dir / "boolean_queries_standard_100.txt"

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = ROOT / "data" / "evaluation"
    trec_results_file = output_dir / f"ranked_results_{args.dataset}_{timestamp}.txt"
    summary_file = output_dir / f"evaluation_summary_{args.dataset}_{timestamp}.json"

    raw_documents = _load_json(documents_file)
    raw_queries = _load_json(queries_file)
    if not isinstance(raw_documents, list):
        raise ValueError(f"Expected a list in documents file: {documents_file}")
    if not isinstance(raw_queries, list):
        raise ValueError(f"Expected a list in queries file: {queries_file}")

    queries = _normalize_queries(raw_queries)

    tokenizer = PipelineTokenizer(args.pipeline, args.language)
    index = LocalIndex(tokenizer)

    t0 = time.perf_counter()
    index.index_documents(raw_documents)
    indexing_seconds = time.perf_counter() - t0

    search_engine = LocalSearchEngine(index)
    t1 = time.perf_counter()
    evaluate_ranked(search_engine, queries, trec_results_file, args.run_id, max(1, args.top_k))
    ranked_seconds = time.perf_counter() - t1

    boolean_summary: dict[str, int] | None = None
    boolean_seconds: float | None = None
    if args.run_boolean and boolean_file.exists():
        t2 = time.perf_counter()
        boolean_summary = evaluate_boolean(search_engine, boolean_file)
        boolean_seconds = time.perf_counter() - t2

    trec_eval_returncode = None
    trec_eval_stdout = ""
    trec_eval_stderr = ""
    if not args.skip_trec_eval:
        rc, out, err = run_trec_eval(args.trec_eval_bin, qrels_file, trec_results_file)
        trec_eval_returncode = rc
        trec_eval_stdout = out
        trec_eval_stderr = err

    summary = {
        "meta": {
            "dataset": args.dataset,
            "pipeline": args.pipeline,
            "language": args.language,
            "query_count": len(queries),
            "document_count": len(index.documents),
            "run_id": args.run_id,
            "created_at": datetime.now().isoformat(),
        },
        "timing_seconds": {
            "indexing": round(indexing_seconds, 3),
            "ranked_batch_search": round(ranked_seconds, 3),
            "boolean_batch_search": round(boolean_seconds, 3) if boolean_seconds is not None else None,
        },
        "thresholds": {
            "indexing_under_300s": indexing_seconds < 300,
            "ranked_batch_under_60s": ranked_seconds < 60,
        },
        "artifacts": {
            "trec_results": str(trec_results_file),
            "qrels": str(qrels_file),
            "summary": str(summary_file),
        },
        "trec_eval": {
            "returncode": trec_eval_returncode,
            "stdout": trec_eval_stdout,
            "stderr": trec_eval_stderr,
        },
        "boolean": {
            "enabled": args.run_boolean,
            "hit_counts": boolean_summary,
        },
    }

    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with summary_file.open("w", encoding="utf-8") as file_handle:
        json.dump(summary, file_handle, ensure_ascii=False, indent=2)

    print(f"[eval] indexed docs={len(index.documents)} in {indexing_seconds:.3f}s")
    print(f"[eval] ranked batch ({len(queries)} queries) in {ranked_seconds:.3f}s")
    print(f"[eval] wrote TREC run: {trec_results_file}")
    print(f"[eval] wrote summary: {summary_file}")
    if trec_eval_returncode is not None:
        print(f"[eval] trec_eval return code: {trec_eval_returncode}")
        if trec_eval_stdout.strip():
            print(trec_eval_stdout)
        if trec_eval_stderr.strip():
            print(trec_eval_stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
