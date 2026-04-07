"""Preprocessing stage runner that writes docs, vocabularies, and index artifacts."""

import time
import json
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone

from .config import DEFAULT_INPUT, DEFAULT_OUTPUT_DIR, build_pipelines, normalize_language_code
from .dataset import load_records, detect_text_keys, normalize_docs
from .orchestration import parse_pipeline_selection, process_pipeline_to_files
from .tokenizer import RegexMatchTokenizer
from .dataset import write_weighted_vocab


def run_preprocessing_stage(
    *,
    input_path: str | Path = DEFAULT_INPUT,
    output_dir: str | Path,
    text_key: str | None = None,
    pipeline_selection: list[str] | None = None,
    language: str = "cs",
    progress_every: int = 1000,
    write_vocab: bool = True,
    no_compat_vocab: bool = False,
    list_text_keys: bool = False,
    skip_index: bool = False,
) -> int:
    """Run preprocessing pipelines and write docs, vocab, index, and artifact metadata."""
    run_start = time.perf_counter()
    run_started_at = datetime.now(timezone.utc)

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_language = normalize_language_code(language)
    artifact_suffix = "" if normalized_language == "cs" else f"_{normalized_language}"

    records = load_records(input_path)
    available_keys = detect_text_keys(records)

    if list_text_keys:
        if available_keys:
            print("Detected text-like keys:")
            for key in available_keys:
                print(f"- {key}")
        else:
            print("No text-like keys detected.")
        return 0

    selected_key = text_key
    if not selected_key:
        if "article_text" in available_keys:
            selected_key = "article_text"
        elif "text" in available_keys:
            selected_key = "text"
        elif available_keys:
            selected_key = available_keys[0]

    if not selected_key:
        raise ValueError("Unable to determine text key automatically. Use --text-key.")

    if available_keys and selected_key not in available_keys:
        raise ValueError(
            f"Selected text key '{selected_key}' not found among detected keys: {', '.join(available_keys)}"
        )

    raw_docs = normalize_docs(records, selected_key)
    print(f"Loaded {len(records)} rows from {input_path}")
    print(f"Using key '{selected_key}' -> {len(raw_docs)} documents")

    if not raw_docs:
        raise ValueError("No non-empty documents found for selected text key.")

    tokenizer = RegexMatchTokenizer()
    pipeline_map = build_pipelines(normalized_language)
    selected_pipeline_names = parse_pipeline_selection(
        pipeline_selection if pipeline_selection is not None else ["all"]
    )
    pipeline_vocabs: dict[str, Counter] = {}
    pipeline_artifacts: dict[str, dict[str, str | None]] = {}

    for name in selected_pipeline_names:
        p_start = time.perf_counter()
        docs_out_path = output_dir / f"docs_{name}{artifact_suffix}.jsonl"
        vocab = process_pipeline_to_files(
            name,
            pipeline_map[name],
            raw_docs,
            tokenizer,
            docs_output_path=docs_out_path,
            progress_every=progress_every,
        )
        pipeline_vocabs[name] = vocab
        pipeline_artifacts[name] = {
            "docs": str(docs_out_path),
            "vocab": None,
            "index": None,
        }

        out_path = output_dir / f"vocab_{name}{artifact_suffix}.txt"
        write_elapsed = 0.0
        if write_vocab:
            write_start = time.perf_counter()
            with out_path.open("w", encoding="utf-8") as f:
                write_weighted_vocab(vocab, f)
            write_elapsed = time.perf_counter() - write_start
            pipeline_artifacts[name]["vocab"] = str(out_path)

        total_elapsed = time.perf_counter() - p_start
        if write_vocab:
            print(f"[{name}] wrote {out_path} in {write_elapsed:.1f}s")
        else:
            print(f"[{name}] skipped vocab output")
        print(f"[{name}] wrote {docs_out_path}")
        print(f"[{name}] done | terms={len(vocab)} | total={total_elapsed:.1f}s")

    if (
        write_vocab
        and ("baseline" in selected_pipeline_names)
        and not no_compat_vocab
        and normalized_language == "cs"
    ):
        compat_path = output_dir / "vocab.txt"
        baseline_vocab = pipeline_vocabs["baseline"]
        with compat_path.open("w", encoding="utf-8") as f:
            write_weighted_vocab(baseline_vocab, f)
        print(f"[baseline_compat] wrote {compat_path}")

    if not skip_index:
        from indexing.main import run_indexing_stage

        for pipeline_name in selected_pipeline_names:
            pipeline_with_language = f"{pipeline_name}{artifact_suffix}"
            docs_path = output_dir / f"docs_{pipeline_with_language}.jsonl"
            index_output = output_dir.parent / "index" / f"inverted_index_{pipeline_with_language}.json"
            rc = run_indexing_stage(
                pipeline=pipeline_with_language,
                input_path=docs_path,
                output_path=index_output,
            )
            if rc != 0:
                return rc
            if pipeline_name in pipeline_artifacts:
                pipeline_artifacts[pipeline_name]["index"] = str(index_output)

    run_duration = time.perf_counter() - run_start
    metadata_path = output_dir / f"artifacts_index_{run_started_at.strftime('%Y%m%d_%H%M%S')}.json"
    metadata_payload = {
        "meta": {
            "created_at_utc": run_started_at.isoformat(),
            "duration_seconds": round(run_duration, 3),
            "input_path": str(input_path),
            "output_dir": str(output_dir),
            "language": normalized_language,
            "text_key": selected_key,
            "pipeline_selection": selected_pipeline_names,
            "write_vocab": write_vocab,
            "skip_index": skip_index,
            "document_count": len(raw_docs),
        },
        "artifacts": pipeline_artifacts,
    }
    with metadata_path.open("w", encoding="utf-8") as file_handle:
        json.dump(metadata_payload, file_handle, ensure_ascii=False, indent=2)
    print(f"[artifacts] wrote {metadata_path}")

    print(f"Run finished in {run_duration:.1f}s")
    return 0


def main() -> int:
    """Run preprocessing stage with default crawler input and all pipelines."""
    return run_preprocessing_stage(
        input_path=DEFAULT_INPUT,
        output_dir=DEFAULT_OUTPUT_DIR,
        pipeline_selection=["all"],
        language="cs",
    )


if __name__ == "__main__":
    raise SystemExit(main())
