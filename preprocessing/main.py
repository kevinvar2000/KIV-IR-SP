import time
from pathlib import Path

try:
    from .cli import parse_args
    from .config import DEFAULT_INPUT, build_pipelines, normalize_language_code
    from .dataset import load_records, detect_text_keys, normalize_docs
    from .orchestration import parse_pipeline_selection, process_pipeline
    from .tokenizer import RegexMatchTokenizer
    from .dataset import write_weighted_vocab, write_jsonl_records
except ImportError:
    from cli import parse_args
    from config import DEFAULT_INPUT, build_pipelines, normalize_language_code
    from dataset import load_records, detect_text_keys, normalize_docs
    from orchestration import parse_pipeline_selection, process_pipeline
    from tokenizer import RegexMatchTokenizer
    from dataset import write_weighted_vocab, write_jsonl_records


def run_preprocessing_stage(
    *,
    input_path: str | Path = DEFAULT_INPUT,
    output_dir: str | Path,
    text_key: str | None = None,
    pipeline_selection: list[str] | None = None,
    language: str = "cs",
    progress_every: int = 1000,
    no_compat_vocab: bool = False,
    list_text_keys: bool = False,
    skip_index: bool = False,
) -> int:
    run_start = time.perf_counter()

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
    pipeline_results: dict[str, tuple[dict, list[dict]]] = {}

    for name in selected_pipeline_names:
        p_start = time.perf_counter()
        vocab, normalized_docs = process_pipeline(
            name,
            pipeline_map[name],
            raw_docs,
            tokenizer,
            progress_every=progress_every,
        )
        pipeline_results[name] = (vocab, normalized_docs)

        out_path = output_dir / f"vocab_{name}{artifact_suffix}.txt"
        write_start = time.perf_counter()
        with out_path.open("w", encoding="utf-8") as f:
            write_weighted_vocab(vocab, f)

        docs_out_path = output_dir / f"docs_{name}{artifact_suffix}.jsonl"
        write_jsonl_records(normalized_docs, docs_out_path)

        write_elapsed = time.perf_counter() - write_start
        total_elapsed = time.perf_counter() - p_start
        print(f"[{name}] wrote {out_path} in {write_elapsed:.1f}s")
        print(f"[{name}] wrote {docs_out_path}")
        print(f"[{name}] done | terms={len(vocab)} | total={total_elapsed:.1f}s")

    if ("baseline" in selected_pipeline_names) and not no_compat_vocab and normalized_language == "cs":
        compat_path = output_dir / "vocab.txt"
        baseline_vocab, _ = pipeline_results["baseline"]
        with compat_path.open("w", encoding="utf-8") as f:
            write_weighted_vocab(baseline_vocab, f)
        print(f"[baseline_compat] wrote {compat_path}")

    if not skip_index:
        try:
            from indexing.main import run_indexing_stage
        except ImportError:
            from ..indexing.main import run_indexing_stage

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

    print(f"Run finished in {time.perf_counter() - run_start:.1f}s")
    return 0


def main() -> int:
    args = parse_args()
    return run_preprocessing_stage(
        input_path=args.input,
        output_dir=args.output_dir,
        text_key=args.text_key,
        pipeline_selection=args.pipelines,
        language=args.language,
        progress_every=args.progress_every,
        no_compat_vocab=args.no_compat_vocab,
        list_text_keys=args.list_text_keys,
        skip_index=args.skip_index,
    )


if __name__ == "__main__":
    raise SystemExit(main())
