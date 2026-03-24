import time
from pathlib import Path

try:
    from .cli import parse_args
    from .config import DEFAULT_INPUT, build_pipelines
    from .dataset import load_records, detect_text_keys, normalize_docs
    from .orchestration import parse_pipeline_selection, process_pipeline
    from .tokenizer import RegexMatchTokenizer
    from .dataset import write_weighted_vocab, write_jsonl_records
except ImportError:
    from cli import parse_args
    from config import DEFAULT_INPUT, build_pipelines
    from dataset import load_records, detect_text_keys, normalize_docs
    from orchestration import parse_pipeline_selection, process_pipeline
    from tokenizer import RegexMatchTokenizer
    from dataset import write_weighted_vocab, write_jsonl_records


def main() -> int:
    args = parse_args()
    run_start = time.perf_counter()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(input_path)
    available_keys = detect_text_keys(records)

    if args.list_text_keys:
        if available_keys:
            print("Detected text-like keys:")
            for key in available_keys:
                print(f"- {key}")
        else:
            print("No text-like keys detected.")
        return 0

    selected_key = args.text_key
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
    pipelines = build_pipelines()
    selected_pipeline_names = parse_pipeline_selection(args.pipelines)
    pipeline_results: dict[str, tuple[dict, list[dict]]] = {}

    for name in selected_pipeline_names:
        p_start = time.perf_counter()
        vocab, normalized_docs = process_pipeline(
            name,
            pipelines[name],
            raw_docs,
            tokenizer,
            progress_every=args.progress_every,
        )
        pipeline_results[name] = (vocab, normalized_docs)

        out_path = output_dir / f"vocab_{name}.txt"
        write_start = time.perf_counter()
        with out_path.open("w", encoding="utf-8") as f:
            write_weighted_vocab(vocab, f)

        docs_out_path = output_dir / f"docs_{name}.jsonl"
        write_jsonl_records(normalized_docs, docs_out_path)

        write_elapsed = time.perf_counter() - write_start
        total_elapsed = time.perf_counter() - p_start
        print(f"[{name}] wrote {out_path} in {write_elapsed:.1f}s")
        print(f"[{name}] wrote {docs_out_path}")
        print(f"[{name}] done | terms={len(vocab)} | total={total_elapsed:.1f}s")

    if ("baseline" in selected_pipeline_names) and not args.no_compat_vocab:
        compat_path = output_dir / "vocab.txt"
        baseline_vocab, _ = pipeline_results["baseline"]
        with compat_path.open("w", encoding="utf-8") as f:
            write_weighted_vocab(baseline_vocab, f)
        print(f"[baseline_compat] wrote {compat_path}")

    print(f"Run finished in {time.perf_counter() - run_start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
