import argparse
import traceback

from crawler.crawler import run_crawler
from indexing.main import run_indexing_stage
from preprocessing.main import run_preprocessing_stage
from retrieval.tfidf_search import run_retrieval_stage
from runner import run_crawler_background


def run_stage(name: str, fn) -> int:
    print(f"\n[{name}] running (in-process)")
    try:
        result = fn()
        exit_code = int(result) if result is not None else 0
    except Exception:
        print(f"[{name}] failed with unhandled exception")
        traceback.print_exc()
        return 1

    if exit_code != 0:
        print(f"[{name}] failed with exit code {exit_code}")
    else:
        print(f"[{name}] finished")
    return exit_code


def run_non_interactive(args: argparse.Namespace) -> int:
    if args.stage == "crawler" and args.crawler_background:
        return run_crawler_background()

    stages = {
        "crawler": (
            "crawler",
            lambda: run_crawler(),
        ),
        "preprocessing": (
            "preprocessing",
            lambda: run_preprocessing_stage(
                input_path=args.input_data,
                output_dir=args.preprocess_output_dir,
                text_key=args.text_key,
                pipeline_selection=args.pipelines,
                skip_index=args.skip_index,
            ),
        ),
        "indexing": (
            "indexing",
            lambda: run_indexing_stage(
                pipeline=args.index_pipeline,
                input_path=args.index_input,
                output_path=args.index_output,
            ),
        ),
        "retrieval": (
            "retrieval",
            lambda: run_retrieval_stage(
                files=args.retrieval_files,
                index_file=args.index_file,
                pipeline=args.index_pipeline,
                query=args.query,
                top_k=args.top_k,
            ),
        ),
        "tfidf": (
            "retrieval",
            lambda: run_retrieval_stage(
                files=args.retrieval_files,
                index_file=args.index_file,
                pipeline=args.index_pipeline,
                query=args.query,
                top_k=args.top_k,
            ),
        ),
    }

    if args.stage == "all":
        if args.skip_index:
            order = ["crawler", "preprocessing", "indexing", "retrieval"]
        else:
            order = ["crawler", "preprocessing", "retrieval"]
    else:
        order = ["retrieval" if args.stage == "tfidf" else args.stage]

    last_exit = 0
    for stage_key in order:
        name, stage_fn = stages[stage_key]
        exit_code = run_stage(name, stage_fn)
        if exit_code != 0:
            last_exit = exit_code
            if not args.continue_on_error:
                return exit_code

    return last_exit
