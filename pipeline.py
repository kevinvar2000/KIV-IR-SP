import argparse

from app_config import CRAWLER_SCRIPT, PREPROCESSING_SCRIPT, TFIDF_SCRIPT
from runner import run_crawler_background, run_step


def run_non_interactive(args: argparse.Namespace) -> int:
    if args.stage == "crawler" and args.crawler_background:
        return run_crawler_background(CRAWLER_SCRIPT)

    stages = {
        "crawler": ("crawler", CRAWLER_SCRIPT, None),
        "preprocessing": (
            "preprocessing",
            PREPROCESSING_SCRIPT,
            [
                "--input",
                args.input_data,
                "--output-dir",
                args.preprocess_output_dir,
                "--pipelines",
                *args.pipelines,
                *(["--text-key", args.text_key] if args.text_key else []),
            ],
        ),
        "tfidf": ("tfidf", TFIDF_SCRIPT, args.tfidf_files),
    }

    if args.stage == "all":
        order = ["crawler", "preprocessing", "tfidf"]
    else:
        order = [args.stage]

    last_exit = 0
    for stage_key in order:
        name, script_path, extra_args = stages[stage_key]
        exit_code = run_step(name, script_path, extra_args)
        if exit_code != 0:
            last_exit = exit_code
            if not args.continue_on_error:
                return exit_code

    return last_exit
