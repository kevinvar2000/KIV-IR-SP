import json
from pathlib import Path
import traceback

from app_config import (
    CRAWLER_DATA_FILE,
    INDEX_DIR,
    PIPELINE_OPTIONS,
    PREPROCESSED_DIR,
    ROOT,
)
from crawler.crawler import run_crawler
from indexing.main import run_indexing_stage
from preprocessing.main import run_preprocessing_stage
from retrieval.tfidf_search import run_retrieval_stage
from retrieval.query_interface import run_interactive_query_loop
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


def has_crawler_data() -> bool:
    return CRAWLER_DATA_FILE.exists() and CRAWLER_DATA_FILE.stat().st_size > 0


def safe_load_json_line(line: str) -> dict | None:
    try:
        loaded = json.loads(line)
        if isinstance(loaded, dict):
            return loaded
        return None
    except json.JSONDecodeError:
        return None


def detect_text_keys(data_file: Path, sample_limit: int = 200) -> list[str]:
    if not data_file.exists():
        return []

    key_score: dict[str, int] = {}
    with data_file.open("r", encoding="utf-8") as file_handle:
        for i, line in enumerate(file_handle, start=1):
            if i > sample_limit:
                break
            row = safe_load_json_line(line.strip())
            if not row:
                continue
            for key, value in row.items():
                if isinstance(value, str) and value.strip():
                    key_score[key] = key_score.get(key, 0) + 1

    return [key for key, _ in sorted(key_score.items(), key=lambda x: (-x[1], x[0]))]


def ask_input(prompt: str, default: str | None = None) -> str:
    if default is None:
        value = input(f"{prompt}: ").strip()
    else:
        value = input(f"{prompt} [{default}]: ").strip()
        if not value:
            value = default
    return value


def ask_yes_no(prompt: str, default_yes: bool = True) -> bool:
    default_label = "Y/n" if default_yes else "y/N"
    answer = input(f"{prompt} ({default_label}): ").strip().lower()
    if not answer:
        return default_yes
    return answer in {"y", "yes"}


def choose_from_list(title: str, options: list[str], default_index: int = 1) -> str:
    print(f"\n{title}")
    for idx, option in enumerate(options, start=1):
        print(f"{idx}. {option}")

    choice_raw = ask_input("Choose option number", str(default_index))
    try:
        idx = int(choice_raw)
        if 1 <= idx <= len(options):
            return options[idx - 1]
    except ValueError:
        pass

    print("Invalid choice, using default.")
    return options[default_index - 1]


def choose_pipelines() -> list[str]:
    print("\nAvailable preprocessing pipelines:")
    print("1. baseline (lowercase, remove punct/tags/URLs, remove stopwords, min-length 2)")
    print("2. stemming (baseline + Czech stemming)")
    print("3. lemmatization (baseline + Czech lemmatization)")
    print("4. stemming_no_diacritics (baseline + stemming + remove accents)")
    print("5. lemmatization_no_diacritics (baseline + lemmatization + remove accents)")
    print(f"{len(PIPELINE_OPTIONS) + 1}. all (runs all pipelines)")
    print("\nNote: All pipelines use the baseline steps. Options 2-5 add additional processing on top.")

    raw = ask_input("Choose one or more numbers (comma-separated)", str(len(PIPELINE_OPTIONS) + 1))
    parts = [part.strip() for part in raw.split(",") if part.strip()]

    if not parts:
        return ["all"]

    chosen: list[str] = []
    for part in parts:
        try:
            idx = int(part)
        except ValueError:
            continue
        if idx == len(PIPELINE_OPTIONS) + 1:
            return ["all"]
        if 1 <= idx <= len(PIPELINE_OPTIONS):
            name = PIPELINE_OPTIONS[idx - 1]
            if name not in chosen:
                chosen.append(name)

    return chosen or ["all"]


def run_preprocessing_interactive() -> int:
    if not has_crawler_data():
        print(f"No crawler data found at: {CRAWLER_DATA_FILE}")
        print("Run crawler first.")
        return 1

    text_keys = detect_text_keys(CRAWLER_DATA_FILE)
    if not text_keys:
        print("Could not detect text keys from crawler data.")
        text_key = ask_input("Enter text key manually", "article_text")
    else:
        default_key = "article_text" if "article_text" in text_keys else text_keys[0]
        text_key = choose_from_list("Detected text keys", text_keys, text_keys.index(default_key) + 1)

    pipelines = choose_pipelines()

    return run_stage(
        "preprocessing",
        lambda: run_preprocessing_stage(
            input_path=CRAWLER_DATA_FILE,
            output_dir=PREPROCESSED_DIR,
            text_key=text_key,
            pipeline_selection=pipelines,
        ),
    )


def list_index_files() -> list[Path]:
    if not INDEX_DIR.exists():
        return []
    return sorted(path for path in INDEX_DIR.glob("inverted_index_*.json") if path.is_file())


def run_indexing_interactive() -> int:
    pipeline_name = choose_from_list("Choose pipeline for indexing", PIPELINE_OPTIONS, 1)
    input_default = PREPROCESSED_DIR / f"docs_{pipeline_name}.jsonl"
    output_default = INDEX_DIR / f"inverted_index_{pipeline_name}.json"
    return run_stage(
        "indexing",
        lambda: run_indexing_stage(
            pipeline=pipeline_name,
            input_path=input_default,
            output_path=output_default,
        ),
    )


def run_retrieval_interactive() -> int:
    index_files = list_index_files()

    if index_files:
        print("\nFound index files in data/index:")
        for idx, path in enumerate(index_files, start=1):
            print(f"{idx}. {path.relative_to(ROOT)}")
    else:
        print("\nNo index files found under data/index.")
        if ask_yes_no("Build an index now?", default_yes=True):
            rc = run_indexing_interactive()
            if rc != 0:
                return rc
            index_files = list_index_files()

    if not index_files:
        print("No retrieval index available.")
        return 1

    default_idx = 1
    if len(index_files) > 1:
        for idx, path in enumerate(index_files, start=1):
            if "baseline" in path.name:
                default_idx = idx
                break

    selected = choose_from_list("Choose index file", [str(p.relative_to(ROOT)) for p in index_files], default_idx)
    selected_path = ROOT / selected

    # Use the new interactive query loop
    return run_interactive_query_loop(str(selected_path), pipeline="baseline")


def interactive_mode() -> int:
    print("Interactive pipeline mode")

    if not has_crawler_data():
        print(f"No crawler data found at: {CRAWLER_DATA_FILE}")
        if ask_yes_no("Run crawler first now?", default_yes=True):
            if ask_yes_no("Run crawler in background?", default_yes=False):
                run_crawler_background()
                return 0
            rc = run_stage("crawler", run_crawler)
            if rc != 0:
                return rc
        else:
            print("Nothing to do without data. Exiting.")
            return 0

    while True:
        print("\nWhat do you want to run?")
        print("1. crawler (foreground)")
        print("2. crawler (background)")
        print("3. preprocessing (auto-index)")
        print("4. indexing")
        print("5. retrieval")
        print("6. preprocessing -> retrieval")
        print("7. all (crawler -> preprocessing -> retrieval)")
        print("0. exit")

        choice = ask_input("Choose", "3")

        if choice == "0":
            return 0
        if choice == "1":
            run_stage("crawler", run_crawler)
            continue
        if choice == "2":
            run_crawler_background()
            continue
        if choice == "3":
            run_preprocessing_interactive()
            continue
        if choice == "4":
            run_indexing_interactive()
            continue
        if choice == "5":
            run_retrieval_interactive()
            continue
        if choice == "6":
            rc = run_preprocessing_interactive()
            if rc != 0:
                continue
            if ask_yes_no("Continue with retrieval now?", default_yes=True):
                run_retrieval_interactive()
            continue
        if choice == "7":
            rc = run_stage("crawler", run_crawler)
            if rc != 0:
                continue
            rc = run_preprocessing_interactive()
            if rc != 0:
                continue
            if ask_yes_no("Continue with retrieval now?", default_yes=True):
                run_retrieval_interactive()
            continue

        print("Invalid menu option.")
