import json
from pathlib import Path
import traceback

from app_config import (
    INDEX_DIR,
    PIPELINE_OPTIONS,
    PREPROCESSED_DIR,
    ROOT,
)
from crawler.crawler import run_crawler
from evaluation.trec_eval import run_trec_evaluation
from indexing.main import run_indexing_stage
from preprocessing.main import run_preprocessing_stage
from retrieval.query_interface import run_interactive_query_loop
from runner import run_crawler_background


DIVIDER = "=" * 60


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


def ask_yes_no_or_nav(prompt: str, default_yes: bool = True) -> bool | str:
    default_label = "Y/n" if default_yes else "y/N"
    answer = input(f"{prompt} ({default_label}, back/home): ").strip().lower()
    if not answer:
        return default_yes
    if answer in {"back", "home"}:
        return answer
    return answer in {"y", "yes"}


def choose_from_list(title: str, options: list[str], default_index: int = 1) -> str:
    while True:
        print(f"\n{title}")
        print(DIVIDER)
        for idx, option in enumerate(options, start=1):
            print(f"{idx}. {option}")
        print(DIVIDER)

        choice_raw = ask_input("Choose option number (or back/home)", str(default_index))
        lowered = choice_raw.lower()
        if lowered in {"back", "home"}:
            return lowered
        try:
            idx = int(choice_raw)
            if 1 <= idx <= len(options):
                return options[idx - 1]
        except ValueError:
            pass

        print("Invalid choice, please try again.")


def list_preprocessing_source_files() -> list[Path]:
    crawler_dir = ROOT / "data" / "crawler"
    if not crawler_dir.exists():
        return []

    candidates = [
        path
        for path in sorted(crawler_dir.iterdir())
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}
    ]
    return candidates


def choose_preprocessing_input_path() -> Path | None:
    source_files = list_preprocessing_source_files()

    while True:
        print("\nChoose source file for preprocessing")
        for idx, path in enumerate(source_files, start=1):
            print(f"{idx}. {path.relative_to(ROOT)}")
        print(f"{len(source_files) + 1}. type custom file path")

        choice = ask_input("Choose (or back/home)", "1" if source_files else str(len(source_files) + 1))
        lowered = choice.lower()
        if lowered in {"back", "home"}:
            return None

        try:
            idx = int(choice)
        except ValueError:
            idx = -1

        if 1 <= idx <= len(source_files):
            return source_files[idx - 1]

        if idx == len(source_files) + 1 or not source_files:
            while True:
                raw_path = ask_input("Enter file path to preprocess")
                lowered_path = raw_path.lower()
                if lowered_path in {"back", "home"}:
                    return None

                candidate = Path(raw_path)
                if candidate.exists() and candidate.is_file():
                    return candidate

                print(f"File not found: {candidate}")

        print("Invalid choice, please try again.")


def choose_language() -> str | None:
    selected = choose_from_list(
        "Choose document language",
        ["czech (cs)", "slovak (sk)", "english (en)"],
        1,
    )
    if selected in {"back", "home"}:
        return None
    return {"czech (cs)": "cs", "slovak (sk)": "sk", "english (en)": "en"}[selected]


def choose_pipelines(language: str) -> list[str]:
    print(f"\nAvailable preprocessing pipelines for {language}:")
    print("1. baseline (lowercase, remove punct/tags/URLs, remove stopwords, min-length 2)")
    print("2. stemming (baseline + language-aware stemming/lemmatization)")
    print("3. lemmatization (baseline + language-aware lemmatization)")
    print("4. stemming_no_diacritics (baseline + stemming + remove accents)")
    print("5. lemmatization_no_diacritics (baseline + lemmatization + remove accents)")
    print(f"{len(PIPELINE_OPTIONS) + 1}. all (runs all pipelines)")
    print("\nNote: Stopwords, stemming, and lemmatization follow the selected language.")

    while True:
        raw = ask_input("Choose one or more numbers (comma-separated)", str(len(PIPELINE_OPTIONS) + 1))
        if raw.lower() in {"back", "home"}:
            return [raw.lower()]
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

        if chosen:
            return chosen

        print("Invalid choice, please try again.")


def run_preprocessing_interactive() -> int:
    input_path = choose_preprocessing_input_path()
    if input_path is None:
        return 0

    language = choose_language()
    if language is None:
        return 0

    text_keys = detect_text_keys(input_path)
    if not text_keys:
        print(f"Could not detect text keys from input file: {input_path}")
        text_key = ask_input("Enter text key manually", "article_text")
        if text_key.lower() in {"back", "home"}:
            return 0
    else:
        default_key = "article_text" if "article_text" in text_keys else text_keys[0]
        text_key = choose_from_list("Detected text keys", text_keys, text_keys.index(default_key) + 1)
        if text_key in {"back", "home"}:
            return 0

    pipelines = choose_pipelines(language)
    if pipelines == ["back"] or pipelines == ["home"]:
        return 0

    return run_stage(
        "preprocessing",
        lambda: run_preprocessing_stage(
            input_path=input_path,
            output_dir=PREPROCESSED_DIR,
            text_key=text_key,
            pipeline_selection=pipelines,
            language=language,
        ),
    )


def run_crawler_interactive() -> int:
    choice = ask_yes_no_or_nav("Run crawler in background?", default_yes=False)
    if choice in {"back", "home"}:
        return 0
    if choice:
        run_crawler_background()
        return 0
    return run_stage("crawler", run_crawler)


def list_index_files() -> list[Path]:
    if not INDEX_DIR.exists():
        return []
    return sorted(path for path in INDEX_DIR.glob("inverted_index_*.json") if path.is_file())


def list_preprocessed_docs_files() -> list[Path]:
    if not PREPROCESSED_DIR.exists():
        return []
    return sorted(path for path in PREPROCESSED_DIR.glob("docs_*.jsonl") if path.is_file())


def choose_preprocessed_docs_file() -> Path | None:
    docs_files = list_preprocessed_docs_files()

    while True:
        print("\nChoose preprocessed docs file")
        for idx, path in enumerate(docs_files, start=1):
            print(f"{idx}. {path.relative_to(ROOT)}")
        print(f"{len(docs_files) + 1}. type custom file path")

        choice = ask_input("Choose (or back/home)", "1" if docs_files else str(len(docs_files) + 1))
        lowered = choice.lower()
        if lowered in {"back", "home"}:
            return None

        try:
            idx = int(choice)
        except ValueError:
            idx = -1

        if 1 <= idx <= len(docs_files):
            return docs_files[idx - 1]

        if idx == len(docs_files) + 1 or not docs_files:
            while True:
                raw_path = ask_input("Enter preprocessed docs file path")
                lowered_path = raw_path.lower()
                if lowered_path in {"back", "home"}:
                    return None

                candidate = Path(raw_path)
                if candidate.exists() and candidate.is_file():
                    return candidate

                print(f"File not found: {candidate}")

        print("Invalid choice, please try again.")


def run_indexing_interactive() -> int:
    print("\nIndexing expects a preprocessed docs_<pipeline>.jsonl file.")
    print("Use this only after preprocessing has already produced normalized documents.")
    input_path = choose_preprocessed_docs_file()
    if input_path is None:
        return 0

    pipeline_name = input_path.stem.removeprefix("docs_")
    if not pipeline_name:
        pipeline_name = "baseline"

    output_default = INDEX_DIR / f"inverted_index_{pipeline_name}.json"
    return run_stage(
        "indexing",
        lambda: run_indexing_stage(
            pipeline=pipeline_name,
            input_path=input_path,
            output_path=output_default,
        ),
    )


def run_retrieval_interactive() -> int:
    index_files = list_index_files()

    if not index_files:
        print("\nNo index files found under data/index.")
        build_now = ask_yes_no_or_nav("Build an index now?", default_yes=True)
        if build_now in {"back", "home"}:
            return 0
        if build_now:
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
    if selected in {"back", "home"}:
        return 0
    selected_path = ROOT / selected
    pipeline_name = selected_path.stem.removeprefix("inverted_index_") or "baseline"

    return run_interactive_query_loop(str(selected_path), pipeline=pipeline_name)


def run_evaluation_interactive() -> int:
    print("\nEvaluation exports TREC run files from documents and queries JSON/JSONL files.")

    while True:
        documents_path = ask_input("Enter documents file path (back/home to cancel)")
        if documents_path.lower() in {"back", "home"}:
            return 0
        documents_file = Path(documents_path)
        if documents_file.exists() and documents_file.is_file():
            break
        print(f"File not found: {documents_file}")

    while True:
        queries_path = ask_input("Enter queries file path (back/home to cancel)")
        if queries_path.lower() in {"back", "home"}:
            return 0
        queries_file = Path(queries_path)
        if queries_file.exists() and queries_file.is_file():
            break
        print(f"File not found: {queries_file}")

    language = choose_language()
    if language is None:
        return 0

    pipeline_choice = choose_from_list(
        "Choose preprocessing pipeline",
        list(PIPELINE_OPTIONS),
        1,
    )
    if pipeline_choice in {"back", "home"}:
        return 0

    model_choice = choose_from_list("Choose retrieval model", ["tf-idf", "boolean"], 1)
    if model_choice in {"back", "home"}:
        return 0

    top_k_raw = ask_input("Results per query", "1000")
    if top_k_raw.lower() in {"back", "home"}:
        return 0
    try:
        top_k = int(top_k_raw)
    except ValueError:
        print("Invalid number, using 1000.")
        top_k = 1000

    default_output = ROOT / "data" / "evaluation" / "results.txt"
    output_raw = ask_input("Output TREC file path", str(default_output))
    if output_raw.lower() in {"back", "home"}:
        return 0
    output_path = Path(output_raw)

    run_id = ask_input("Run id", "run")
    if run_id.lower() in {"back", "home"}:
        return 0

    qrels_path = ask_input("Optional qrels file path (empty to skip)", "")
    if qrels_path.lower() in {"back", "home"}:
        return 0

    return run_stage(
        "evaluation",
        lambda: run_trec_evaluation(
            documents_path=documents_file,
            queries_path=queries_file,
            output_path=output_path,
            pipeline=pipeline_choice,
            language=language,
            model="tfidf" if model_choice == "tf-idf" else "boolean",
            top_k=top_k,
            run_id=run_id,
            qrels_path=qrels_path or None,
        ),
    )


def interactive_mode() -> int:
    print("Interactive pipeline mode")

    while True:
        print("\nWhat do you want to run?")
        print(DIVIDER)
        print("1. crawler")
        print("2. preprocessing + indexing")
        print("3. indexing preprocessed docs")
        print("4. retrieval")
        print("5. evaluation export")
        print("0. exit")
        print(DIVIDER)

        choice = ask_input("Choose", "2")

        if choice == "0":
            return 0
        if choice == "1":
            run_crawler_interactive()
            continue
        if choice == "2":
            run_preprocessing_interactive()
            continue
        if choice == "3":
            run_indexing_interactive()
            continue
        if choice == "4":
            run_retrieval_interactive()
            continue
        if choice == "5":
            run_evaluation_interactive()
            continue

        print("Invalid menu option.")
