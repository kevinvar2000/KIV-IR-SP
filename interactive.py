"""Interactive CLI orchestration for crawl, preprocess, index, and retrieval stages."""

import json
from pathlib import Path
import traceback
import interactive_config as ui

from app_config import (
    INDEX_DIR,
    PIPELINE_OPTIONS,
    PREPROCESSED_DIR,
    ROOT,
)
from crawler.crawler import run_crawler
from indexing.main import run_indexing_stage
from preprocessing.main import run_preprocessing_stage
from retrieval.query_interface import run_interactive_query_loop
from runner import run_crawler_background


def _sorted_key_names(key_score: dict[str, int]) -> list[str]:
    """Return key names sorted by descending frequency and then alphabetically."""
    return [key for key, _ in sorted(key_score.items(), key=lambda x: (-x[1], x[0]))]


def _collect_json_files(directory: Path, recursive: bool = False) -> list[Path]:
    """Collect JSON and JSONL files from a directory."""
    if not directory.exists() or not directory.is_dir():
        return []
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}
    )


def _choose_file_from_menu(
    files: list[Path],
    title: str,
    custom_prompt: str,
) -> Path | None:
    """Display a file selection menu with optional manual path entry."""
    while True:
        print(title)
        for idx, path in enumerate(files, start=1):
            print(f"{idx}. {path.relative_to(ROOT)}")
        print(f"{len(files) + 1}. {ui.LABEL_TYPE_CUSTOM_FILE}")

        choice = ask_input(ui.PROMPT_CHOOSE_NAV, "1" if files else str(len(files) + 1))
        lowered = choice.lower()
        if lowered in {"back", "home"}:
            return None

        try:
            idx = int(choice)
        except ValueError:
            idx = -1

        if 1 <= idx <= len(files):
            return files[idx - 1]

        if idx == len(files) + 1 or not files:
            while True:
                raw_path = ask_input(custom_prompt)
                lowered_path = raw_path.lower()
                if lowered_path in {"back", "home"}:
                    return None

                candidate = Path(raw_path)
                if candidate.exists() and candidate.is_file():
                    return candidate

                print(ui.ERROR_FILE_NOT_FOUND.format(path=candidate))

        print(ui.INVALID_CHOICE)


def run_stage(name: str, fn) -> int:
    """Execute a stage callable with unified logging and error handling."""
    print(ui.STAGE_RUNNING.format(name=name))
    try:
        result = fn()
        exit_code = int(result) if result is not None else 0
    except Exception:
        print(ui.STAGE_UNHANDLED_EXCEPTION.format(name=name))
        traceback.print_exc()
        return 1

    if exit_code != 0:
        print(ui.STAGE_FAILED_EXIT_CODE.format(name=name, exit_code=exit_code))
    else:
        print(ui.STAGE_FINISHED.format(name=name))
    return exit_code


def safe_load_json_line(line: str) -> dict | None:
    """Parse a JSON line and return only dictionary records."""
    try:
        loaded = json.loads(line)
        if isinstance(loaded, dict):
            return loaded
        return None
    except json.JSONDecodeError:
        return None


def detect_text_keys(data_file: Path, sample_limit: int = 200) -> list[str]:
    """Detect candidate text fields by sampling records from JSON/JSONL input."""
    if not data_file.exists():
        return []

    key_score: dict[str, int] = {}

    def collect_row_keys(row: dict) -> None:
        """Collect non-empty string keys from a single record."""
        for key, value in row.items():
            if isinstance(value, str) and value.strip():
                key_score[key] = key_score.get(key, 0) + 1

    # Try standard JSON first so array/object files (e.g. documents.json) work.
    try:
        with data_file.open("r", encoding="utf-8") as file_handle:
            loaded = json.load(file_handle)

        if isinstance(loaded, dict):
            collect_row_keys(loaded)
            return _sorted_key_names(key_score)

        if isinstance(loaded, list):
            for row in loaded[:sample_limit]:
                if isinstance(row, dict):
                    collect_row_keys(row)
            return _sorted_key_names(key_score)
    except (json.JSONDecodeError, OSError):
        # Fall back to JSONL-style parsing below.
        pass

    # Fallback for JSONL files: one JSON object per line.
    with data_file.open("r", encoding="utf-8") as file_handle:
        for i, line in enumerate(file_handle, start=1):
            if i > sample_limit:
                break
            row = safe_load_json_line(line.strip())
            if not row:
                continue
            collect_row_keys(row)

    return _sorted_key_names(key_score)


def ask_input(prompt: str, default: str | None = None) -> str:
    """Read trimmed user input with optional default value."""
    if default is None:
        value = input(f"{prompt}: ").strip()
    else:
        value = input(f"{prompt} [{default}]: ").strip()
        if not value:
            value = default
    return value


def ask_yes_no(prompt: str, default_yes: bool = True) -> bool:
    """Prompt for a yes/no answer with configurable default."""
    default_label = "Y/n" if default_yes else "y/N"
    answer = input(f"{prompt} ({default_label}): ").strip().lower()
    if not answer:
        return default_yes
    return answer in {"y", "yes"}


def ask_yes_no_or_nav(prompt: str, default_yes: bool = True) -> bool | str:
    """Prompt for yes/no and allow navigation commands back/home."""
    default_label = "Y/n" if default_yes else "y/N"
    answer = input(f"{prompt} ({default_label}, back/home): ").strip().lower()
    if not answer:
        return default_yes
    if answer in {"back", "home"}:
        return answer
    return answer in {"y", "yes"}


def choose_from_list(title: str, options: list[str], default_index: int = 1) -> str:
    """Render an option menu and return selected value or navigation command."""
    while True:
        print(f"\n{title}")
        print(ui.DIVIDER)
        for idx, option in enumerate(options, start=1):
            print(f"{idx}. {option}")
        print(ui.DIVIDER)

        choice_raw = ask_input(ui.PROMPT_CHOOSE_NUMBER_NAV, str(default_index))
        lowered = choice_raw.lower()
        if lowered in {"back", "home"}:
            return lowered
        try:
            idx = int(choice_raw)
            if 1 <= idx <= len(options):
                return options[idx - 1]
        except ValueError:
            pass

        print(ui.INVALID_CHOICE)


def list_preprocessing_source_files() -> list[Path]:
    """List candidate source files used by preprocessing."""
    data_dir = ROOT / "data"
    candidates = _collect_json_files(ROOT / "data" / "crawler")

    if data_dir.exists() and data_dir.is_dir():
        for eval_subdir in sorted(data_dir.glob("eval_data_*")):
            candidates.extend(_collect_json_files(eval_subdir, recursive=True))

    return sorted(candidates)


def choose_preprocessing_input_path() -> Path | None:
    """Choose or manually enter input file path for preprocessing."""
    return _choose_file_from_menu(
        files=list_preprocessing_source_files(),
        title=ui.TITLE_PREPROCESS_SOURCE,
        custom_prompt=ui.PROMPT_ENTER_FILE_TO_PREPROCESS,
    )


def choose_language() -> str | None:
    """Choose normalized language code for preprocessing."""
    selected = choose_from_list(
        ui.TITLE_LANGUAGE,
        ui.LANGUAGE_OPTIONS,
        1,
    )
    if selected in {"back", "home"}:
        return None
    return ui.LANGUAGE_MAP[selected]


def choose_pipelines(language: str) -> list[str]:
    """Choose one or more preprocessing pipelines for the given language."""
    print(ui.PIPELINE_TITLE.format(language=language))
    for line in ui.PIPELINE_LINES:
        print(line)
    print(ui.PIPELINE_ALL.format(all_index=len(PIPELINE_OPTIONS) + 1))
    print(ui.PIPELINE_NOTE)

    while True:
        raw = ask_input(ui.PROMPT_CHOOSE_PIPELINES, str(len(PIPELINE_OPTIONS) + 1))
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

        print(ui.INVALID_CHOICE)


def run_preprocessing_interactive() -> int:
    """Run interactive preprocessing flow and trigger selected pipelines."""
    input_path = choose_preprocessing_input_path()
    if input_path is None:
        return 0

    language = choose_language()
    if language is None:
        return 0

    text_keys = detect_text_keys(input_path)
    if not text_keys:
        print(ui.ERROR_COULD_NOT_DETECT_TEXT_KEYS.format(path=input_path))
        text_key = ask_input(ui.PROMPT_ENTER_TEXT_KEY, "article_text")
        if text_key.lower() in {"back", "home"}:
            return 0
    else:
        default_key = "article_text" if "article_text" in text_keys else text_keys[0]
        text_key = choose_from_list(ui.TITLE_DETECTED_TEXT_KEYS, text_keys, text_keys.index(default_key) + 1)
        if text_key in {"back", "home"}:
            return 0

    pipelines = choose_pipelines(language)
    if pipelines == ["back"] or pipelines == ["home"]:
        return 0

    write_vocab = ask_yes_no_or_nav(ui.PROMPT_WRITE_VOCAB, default_yes=False)
    if write_vocab in {"back", "home"}:
        return 0

    return run_stage(
        "preprocessing",
        lambda: run_preprocessing_stage(
            input_path=input_path,
            output_dir=PREPROCESSED_DIR,
            text_key=text_key,
            pipeline_selection=pipelines,
            language=language,
            write_vocab=bool(write_vocab),
        ),
    )


def run_crawler_interactive() -> int:
    """Run crawler either in background mode or foreground stage mode."""
    choice = ask_yes_no_or_nav(ui.PROMPT_RUN_CRAWLER_BACKGROUND, default_yes=False)
    if choice in {"back", "home"}:
        return 0
    if choice:
        run_crawler_background()
        return 0
    return run_stage("crawler", run_crawler)


def list_index_files() -> list[Path]:
    """List persisted inverted index files available for retrieval."""
    if not INDEX_DIR.exists():
        return []
    return sorted(path for path in INDEX_DIR.glob("inverted_index_*.json") if path.is_file())


def list_preprocessed_docs_files() -> list[Path]:
    """List preprocessed documents files that can be indexed."""
    if not PREPROCESSED_DIR.exists():
        return []
    return sorted(path for path in PREPROCESSED_DIR.glob("docs_*.jsonl") if path.is_file())


def choose_preprocessed_docs_file() -> Path | None:
    """Choose or manually enter a preprocessed docs file for indexing."""
    return _choose_file_from_menu(
        files=list_preprocessed_docs_files(),
        title=ui.TITLE_PREPROCESSED_DOCS,
        custom_prompt=ui.PROMPT_ENTER_PREPROCESSED_DOCS_PATH,
    )


def run_indexing_interactive() -> int:
    """Run indexing using an interactively selected preprocessed docs file."""
    print(ui.INDEXING_EXPECTS_LINE)
    print(ui.INDEXING_USE_AFTER_PREPROCESS_LINE)
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
    """Run retrieval against a selected index, building one if needed."""
    index_files = list_index_files()

    if not index_files:
        print(ui.NO_INDEX_FILES_FOUND)
        build_now = ask_yes_no_or_nav(ui.PROMPT_BUILD_INDEX_NOW, default_yes=True)
        if build_now in {"back", "home"}:
            return 0
        if build_now:
            rc = run_indexing_interactive()
            if rc != 0:
                return rc
            index_files = list_index_files()

    if not index_files:
        print(ui.NO_RETRIEVAL_INDEX)
        return 1

    default_idx = 1
    if len(index_files) > 1:
        for idx, path in enumerate(index_files, start=1):
            if "baseline" in path.name:
                default_idx = idx
                break

    selected = choose_from_list(ui.TITLE_CHOOSE_INDEX_FILE, [str(p.relative_to(ROOT)) for p in index_files], default_idx)
    if selected in {"back", "home"}:
        return 0
    selected_path = ROOT / selected
    pipeline_name = selected_path.stem.removeprefix("inverted_index_") or "baseline"

    return run_interactive_query_loop(str(selected_path), pipeline=pipeline_name)


def interactive_mode() -> int:
    """Main interactive loop for selecting and running pipeline stages."""
    print(ui.INTERACTIVE_MODE_TITLE)

    while True:
        try:
            print(ui.MENU_WHAT_TO_RUN)
            print(ui.DIVIDER)
            for line in ui.MAIN_MENU_LINES:
                print(line)
            print(ui.DIVIDER)

            choice = ask_input(ui.PROMPT_MAIN_CHOOSE, ui.PROMPT_MAIN_DEFAULT)

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

            print(ui.INVALID_MENU_OPTION)
        except KeyboardInterrupt:
            print(ui.INTERRUPTED_EXIT)
            return 0
