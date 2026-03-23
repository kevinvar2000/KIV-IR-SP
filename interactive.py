import json
from pathlib import Path

from app_config import (
    CRAWLER_DATA_FILE,
    CRAWLER_SCRIPT,
    PIPELINE_OPTIONS,
    PREPROCESSED_DIR,
    RETRIEVAL_SCRIPT,
    ROOT,
)
from runner import run_crawler_background, run_step


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

    cmd_args = [
        "--input",
        str(CRAWLER_DATA_FILE),
        "--text-key",
        text_key,
        "--output-dir",
        str(PREPROCESSED_DIR),
        "--pipelines",
        *pipelines,
    ]

    from app_config import PREPROCESSING_SCRIPT

    return run_step("preprocessing", PREPROCESSING_SCRIPT, cmd_args)


def list_data_collection_files() -> list[Path]:
    data_dir = ROOT / "data"
    if not data_dir.exists():
        return []

    return sorted(path for path in data_dir.rglob("*.txt") if path.is_file())


def run_retrieval_interactive() -> int:
    discovered_files = list_data_collection_files()

    if discovered_files:
        print("\nFound collection files in data/:")
        for idx, path in enumerate(discovered_files, start=1):
            print(f"{idx}. {path.relative_to(ROOT)}")
    else:
        print("\nNo .txt collection files found under data/.")

    raw = ask_input(
        "Select numbers (comma-separated) or write file paths from project root",
        "",
    )

    files: list[str] = []
    if raw.strip():
        for item in [part.strip() for part in raw.split(",") if part.strip()]:
            if item.isdigit() and discovered_files:
                idx = int(item)
                if 1 <= idx <= len(discovered_files):
                    files.append(str(discovered_files[idx - 1]))
            else:
                files.append(str(ROOT / item))
    else:
        files = [str(path) for path in discovered_files]

    if not files:
        print("No retrieval input files selected.")
        return 1

    return run_step("retrieval", RETRIEVAL_SCRIPT, files)


def interactive_mode() -> int:
    print("Interactive pipeline mode")

    if not has_crawler_data():
        print(f"No crawler data found at: {CRAWLER_DATA_FILE}")
        if ask_yes_no("Run crawler first now?", default_yes=True):
            if ask_yes_no("Run crawler in background?", default_yes=False):
                run_crawler_background(CRAWLER_SCRIPT)
                return 0
            rc = run_step("crawler", CRAWLER_SCRIPT)
            if rc != 0:
                return rc
        else:
            print("Nothing to do without data. Exiting.")
            return 0

    while True:
        print("\nWhat do you want to run?")
        print("1. crawler (foreground)")
        print("2. crawler (background)")
        print("3. preprocessing")
        print("4. retrieval")
        print("5. preprocessing -> retrieval")
        print("6. all (crawler -> preprocessing -> retrieval)")
        print("0. exit")

        choice = ask_input("Choose", "3")

        if choice == "0":
            return 0
        if choice == "1":
            run_step("crawler", CRAWLER_SCRIPT)
            continue
        if choice == "2":
            run_crawler_background(CRAWLER_SCRIPT)
            continue
        if choice == "3":
            run_preprocessing_interactive()
            continue
        if choice == "4":
            run_retrieval_interactive()
            continue
        if choice == "5":
            rc = run_preprocessing_interactive()
            if rc == 0 and ask_yes_no("Continue with retrieval now?", default_yes=True):
                run_retrieval_interactive()
            continue
        if choice == "6":
            rc = run_step("crawler", CRAWLER_SCRIPT)
            if rc != 0:
                continue
            rc = run_preprocessing_interactive()
            if rc != 0:
                continue
            if ask_yes_no("Continue with retrieval now?", default_yes=True):
                run_retrieval_interactive()
            continue

        print("Invalid menu option.")
