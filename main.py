import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CRAWLER_SCRIPT = ROOT / "crawler" / "crawler.py"
PREPROCESSING_SCRIPT = ROOT / "preprocessing" / "main.py"
TFIDF_SCRIPT = ROOT / "tfidf" / "tfidf_search.py"

CRAWLER_DATA_FILE = ROOT / "data" / "crawler" / "crawled_pages.json"
CRAWLER_LOG_FILE = ROOT / "data" / "crawler" / "crawler.log"
PREPROCESSED_DIR = ROOT / "data" / "preprocessed"

PIPELINE_OPTIONS = [
	"baseline",
	"stemming",
	"lemmatization",
	"stemming_no_diacritics",
	"lemmatization_no_diacritics",
]


def run_step(name: str, script_path: Path, args: list[str] | None = None) -> int:
	cmd = [sys.executable, str(script_path)]
	if args:
		cmd.extend(args)

	print(f"\n[{name}] running: {' '.join(cmd)}")
	result = subprocess.run(cmd, cwd=str(ROOT))
	if result.returncode != 0:
		print(f"[{name}] failed with exit code {result.returncode}")
	else:
		print(f"[{name}] finished")
	return result.returncode


def run_crawler_background() -> int:
	CRAWLER_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
	cmd = [sys.executable, str(CRAWLER_SCRIPT)]
	creation_flags = 0
	if sys.platform.startswith("win") and hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
		creation_flags |= subprocess.CREATE_NEW_PROCESS_GROUP

	with CRAWLER_LOG_FILE.open("a", encoding="utf-8") as log_file:
		proc = subprocess.Popen(
			cmd,
			cwd=str(ROOT),
			stdout=log_file,
			stderr=subprocess.STDOUT,
			creationflags=creation_flags,
		)

	print(f"[crawler] started in background with PID={proc.pid}")
	print(f"[crawler] log file: {CRAWLER_LOG_FILE}")
	return 0


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
	with data_file.open("r", encoding="utf-8") as f:
		for i, line in enumerate(f, start=1):
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
	for idx, option in enumerate(PIPELINE_OPTIONS, start=1):
		print(f"{idx}. {option}")
	print(f"{len(PIPELINE_OPTIONS) + 1}. all")

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
		"--input", str(CRAWLER_DATA_FILE),
		"--text-key", text_key,
		"--output-dir", str(PREPROCESSED_DIR),
		"--pipelines",
		*pipelines,
	]

	return run_step("preprocessing", PREPROCESSING_SCRIPT, cmd_args)


def run_tfidf_interactive() -> int:
	default_files = [
		str(path) for path in sorted((ROOT / "tfidf").glob("*.txt"))
	]

	if default_files:
		print("\nDetected TF-IDF candidate files:")
		for path in default_files:
			print(f"- {path}")

	raw = ask_input(
		"Enter TF-IDF input files (comma-separated) or leave blank for defaults",
		"",
	)

	if raw.strip():
		files = [item.strip() for item in raw.split(",") if item.strip()]
	else:
		files = default_files

	return run_step("tfidf", TFIDF_SCRIPT, files)


def interactive_mode() -> int:
	print("Interactive pipeline mode")

	if not has_crawler_data():
		print(f"No crawler data found at: {CRAWLER_DATA_FILE}")
		if ask_yes_no("Run crawler first now?", default_yes=True):
			if ask_yes_no("Run crawler in background?", default_yes=False):
				run_crawler_background()
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
		print("4. tfidf")
		print("5. preprocessing -> tfidf")
		print("6. all (crawler -> preprocessing -> tfidf)")
		print("0. exit")

		choice = ask_input("Choose", "3")

		if choice == "0":
			return 0
		if choice == "1":
			run_step("crawler", CRAWLER_SCRIPT)
			continue
		if choice == "2":
			run_crawler_background()
			continue
		if choice == "3":
			run_preprocessing_interactive()
			continue
		if choice == "4":
			run_tfidf_interactive()
			continue
		if choice == "5":
			rc = run_preprocessing_interactive()
			if rc == 0 and ask_yes_no("Continue with TF-IDF now?", default_yes=True):
				run_tfidf_interactive()
			continue
		if choice == "6":
			rc = run_step("crawler", CRAWLER_SCRIPT)
			if rc != 0:
				continue
			rc = run_preprocessing_interactive()
			if rc != 0:
				continue
			if ask_yes_no("Continue with TF-IDF now?", default_yes=True):
				run_tfidf_interactive()
			continue

		print("Invalid menu option.")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Run project pipeline stages from one CLI entry point.",
	)
	parser.add_argument(
		"stage",
		nargs="?",
		default="interactive",
		choices=["interactive", "all", "crawler", "preprocessing", "tfidf"],
		help="Which stage to run. Default is interactive mode.",
	)
	parser.add_argument(
		"--tfidf-files",
		nargs="*",
		default=None,
		help="Optional input files for TF-IDF stage.",
	)
	parser.add_argument(
		"--text-key",
		default=None,
		help="Text key used for preprocessing (non-interactive mode).",
	)
	parser.add_argument(
		"--pipelines",
		nargs="*",
		default=["all"],
		help="Preprocessing pipelines for non-interactive mode.",
	)
	parser.add_argument(
		"--input-data",
		default=str(CRAWLER_DATA_FILE),
		help="Input data file for preprocessing in non-interactive mode.",
	)
	parser.add_argument(
		"--preprocess-output-dir",
		default=str(PREPROCESSED_DIR),
		help="Output directory for preprocessing artifacts.",
	)
	parser.add_argument(
		"--crawler-background",
		action="store_true",
		help="Run crawler in background (only for stage=crawler).",
	)
	parser.add_argument(
		"--continue-on-error",
		action="store_true",
		help="Continue with next stages even if one fails (applies to stage=all).",
	)
	return parser.parse_args()


def main() -> int:
	args = parse_args()

	if args.stage == "interactive":
		return interactive_mode()

	if args.stage == "crawler" and args.crawler_background:
		return run_crawler_background()

	stages = {
		"crawler": ("crawler", CRAWLER_SCRIPT, None),
		"preprocessing": (
			"preprocessing",
			PREPROCESSING_SCRIPT,
			[
				"--input", args.input_data,
				"--output-dir", args.preprocess_output_dir,
				"--pipelines", *args.pipelines,
				*( ["--text-key", args.text_key] if args.text_key else [] ),
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


if __name__ == "__main__":
	raise SystemExit(main())
