import subprocess
import sys
from pathlib import Path

from app_config import CRAWLER_LOG_FILE, ROOT


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


def run_crawler_background(script_path: Path) -> int:
    CRAWLER_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(script_path)]
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
