"""Helpers for launching long-running pipeline stages in the background."""

import subprocess
import sys

from app_config import CRAWLER_LOG_FILE, CRAWLER_SCRIPT, ROOT


def run_crawler_background() -> int:
    """Start the crawler as a background process and stream logs to file."""
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
