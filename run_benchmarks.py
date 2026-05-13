"""Automated benchmark runner and report generator for the IR pipeline."""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EVAL_DIR = ROOT / "data" / "evaluation"

# Define the configurations you want to benchmark
CONFIGS = [
    {"dataset": "eval_data_cs", "language": "cs", "pipeline": "baseline", "scorer": "tfidf"},
    {"dataset": "eval_data_cs", "language": "cs", "pipeline": "baseline", "scorer": "bm25"},
    {"dataset": "eval_data_cs", "language": "cs", "pipeline": "stemming", "scorer": "tfidf"},
    {"dataset": "eval_data_cs", "language": "cs", "pipeline": "stemming", "scorer": "bm25"},
    {"dataset": "eval_data_cs", "language": "cs", "pipeline": "lemmatization", "scorer": "tfidf"},
    {"dataset": "eval_data_cs", "language": "cs", "pipeline": "lemmatization", "scorer": "bm25"},
    {"dataset": "eval_data_en", "language": "en", "pipeline": "baseline", "scorer": "tfidf"},
    {"dataset": "eval_data_en", "language": "en", "pipeline": "baseline", "scorer": "bm25"},
    {"dataset": "eval_data_en", "language": "en", "pipeline": "stemming", "scorer": "tfidf"},
    {"dataset": "eval_data_en", "language": "en", "pipeline": "stemming", "scorer": "bm25"},
    {"dataset": "eval_data_en", "language": "en", "pipeline": "lemmatization", "scorer": "tfidf"},
    {"dataset": "eval_data_en", "language": "en", "pipeline": "lemmatization", "scorer": "bm25"},
]


def extract_metric(stdout: str, metric_name: str) -> str:
    """Extract a specific metric value from trec_eval stdout."""
    if not stdout:
        return "N/A"
    # trec_eval output format typically looks like: map                     all     0.1234
    pattern = rf"^{metric_name}\s+all\s+([0-9.]+)"
    match = re.search(pattern, stdout, re.MULTILINE)
    if match:
        return match.group(1)
    return "N/A"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmarks and generate a performance report.")
    default_bin = "trec_eval-main/trec_eval.exe" if sys.platform == "win32" else "trec_eval-main/trec_eval"
    parser.add_argument(
        "--trec-eval-bin", 
        default=default_bin, 
        help="Path to the trec_eval binary."
    )
    parser.add_argument(
        "--output", 
        default="PERFORMANCE_REPORT.md", 
        help="Path to save the generated Markdown report."
    )
    return parser.parse_args()


def run_evaluations(trec_eval_bin: str) -> list[dict]:
    """Run the evaluate script for all configurations and collect results."""
    results = []

    for config in CONFIGS:
        print(f"[*] Running benchmark: {config['dataset']} | {config['language']} | {config['pipeline']} | {config['scorer']}")
        
        # Construct command to call evaluate.py
        cmd = [
            sys.executable, str(ROOT / "eval_interface" / "evaluate.py"),
            "--dataset", config["dataset"],
            "--pipeline", config["pipeline"],
            "--language", config["language"],
            "--scorer", config["scorer"],
            "--trec-eval-bin", trec_eval_bin,
            "--run-boolean"
        ]
        
        # Execute evaluate.py
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        
        if proc.returncode != 0:
            print(f"[!] Error running config {config}:")
            print(proc.stderr)
            continue
            
        # Find the latest summary JSON generated for this dataset
        summary_files = list(EVAL_DIR.glob(f"evaluation_summary_{config['dataset']}_*.json"))
        if not summary_files:
            print(f"[!] No summary file found for {config['dataset']}. Skipping.")
            continue
            
        latest_summary_path = max(summary_files, key=lambda p: p.stat().st_mtime)
        
        with latest_summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
            
        trec_stdout = summary.get("trec_eval", {}).get("stdout", "")
        trec_stderr = summary.get("trec_eval", {}).get("stderr", "")
        trec_rc = summary.get("trec_eval", {}).get("returncode")
        
        # Extract key metrics
        results.append({
            "config": config,
            "summary": summary,
            "trec_found": trec_rc == 0,
            "trec_error": trec_stderr,
            "map": extract_metric(trec_stdout, "map"),
            "p10": extract_metric(trec_stdout, "P_10"),
            "ndcg": extract_metric(trec_stdout, "ndcg"),
            "recip_rank": extract_metric(trec_stdout, "recip_rank"),
        })
        
        print(f"    -> Done. Indexing: {summary['timing_seconds']['indexing']}s, Search: {summary['timing_seconds']['ranked_batch_search']}s")
        
    return results


def generate_markdown_report(results: list[dict], output_path: str):
    """Generate a formatted Markdown report from collected metrics."""
    md = [
        "# Evaluation & Performance Report\n",
        f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "## Overview\n",
        "This report aggregates the performance and accuracy metrics across different preprocessing pipelines and languages, evaluated against `trec_eval`.\n"
    ]
    
    # Warning if trec_eval was not found
    missing_trec = [r for r in results if not r["trec_found"]]
    if missing_trec:
        md.append("> **⚠️ Warning:** `trec_eval` binary was not found or failed for some runs. Metrics like MAP and P@10 will display as N/A.\n")

    md.append("## Retrieval Effectiveness (TREC Metrics)\n")
    md.append("| Dataset | Language | Pipeline | Scorer | MAP | P@10 | NDCG | MRR |")
    md.append("|---------|----------|----------|--------|-----|------|------|-----|")
    
    for res in results:
        c = res["config"]
        md.append(f"| {c['dataset']} | {c['language']} | `{c['pipeline']}` | **{c['scorer'].upper()}** | {res['map']} | {res['p10']} | {res['ndcg']} | {res['recip_rank']} |")
        
    md.append("\n## Speed & Indexing Performance\n")
    md.append("| Dataset | Pipeline | Scorer | Docs | Queries | Indexing Time (s) | Ranked Search Time (s) | Boolean Time (s) |")
    md.append("|---------|----------|--------|------|---------|-------------------|------------------------|------------------|")
    
    for res in results:
        c = res["config"]
        meta = res["summary"]["meta"]
        timing = res["summary"]["timing_seconds"]
        bool_time = timing.get("boolean_batch_search")
        bool_time_str = f"{bool_time:.3f}" if bool_time is not None else "N/A"
        
        md.append(f"| {c['dataset']} | `{c['pipeline']}` | **{c['scorer'].upper()}** | {meta['document_count']} | {meta['query_count']} | {timing['indexing']:.3f} | {timing['ranked_batch_search']:.3f} | {bool_time_str} |")

    md.append("\n---\n*Note: Timings reflect local execution and may vary based on hardware.*")

    out_file = Path(output_path)
    with out_file.open("w", encoding="utf-8") as f:
        f.write("\n".join(md))
        
    print(f"\n[+] Successfully generated performance report at: {out_file.resolve()}")


def main() -> int:
    args = parse_args()
    print(f"Starting automated benchmark suite using trec_eval at: '{args.trec_eval_bin}'")
    
    results = run_evaluations(args.trec_eval_bin)
    if results:
        generate_markdown_report(results, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())