#!/usr/bin/env python3
"""Generate daily benchmark artifacts and delta reports.

Calls evaluate_meta_stack.py via subprocess to produce the canonical
model_benchmark_YYYY-MM-DD.md, then computes a compact delta report
comparing today's metrics against yesterday's artifact.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"
SEASON_START = "2026-02-18"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Daily benchmark artifact generation")
    parser.add_argument(
        "--retain",
        type=int,
        default=30,
        help="Keep only last N daily artifacts (default 30)",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Override today's date (YYYY-MM-DD), mainly for testing",
    )
    return parser.parse_args(argv)


def run_benchmark(today: str) -> Path:
    """Run evaluate_meta_stack.py and return path to the generated artifact."""
    out_path = ARTIFACTS_DIR / f"model_benchmark_{today}.md"
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate_meta_stack.py"),
        "--start-date", SEASON_START,
        "--end-date", today,
        "--out", str(out_path),
    ]
    subprocess.run(cmd, check=True)
    return out_path


def parse_leaderboard(text: str) -> dict[str, dict[str, float]]:
    """Extract leak-safe per-model leaderboard metrics from benchmark markdown.

    Returns {model_name: {accuracy, brier, logloss, ece}}.
    """
    metrics: dict[str, dict[str, float]] = {}
    in_section = False
    for line in text.splitlines():
        if "## Leak-safe per-model leaderboard" in line:
            in_section = True
            continue
        if in_section and line.startswith("##"):
            break
        if not in_section:
            continue
        # Match table rows: | model | n | accuracy | brier | logloss | ece |
        m = re.match(
            r"\|\s*(\S+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|",
            line,
        )
        if m:
            model = m.group(1)
            metrics[model] = {
                "n": int(m.group(2)),
                "accuracy": float(m.group(3)),
                "brier": float(m.group(4)),
                "logloss": float(m.group(5)),
                "ece": float(m.group(6)),
            }
    return metrics


def generate_delta(today_path: Path, yesterday_path: Path, today_str: str) -> Path:
    """Compare today's and yesterday's benchmark and write delta report."""
    today_metrics = parse_leaderboard(today_path.read_text(encoding="utf-8"))
    yesterday_metrics = parse_leaderboard(yesterday_path.read_text(encoding="utf-8"))

    lines = [
        f"# Benchmark Delta — {today_str}",
        "",
        f"Comparing against previous artifact: `{yesterday_path.name}`",
        "",
        "| Model | Accuracy | \u0394Acc | Brier | \u0394Brier | Log Loss | \u0394LogLoss |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    all_models = sorted(set(today_metrics) | set(yesterday_metrics))
    for model in all_models:
        t = today_metrics.get(model)
        y = yesterday_metrics.get(model)
        if t and y:
            d_acc = t["accuracy"] - y["accuracy"]
            d_brier = t["brier"] - y["brier"]
            d_ll = t["logloss"] - y["logloss"]
            lines.append(
                f"| {model} | {t['accuracy']:.4f} | {d_acc:+.4f} | "
                f"{t['brier']:.4f} | {d_brier:+.4f} | "
                f"{t['logloss']:.4f} | {d_ll:+.4f} |"
            )
        elif t:
            lines.append(
                f"| {model} | {t['accuracy']:.4f} | NEW | "
                f"{t['brier']:.4f} | NEW | "
                f"{t['logloss']:.4f} | NEW |"
            )
        elif y:
            lines.append(f"| {model} | — | REMOVED | — | REMOVED | — | REMOVED |")

    lines.append("")
    report = "\n".join(lines)
    delta_path = ARTIFACTS_DIR / f"benchmark_delta_{today_str}.md"
    delta_path.write_text(report, encoding="utf-8")
    return delta_path


def find_yesterday_artifact(today_str: str) -> Path | None:
    """Find the most recent benchmark artifact before today."""
    today_date = date.fromisoformat(today_str)
    # Look back up to 7 days for the previous artifact
    for i in range(1, 8):
        candidate = today_date - timedelta(days=i)
        path = ARTIFACTS_DIR / f"model_benchmark_{candidate.isoformat()}.md"
        if path.exists():
            return path
    return None


def cleanup_artifacts(retain: int) -> list[Path]:
    """Keep only the last N daily benchmark + delta artifacts. Returns deleted paths."""
    deleted: list[Path] = []
    for prefix in ("model_benchmark_", "benchmark_delta_"):
        pattern = f"{prefix}????-??-??.md"
        files = sorted(ARTIFACTS_DIR.glob(pattern))
        if len(files) > retain:
            to_remove = files[: len(files) - retain]
            for f in to_remove:
                f.unlink()
                deleted.append(f)
    return deleted


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    today_str = args.date or date.today().isoformat()

    print(f"=== Daily Benchmark — {today_str} ===")

    # 1. Generate today's benchmark via evaluate_meta_stack.py
    print(f"Running evaluate_meta_stack.py for {SEASON_START} to {today_str} ...")
    out_path = run_benchmark(today_str)
    print(f"Benchmark saved: {out_path}")

    # 2. Generate delta against yesterday (if available)
    yesterday_path = find_yesterday_artifact(today_str)
    if yesterday_path:
        delta_path = generate_delta(out_path, yesterday_path, today_str)
        print(f"Delta saved: {delta_path}")
    else:
        print("No previous artifact found — skipping delta.")

    # 3. Retain only last N artifacts
    deleted = cleanup_artifacts(args.retain)
    if deleted:
        print(f"Cleaned up {len(deleted)} old artifact(s).")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
