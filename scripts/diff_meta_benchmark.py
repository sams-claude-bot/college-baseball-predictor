#!/usr/bin/env python3
"""Generate a markdown diff between two benchmark reports.

Usage:
  python3 scripts/diff_meta_benchmark.py \
    --baseline artifacts/model_benchmark_2026-03-05.md \
    --current artifacts/model_benchmark_post_p12_2026-03-05.md \
    --out artifacts/model_benchmark_diff_post_p12_2026-03-05.md
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.evaluate_meta_stack import (
    ACTIVE_MODELS,
    apply_leak_safe_filter,
    build_intersection,
    fetch_rows,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diff two meta benchmark markdown reports")
    p.add_argument("--baseline", required=True)
    p.add_argument("--current", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--start-date", default="2026-01-01")
    p.add_argument("--end-date", default="2026-12-31")
    return p.parse_args()


def _read(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _extract_table(lines: List[str], section_header: str) -> List[List[str]]:
    start = None
    for i, line in enumerate(lines):
        if line.strip() == section_header:
            start = i
            break
    if start is None:
        return []

    table_start = None
    for i in range(start + 1, len(lines)):
        if lines[i].startswith("| Model") or lines[i].startswith("| Model A"):
            table_start = i
            break
    if table_start is None:
        return []

    out: List[List[str]] = []
    for i in range(table_start + 2, len(lines)):
        line = lines[i].strip()
        if not line.startswith("|"):
            break
        cols = [c.strip() for c in line.strip("|").split("|")]
        out.append(cols)
    return out


def _extract_corr_table(lines: List[str]) -> List[List[str]]:
    for i, line in enumerate(lines):
        if line.strip().startswith("Top 10 highest absolute pairwise correlations"):
            for j in range(i + 1, len(lines)):
                if lines[j].startswith("| Model A"):
                    out: List[List[str]] = []
                    for k in range(j + 2, len(lines)):
                        row = lines[k].strip()
                        if not row.startswith("|"):
                            return out
                        out.append([c.strip() for c in row.strip("|").split("|")])
                    return out
    return []


def _table_to_metrics(rows: List[List[str]]) -> Dict[str, Dict[str, float]]:
    d = {}
    for r in rows:
        if len(r) < 6:
            continue
        try:
            d[r[0]] = {
                "n": float(r[1]),
                "accuracy": float(r[2]),
                "brier": float(r[3]),
                "log_loss": float(r[4]),
                "ece": float(r[5]),
            }
        except ValueError:
            continue
    return d


def _delta(curr: float, base: float) -> float:
    return curr - base


def _rank_top5(rows: List[List[str]]) -> List[Tuple[str, float]]:
    vals: List[Tuple[str, float]] = []
    for r in rows:
        if len(r) < 3:
            continue
        try:
            vals.append((r[0], float(r[2])))
        except ValueError:
            continue
    vals.sort(key=lambda x: x[1], reverse=True)
    return vals[:5]


def _fmt_signed(v: float, places: int = 4) -> str:
    return f"{v:+.{places}f}"


def _upset_diagnostics(start_date: str, end_date: str) -> Dict[str, float]:
    rows = fetch_rows(start_date, end_date)
    leak_safe_rows, _ = apply_leak_safe_filter(rows)
    strict = build_intersection(leak_safe_rows, ACTIVE_MODELS)

    upset_rows_leak = [r for r in leak_safe_rows if r["model_name"] == "upset"]
    upset_rows_strict = strict.get("upset", [])

    def compute(rows_local):
        n = len(rows_local)
        if n == 0:
            return {
                "n": 0,
                "base_upset_rate": 0.0,
                "majority_acc": 0.0,
                "precision": 0.0,
                "recall": 0.0,
            }

        # proxy: "upset" := away/home-underdog side wins => home team loses.
        y_upset = [1 - int(r["home_won"]) for r in rows_local]
        pred_upset = [1 if float(r["predicted_home_prob"]) < 0.5 else 0 for r in rows_local]

        pos = sum(y_upset)
        neg = n - pos
        base_upset_rate = pos / n
        majority_acc = max(base_upset_rate, 1.0 - base_upset_rate)

        tp = sum(1 for y, p in zip(y_upset, pred_upset) if y == 1 and p == 1)
        fp = sum(1 for y, p in zip(y_upset, pred_upset) if y == 0 and p == 1)
        fn = sum(1 for y, p in zip(y_upset, pred_upset) if y == 1 and p == 0)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        return {
            "n": n,
            "base_upset_rate": base_upset_rate,
            "majority_acc": majority_acc,
            "precision": precision,
            "recall": recall,
        }

    return {
        "leak": compute(upset_rows_leak),
        "strict": compute(upset_rows_strict),
    }


def main() -> int:
    args = parse_args()
    baseline = Path(args.baseline)
    current = Path(args.current)
    out = Path(args.out)

    b_lines = _read(baseline)
    c_lines = _read(current)

    b_leak = _table_to_metrics(_extract_table(b_lines, "## Leak-safe per-model leaderboard"))
    c_leak = _table_to_metrics(_extract_table(c_lines, "## Leak-safe per-model leaderboard"))

    b_strict = _table_to_metrics(_extract_table(b_lines, "## Strict cohort leaderboard"))
    c_strict = _table_to_metrics(_extract_table(c_lines, "## Strict cohort leaderboard"))

    b_top5 = _rank_top5(_extract_table(b_lines, "## Leak-safe per-model leaderboard"))
    c_top5 = _rank_top5(_extract_table(c_lines, "## Leak-safe per-model leaderboard"))

    b_corr = _extract_corr_table(b_lines)
    c_corr = _extract_corr_table(c_lines)

    b_meta_corr = {
        tuple(sorted((r[0], r[1]))): float(r[2])
        for r in b_corr
        if len(r) >= 3 and (r[0] == "meta_ensemble" or r[1] == "meta_ensemble")
    }
    c_meta_corr = {
        tuple(sorted((r[0], r[1]))): float(r[2])
        for r in c_corr
        if len(r) >= 3 and (r[0] == "meta_ensemble" or r[1] == "meta_ensemble")
    }

    upset_diag = _upset_diagnostics(args.start_date, args.end_date)

    lines: List[str] = []
    lines.append("# P1.2 Benchmark Diff (Baseline vs Post-P1.2)")
    lines.append("")
    lines.append(f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append(f"- Baseline: `{baseline}`")
    lines.append(f"- Current: `{current}`")
    lines.append("")

    lines.append("## Meta-ensemble delta (leak-safe cohort)")
    lines.append("")
    lines.append("| Metric | Baseline | Current | Delta (current-baseline) |")
    lines.append("|---|---:|---:|---:|")
    for m in ["accuracy", "brier", "log_loss", "ece"]:
        b = b_leak["meta_ensemble"][m]
        c = c_leak["meta_ensemble"][m]
        lines.append(f"| {m} | {b:.4f} | {c:.4f} | {_fmt_signed(_delta(c, b))} |")

    lines.append("")
    lines.append("## Meta-ensemble delta (strict cohort)")
    lines.append("")
    lines.append("| Metric | Baseline | Current | Delta (current-baseline) |")
    lines.append("|---|---:|---:|---:|")
    for m in ["accuracy", "brier", "log_loss", "ece"]:
        b = b_strict["meta_ensemble"][m]
        c = c_strict["meta_ensemble"][m]
        lines.append(f"| {m} | {b:.4f} | {c:.4f} | {_fmt_signed(_delta(c, b))} |")

    lines.append("")
    lines.append("## Top-5 ranking changes (leak-safe, by win accuracy)")
    lines.append("")
    lines.append("| Rank | Baseline | Current | Changed? |")
    lines.append("|---:|---|---|---|")
    for i in range(5):
        b = b_top5[i] if i < len(b_top5) else ("-", 0.0)
        c = c_top5[i] if i < len(c_top5) else ("-", 0.0)
        changed = "yes" if b[0] != c[0] else "no"
        lines.append(f"| {i+1} | {b[0]} ({b[1]:.4f}) | {c[0]} ({c[1]:.4f}) | {changed} |")

    lines.append("")
    lines.append("## Correlation shifts relevant to stack quality")
    lines.append("")
    lines.append("Meta-to-base pair shifts from strict-cohort top-correlation tables:")
    lines.append("")
    lines.append("| Pair | Baseline corr | Current corr | Delta |")
    lines.append("|---|---:|---:|---:|")
    for pair in sorted(set(b_meta_corr) | set(c_meta_corr)):
        b = b_meta_corr.get(pair)
        c = c_meta_corr.get(pair)
        if b is None:
            lines.append(f"| {' vs '.join(pair)} | - | {c:.4f} | - |")
        elif c is None:
            lines.append(f"| {' vs '.join(pair)} | {b:.4f} | - | - |")
        else:
            lines.append(f"| {' vs '.join(pair)} | {b:.4f} | {c:.4f} | {_fmt_signed(c-b)} |")

    lines.append("")
    lines.append("## Upset-model diagnostics (quick extension)")
    lines.append("")
    lines.append("Proxy definition used for this diagnostic: upset event = home team loss (away win).")
    lines.append("Predicted upset at threshold 0.5 when `upset predicted_home_prob < 0.5`.")
    lines.append("")
    lines.append("| Cohort | n | Base upset rate | Majority-class baseline acc | Upset precision@0.5 | Upset recall@0.5 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, k in [("Leak-safe upset rows", "leak"), ("Strict upset rows", "strict")]:
        d = upset_diag[k]
        lines.append(
            f"| {name} | {d['n']} | {d['base_upset_rate']:.4f} | {d['majority_acc']:.4f} | {d['precision']:.4f} | {d['recall']:.4f} |"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote diff report to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
