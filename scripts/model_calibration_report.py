#!/usr/bin/env python3
"""Generate a lightweight calibration report for model win probabilities.

Shows raw vs calibrated Brier scores and reliability bins for each model.
"""

import math
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.database import get_connection

OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "model_calibration_report.md"
N_BINS = 10


def _clamp_prob(p):
    return min(max(float(p), 1e-6), 1.0 - 1e-6)


def _brier_and_logloss(rows):
    n = len(rows)
    if n == 0:
        return None, None
    brier = sum((p - y) ** 2 for p, y in rows) / n
    logloss = -sum(y * math.log(_clamp_prob(p)) + (1 - y) * math.log(_clamp_prob(1 - p)) for p, y in rows) / n
    return brier, logloss


def _reliability_bins(rows, n_bins=N_BINS):
    bins = []
    if not rows:
        return bins

    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        bucket = [
            (p, y) for p, y in rows
            if (lo <= p < hi) or (i == n_bins - 1 and lo <= p <= hi)
        ]
        if not bucket:
            continue
        avg_p = sum(p for p, _ in bucket) / len(bucket)
        win_rate = sum(y for _, y in bucket) / len(bucket)
        bins.append({
            "bin": f"[{lo:.1f}, {hi:.1f}{']' if i == n_bins - 1 else ')'}",
            "n": len(bucket),
            "avg_pred": avg_p,
            "actual_home_win_rate": win_rate,
            "gap": win_rate - avg_p,
        })
    return bins


def _fetch_rows():
    conn = get_connection()
    try:
        cur = conn.cursor()
        rows = cur.execute(
            """
            SELECT
                mp.model_name,
                mp.predicted_home_prob,
                mp.raw_home_prob,
                CASE
                    WHEN g.winner_id = g.home_team_id THEN 1
                    WHEN g.winner_id = g.away_team_id THEN 0
                    ELSE NULL
                END AS home_won
            FROM model_predictions mp
            JOIN games g ON g.id = mp.game_id
            WHERE mp.predicted_home_prob IS NOT NULL
              AND g.status = 'final'
              AND g.home_score IS NOT NULL
              AND g.away_score IS NOT NULL
              AND g.winner_id IS NOT NULL
            ORDER BY mp.model_name, mp.predicted_at DESC
            """
        ).fetchall()
        grouped = {}
        for row in rows:
            if row["home_won"] is None:
                continue
            raw = float(row["raw_home_prob"]) if row["raw_home_prob"] is not None else None
            grouped.setdefault(row["model_name"], []).append(
                (float(row["predicted_home_prob"]), int(row["home_won"]), raw)
            )
        return grouped
    finally:
        conn.close()


def generate_report():
    grouped = _fetch_rows()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Model Calibration Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")

    if not grouped:
        lines.append("No evaluated `model_predictions` rows found.")
        OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return OUTPUT_PATH

    lines.append("## Summary")
    lines.append("")
    lines.append("| Model | N | Brier (raw) | Brier (cal) | Improvement | LogLoss | Avg Pred | Actual Home Win Rate | |Gap| |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    summaries = []
    for model_name, rows in sorted(grouped.items()):
        cal_pairs = [(p, y) for p, y, _ in rows]
        brier, logloss = _brier_and_logloss(cal_pairs)
        n = len(rows)
        avg_pred = sum(p for p, _, _ in rows) / n
        actual_rate = sum(y for _, y, _ in rows) / n
        gap = abs(actual_rate - avg_pred)

        # Compute raw Brier where raw probs are available
        raw_pairs = [(r, y) for _, y, r in rows if r is not None]
        if raw_pairs:
            brier_raw, _ = _brier_and_logloss(raw_pairs)
        else:
            brier_raw = None

        summaries.append((model_name, n, brier_raw, brier, logloss, avg_pred, actual_rate, gap))

    for model_name, n, brier_raw, brier, logloss, avg_pred, actual_rate, gap in sorted(summaries, key=lambda x: (-x[1], x[0])):
        raw_str = f"{brier_raw:.4f}" if brier_raw is not None else "—"
        if brier_raw is not None and brier is not None:
            improvement = brier_raw - brier
            imp_str = f"{improvement:+.4f}"
        else:
            imp_str = "—"
        lines.append(
            f"| {model_name} | {n} | {raw_str} | {brier:.4f} | {imp_str} | {logloss:.4f} | {avg_pred:.4f} | {actual_rate:.4f} | {gap:.4f} |"
        )

    for model_name, rows in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        cal_pairs = [(p, y) for p, y, _ in rows]
        lines.append("")
        lines.append(f"## Reliability Bins: `{model_name}`")
        lines.append("")
        lines.append(f"Samples: {len(rows)}")
        lines.append("")
        lines.append("| Bin | N | Avg Pred | Actual Home Win Rate | Gap |")
        lines.append("|---|---:|---:|---:|---:|")
        for b in _reliability_bins(cal_pairs):
            lines.append(
                f"| {b['bin']} | {b['n']} | {b['avg_pred']:.4f} | {b['actual_home_win_rate']:.4f} | {b['gap']:+.4f} |"
            )

    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return OUTPUT_PATH


def main():
    path = generate_report()
    print(f"Wrote calibration report: {path}")


if __name__ == "__main__":
    main()
