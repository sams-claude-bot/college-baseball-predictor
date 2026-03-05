#!/usr/bin/env python3
"""Canonical benchmark report for the meta stack on a leak-safe cohort."""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.database import get_connection

ACTIVE_MODELS: List[str] = [
    "meta_ensemble",
    "elo",
    "pythagorean",
    "lightgbm",
    "poisson",
    "xgboost",
    "pitching",
    "pear",
    "quality",
    "neural",
    "venue",
    "rest_travel",
    "upset",
]

META_COMPARE_MODELS: List[str] = ["upset", "pitching", "pear", "venue", "rest_travel"]
N_BINS = 10
EPS = 1e-12


@dataclass
class CohortFilterStats:
    total_rows: int = 0
    kept_rows: int = 0
    excluded_backfill: int = 0
    excluded_source: int = 0
    excluded_late: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate meta stack using leak-safe cohorts")
    parser.add_argument("--start-date", default="2026-01-01", help="YYYY-MM-DD")
    parser.add_argument("--end-date", default="2026-12-31", help="YYYY-MM-DD")
    parser.add_argument("--out", default=None, help="Output markdown path")
    return parser.parse_args()


def _parse_dt(date_str: str, time_str: Optional[str]) -> datetime:
    if time_str:
        candidate = f"{date_str} {time_str.strip()}"
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                return datetime.strptime(candidate, fmt)
            except ValueError:
                pass
    return datetime.strptime(f"{date_str} 23:59:59", "%Y-%m-%d %H:%M:%S")


def pregame_cutoff(date_str: str, time_str: Optional[str]) -> datetime:
    return _parse_dt(date_str, time_str) - timedelta(minutes=5)


def is_prediction_source_allowed(prediction_source: Optional[str]) -> bool:
    if prediction_source is None:
        return True
    src = prediction_source.strip().lower()
    if src == "":
        return True
    return src in {"live", "refresh"}


def passes_leak_safe_row(row: Dict[str, object]) -> Tuple[bool, Optional[str]]:
    src_raw = row.get("prediction_source")
    src = "" if src_raw is None else str(src_raw).strip().lower()
    if src == "backfill":
        return False, "backfill"
    if not is_prediction_source_allowed(None if src_raw is None else str(src_raw)):
        return False, "source"

    predicted_at_raw = row.get("predicted_at")
    if not predicted_at_raw:
        return False, "late"

    try:
        predicted_at = datetime.fromisoformat(str(predicted_at_raw).replace("Z", ""))
    except ValueError:
        return False, "late"

    cutoff = pregame_cutoff(str(row["game_date"]), row.get("game_time"))
    if predicted_at > cutoff:
        return False, "late"
    return True, None


def apply_leak_safe_filter(rows: Sequence[Dict[str, object]]) -> Tuple[List[Dict[str, object]], CohortFilterStats]:
    stats = CohortFilterStats(total_rows=len(rows))
    kept: List[Dict[str, object]] = []

    for row in rows:
        ok, reason = passes_leak_safe_row(row)
        if ok:
            kept.append(row)
            stats.kept_rows += 1
            continue
        if reason == "backfill":
            stats.excluded_backfill += 1
        elif reason == "source":
            stats.excluded_source += 1
        elif reason == "late":
            stats.excluded_late += 1

    return kept, stats


def fetch_rows(start_date: str, end_date: str) -> List[Dict[str, object]]:
    conn = get_connection()
    try:
        cur = conn.cursor()
        out = cur.execute(
            """
            SELECT
                mp.game_id,
                mp.model_name,
                mp.predicted_home_prob,
                mp.predicted_at,
                mp.prediction_source,
                g.date AS game_date,
                g.time AS game_time,
                CASE
                    WHEN g.winner_id = g.home_team_id THEN 1
                    WHEN g.winner_id = g.away_team_id THEN 0
                    ELSE NULL
                END AS home_won
            FROM model_predictions mp
            JOIN games g ON g.id = mp.game_id
            WHERE g.status = 'final'
              AND g.winner_id IS NOT NULL
              AND g.home_score IS NOT NULL
              AND g.away_score IS NOT NULL
              AND g.date >= ?
              AND g.date <= ?
              AND mp.predicted_home_prob IS NOT NULL
            ORDER BY g.date, mp.game_id, mp.model_name
            """,
            (start_date, end_date),
        ).fetchall()
        return [dict(r) for r in out]
    finally:
        conn.close()


def clamp_prob(p: float) -> float:
    return max(EPS, min(1.0 - EPS, float(p)))


def log_loss(y: int, p: float) -> float:
    p = clamp_prob(p)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def reliability_bins(rows: Sequence[Dict[str, object]], n_bins: int = N_BINS) -> List[Dict[str, float]]:
    bins: List[Dict[str, float]] = []
    if not rows:
        return bins

    for idx in range(n_bins):
        lo = idx / n_bins
        hi = (idx + 1) / n_bins
        bucket = [
            r for r in rows
            if (lo <= float(r["predicted_home_prob"]) < hi)
            or (idx == n_bins - 1 and lo <= float(r["predicted_home_prob"]) <= hi)
        ]
        if not bucket:
            continue
        avg_pred = sum(float(r["predicted_home_prob"]) for r in bucket) / len(bucket)
        actual = sum(int(r["home_won"]) for r in bucket) / len(bucket)
        bins.append({
            "bin_lo": lo,
            "bin_hi": hi,
            "n": len(bucket),
            "avg_pred": avg_pred,
            "actual": actual,
            "gap": actual - avg_pred,
        })
    return bins


def ece(rows: Sequence[Dict[str, object]], n_bins: int = N_BINS) -> Optional[float]:
    if not rows:
        return None
    b = reliability_bins(rows, n_bins=n_bins)
    n = len(rows)
    return sum((x["n"] / n) * abs(x["actual"] - x["avg_pred"]) for x in b)


def compute_metrics(rows: Sequence[Dict[str, object]]) -> Dict[str, Optional[float]]:
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "accuracy": None,
            "brier": None,
            "log_loss": None,
            "ece": None,
        }

    correct = 0
    brier_sum = 0.0
    ll_sum = 0.0
    for r in rows:
        p = clamp_prob(float(r["predicted_home_prob"]))
        y = int(r["home_won"])
        correct += int((1 if p >= 0.5 else 0) == y)
        brier_sum += (p - y) ** 2
        ll_sum += log_loss(y, p)

    return {
        "n": n,
        "accuracy": correct / n,
        "brier": brier_sum / n,
        "log_loss": ll_sum / n,
        "ece": ece(rows),
    }


def group_by_model(rows: Sequence[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["model_name"])].append(row)
    return grouped


def build_intersection(rows: Sequence[Dict[str, object]], required_models: Sequence[str]) -> Dict[str, List[Dict[str, object]]]:
    by_model_game: Dict[str, Dict[str, Dict[str, object]]] = {m: {} for m in required_models}
    for r in rows:
        model = str(r["model_name"])
        if model in by_model_game:
            by_model_game[model][str(r["game_id"])] = r

    common_games = set(by_model_game[required_models[0]].keys()) if required_models else set()
    for m in required_models[1:]:
        common_games &= set(by_model_game[m].keys())

    out: Dict[str, List[Dict[str, object]]] = {m: [] for m in required_models}
    for gid in sorted(common_games):
        for m in required_models:
            out[m].append(by_model_game[m][gid])
    return out


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    den = den_x * den_y
    if den == 0:
        return None
    return num / den


def correlation_pairs(strict_by_model: Dict[str, List[Dict[str, object]]]) -> List[Tuple[str, str, Optional[float]]]:
    models = list(strict_by_model.keys())
    pairs: List[Tuple[str, str, Optional[float]]] = []
    for i, m1 in enumerate(models):
        for m2 in models[i + 1:]:
            x = [float(r["predicted_home_prob"]) for r in strict_by_model[m1]]
            y = [float(r["predicted_home_prob"]) for r in strict_by_model[m2]]
            pairs.append((m1, m2, _pearson(x, y)))
    return pairs


def _fmt(v: Optional[float], d: int = 4) -> str:
    if v is None:
        return "-"
    return f"{v:.{d}f}"


def build_report(
    start_date: str,
    end_date: str,
    leak_safe_rows: List[Dict[str, object]],
    leak_stats: CohortFilterStats,
    strict_by_model: Dict[str, List[Dict[str, object]]],
    all_rows: List[Dict[str, object]],
) -> str:
    by_model_leak = group_by_model(leak_safe_rows)
    active_present = [m for m in ACTIVE_MODELS if m in by_model_leak]
    strict_game_rows = strict_by_model[ACTIVE_MODELS[0]] if ACTIVE_MODELS[0] in strict_by_model else []

    lines: List[str] = []
    lines.append("# Model Benchmark (Leak-Safe Meta Stack)")
    lines.append("")
    lines.append(f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append(f"- Date window: `{start_date}` to `{end_date}`")
    lines.append("")
    lines.append("## Data/Cohort definition")
    lines.append("")
    lines.append("Final games with known winner. Prediction rows are leak-safe only:")
    lines.append("- exclude `prediction_source='backfill'`")
    lines.append("- include only `prediction_source IN ('live','refresh')` or NULL/empty (legacy-live)")
    lines.append("- require `predicted_at <= (game_datetime - 5 minutes)`; fallback cutoff = `game_date 23:54:59` when game time missing")
    lines.append("")
    lines.append("### Filter counts")
    lines.append("")
    lines.append(f"- Total candidate prediction rows: **{leak_stats.total_rows}**")
    lines.append(f"- Excluded backfill rows: **{leak_stats.excluded_backfill}**")
    lines.append(f"- Excluded disallowed source rows: **{leak_stats.excluded_source}**")
    lines.append(f"- Excluded late/invalid timestamp rows: **{leak_stats.excluded_late}**")
    lines.append(f"- Kept leak-safe rows: **{leak_stats.kept_rows}**")
    lines.append("")

    lines.append("## Leak-safe per-model leaderboard")
    lines.append("")
    lines.append("| Model | n predictions | Win accuracy | Brier | Log loss | ECE |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for m in ACTIVE_MODELS:
        metrics = compute_metrics(by_model_leak.get(m, []))
        lines.append(
            f"| {m} | {metrics['n']} | {_fmt(metrics['accuracy'])} | {_fmt(metrics['brier'])} | {_fmt(metrics['log_loss'])} | {_fmt(metrics['ece'])} |"
        )

    legacy_models = sorted(set(r["model_name"] for r in all_rows) - set(ACTIVE_MODELS))
    if legacy_models:
        lines.append("")
        lines.append("### Legacy models present (leak-safe cohort)")
        lines.append("")
        lines.append("| Model | n predictions | Win accuracy | Brier | Log loss | ECE |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for m in legacy_models:
            metrics = compute_metrics(by_model_leak.get(m, []))
            lines.append(
                f"| {m} | {metrics['n']} | {_fmt(metrics['accuracy'])} | {_fmt(metrics['brier'])} | {_fmt(metrics['log_loss'])} | {_fmt(metrics['ece'])} |"
            )

    lines.append("")
    lines.append("## Strict cohort leaderboard")
    lines.append("")
    if strict_game_rows:
        strict_dates = sorted({str(r["game_date"]) for r in strict_game_rows})
        lines.append(f"- Strict cohort size (games with predictions from all active models + meta): **{len(strict_game_rows)}**")
        lines.append(f"- Strict cohort date range: **{strict_dates[0]} to {strict_dates[-1]}**")
    else:
        lines.append("- Strict cohort size: **0**")
    lines.append("")
    lines.append("| Model | n predictions | Win accuracy | Brier | Log loss | ECE |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for m in ACTIVE_MODELS:
        metrics = compute_metrics(strict_by_model.get(m, []))
        lines.append(
            f"| {m} | {metrics['n']} | {_fmt(metrics['accuracy'])} | {_fmt(metrics['brier'])} | {_fmt(metrics['log_loss'])} | {_fmt(metrics['ece'])} |"
        )

    lines.append("")
    lines.append("## Calibration table")
    lines.append("")
    lines.append("Reliability bins on strict cohort (10 bins):")
    lines.append("")
    lines.append("| Model | Bin range | n | Avg predicted home win | Actual home win | Gap |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for m in ACTIVE_MODELS:
        bins = reliability_bins(strict_by_model.get(m, []))
        if not bins:
            lines.append(f"| {m} | - | 0 | - | - | - |")
            continue
        first = True
        for b in bins:
            label = f"[{b['bin_lo']:.1f}, {b['bin_hi']:.1f}{']' if abs(b['bin_hi'] - 1.0) < 1e-9 else ')'}"
            model_label = m if first else ""
            lines.append(
                f"| {model_label} | {label} | {b['n']} | {b['avg_pred']:.4f} | {b['actual']:.4f} | {b['gap']:+.4f} |"
            )
            first = False

    lines.append("")
    lines.append("## Correlation findings")
    lines.append("")
    corr = correlation_pairs(strict_by_model)
    sorted_corr = sorted([p for p in corr if p[2] is not None], key=lambda x: abs(x[2]), reverse=True)
    lines.append("Top 10 highest absolute pairwise correlations (strict cohort):")
    lines.append("")
    lines.append("| Model A | Model B | Correlation | abs(corr) |")
    lines.append("|---|---|---:|---:|")
    for m1, m2, c in sorted_corr[:10]:
        lines.append(f"| {m1} | {m2} | {c:.4f} | {abs(c):.4f} |")

    lines.append("")
    lines.append("Meta disagreement analysis (strict cohort):")
    lines.append("")
    lines.append("| Meta vs model | Agreement rate | Meta accuracy when agree | Meta accuracy when disagree |")
    lines.append("|---|---:|---:|---:|")
    meta_rows = strict_by_model.get("meta_ensemble", [])
    meta_map = {r["game_id"]: r for r in meta_rows}
    for m in META_COMPARE_MODELS:
        other_rows = strict_by_model.get(m, [])
        total = len(other_rows)
        if total == 0:
            lines.append(f"| {m} | - | - | - |")
            continue
        agree = 0
        meta_correct_agree = 0
        meta_correct_disagree = 0
        disagree = 0
        for r in other_rows:
            mr = meta_map[r["game_id"]]
            m_side = 1 if float(mr["predicted_home_prob"]) >= 0.5 else 0
            o_side = 1 if float(r["predicted_home_prob"]) >= 0.5 else 0
            meta_correct = int(m_side == int(mr["home_won"]))
            if m_side == o_side:
                agree += 1
                meta_correct_agree += meta_correct
            else:
                disagree += 1
                meta_correct_disagree += meta_correct
        agreement_rate = agree / total if total else None
        acc_agree = (meta_correct_agree / agree) if agree else None
        acc_disagree = (meta_correct_disagree / disagree) if disagree else None
        lines.append(f"| {m} | {_fmt(agreement_rate)} | {_fmt(acc_agree)} | {_fmt(acc_disagree)} |")

    lines.append("")
    lines.append("## Risks/interpretation notes")
    lines.append("")
    lines.append("- Strict cohort can be much smaller than leak-safe per-model cohort; this may change rank ordering.")
    lines.append("- High correlation means less independent signal and limits stacking upside.")
    lines.append("- ECE is sample-size sensitive; sparse bins can look noisy on small cohorts.")
    lines.append("- Accuracy alone can hide confidence miscalibration, so Brier/log loss/ECE should be considered together.")

    lines.append("")
    lines.append("## Recommended next experiments")
    lines.append("")
    lines.append("- Re-rank meta base-feature set using strict-cohort incremental gain vs high-correlation redundancy.")
    lines.append("- Compare meta calibration before/after isotonic/Platt post-calibration on strict cohort only.")
    lines.append("- Stress-test upset/pitching/venue contributions on disagreement-only slices.")
    lines.append("- Add time-slice stability (weekly rolling strict-cohort metrics) before any hyperparameter tuning.")
    lines.append("- Audit games dropped from strict cohort to identify model coverage gaps by source/date.")

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    rows = fetch_rows(args.start_date, args.end_date)
    leak_safe_rows, stats = apply_leak_safe_filter(rows)
    strict_by_model = build_intersection(leak_safe_rows, ACTIVE_MODELS)

    today = datetime.now().date().isoformat()
    out_path = Path(args.out) if args.out else ROOT / "artifacts" / f"model_benchmark_{today}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report = build_report(
        start_date=args.start_date,
        end_date=args.end_date,
        leak_safe_rows=leak_safe_rows,
        leak_stats=stats,
        strict_by_model=strict_by_model,
        all_rows=rows,
    )
    out_path.write_text(report, encoding="utf-8")

    print(f"Leak-safe filter counts: total={stats.total_rows}, backfill={stats.excluded_backfill}, source={stats.excluded_source}, late={stats.excluded_late}, kept={stats.kept_rows}")
    print(report)
    print(f"Saved report to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
