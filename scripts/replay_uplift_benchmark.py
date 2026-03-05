#!/usr/bin/env python3
"""Trusted replay uplift benchmark.

Methodology:
1) Build leak-safe final-game cohort in a date range.
2) Baseline = stored pregame `meta_ensemble` probabilities.
3) Candidate (trusted) = recompute ONLY meta score using CURRENT meta model
   over STORED base model probabilities.
4) Compare baseline vs candidate on identical game IDs.

Optional `--exploratory-shadow-submodels` runs non-trusted directional checks by
recomputing live submodel probabilities (lightgbm/xgboost/upset).
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.meta_ensemble import MODEL_NAMES, MetaEnsemble
from scripts.database import get_connection
from scripts.evaluate_meta_stack import apply_leak_safe_filter, compute_metrics

EPS = 1e-12
EXPLORATORY_MODELS = ["lightgbm", "xgboost", "upset"]


@dataclass
class ReplayCohort:
    raw_rows: List[Dict[str, object]]
    filtered_rows: List[Dict[str, object]]
    filter_stats: object



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trusted replay uplift benchmark")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--out", default=None, help="Output markdown path")
    parser.add_argument(
        "--exploratory-shadow-submodels",
        action="store_true",
        help="Recompute live lgb/xgb/upset directional shadow results (NON-TRUSTED)",
    )
    return parser.parse_args()



def clamp_prob(p: float) -> float:
    return max(EPS, min(1.0 - EPS, float(p)))



def fetch_prediction_rows(start_date: str, end_date: str) -> List[Dict[str, object]]:
    conn = get_connection()
    try:
        cur = conn.cursor()
        rows = cur.execute(
            """
            SELECT
                mp.game_id,
                mp.model_name,
                mp.predicted_home_prob,
                mp.predicted_at,
                mp.prediction_source,
                g.date AS game_date,
                g.time AS game_time,
                g.home_team_id,
                g.away_team_id,
                COALESCE(g.is_neutral_site, 0) AS is_neutral_site,
                CASE
                    WHEN g.winner_id = g.home_team_id THEN 1
                    WHEN g.winner_id = g.away_team_id THEN 0
                    ELSE NULL
                END AS home_won
            FROM model_predictions mp
            JOIN games g ON g.id = mp.game_id
            WHERE g.status = 'final'
              AND g.home_score IS NOT NULL
              AND g.away_score IS NOT NULL
              AND g.winner_id IS NOT NULL
              AND g.date >= ?
              AND g.date <= ?
              AND mp.predicted_home_prob IS NOT NULL
              AND mp.model_name IN ({placeholders})
            ORDER BY g.date, mp.game_id, mp.model_name
            """.format(placeholders=",".join(["?"] * (len(MODEL_NAMES) + 1))),
            [start_date, end_date, "meta_ensemble", *MODEL_NAMES],
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()



def build_replay_cohort(start_date: str, end_date: str) -> ReplayCohort:
    raw_rows = fetch_prediction_rows(start_date, end_date)
    filtered_rows, stats = apply_leak_safe_filter(raw_rows)
    return ReplayCohort(raw_rows=raw_rows, filtered_rows=filtered_rows, filter_stats=stats)



def build_meta_score_fn(meta: Optional[MetaEnsemble] = None) -> Callable[[Dict[str, float]], float]:
    model = meta or MetaEnsemble()

    def _fallback(_probs: Dict[str, float]) -> float:
        return 0.5

    if not model._load():
        return _fallback

    predictor = model.xgb_model if model.xgb_model is not None else model.lr_model
    if predictor is None:
        return _fallback

    def _score(probs_by_model: Dict[str, float]) -> float:
        probs = [clamp_prob(probs_by_model.get(name, 0.5)) for name in MODEL_NAMES]
        features = np.array([model._build_meta_features(probs)], dtype=float)
        score = float(predictor.predict_proba(features)[:, 1][0])
        return clamp_prob(score)

    return _score



def _index_first_by_game(rows: Sequence[Dict[str, object]], model_name: str) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for row in rows:
        if row["model_name"] != model_name:
            continue
        gid = str(row["game_id"])
        if gid not in out:
            out[gid] = row
    return out



def _index_probs_by_game(rows: Sequence[Dict[str, object]], model_names: Sequence[str]) -> Dict[str, Dict[str, float]]:
    allowed = set(model_names)
    out: Dict[str, Dict[str, float]] = {}
    for row in rows:
        model_name = str(row["model_name"])
        if model_name not in allowed:
            continue
        gid = str(row["game_id"])
        out.setdefault(gid, {})[model_name] = float(row["predicted_home_prob"])
    return out



def build_trusted_comparison_rows(
    filtered_rows: Sequence[Dict[str, object]],
    score_fn: Callable[[Dict[str, float]], float],
) -> List[Dict[str, object]]:
    baseline_meta = _index_first_by_game(filtered_rows, "meta_ensemble")
    base_probs = _index_probs_by_game(filtered_rows, MODEL_NAMES)

    out: List[Dict[str, object]] = []
    for gid, meta_row in sorted(baseline_meta.items()):
        probs_by_model = base_probs.get(gid, {})
        candidate = score_fn(probs_by_model)
        out.append(
            {
                "game_id": gid,
                "home_won": int(meta_row["home_won"]),
                "baseline_meta": float(meta_row["predicted_home_prob"]),
                "candidate_meta": float(candidate),
                "base_models_present": len(probs_by_model),
                "game_date": meta_row["game_date"],
                "home_team_id": meta_row.get("home_team_id"),
                "away_team_id": meta_row.get("away_team_id"),
                "is_neutral_site": int(meta_row.get("is_neutral_site") or 0),
            }
        )
    return out



def compute_probability_metrics(pairs: Sequence[Dict[str, object]], prob_key: str) -> Dict[str, Optional[float]]:
    rows = [
        {
            "predicted_home_prob": float(p[prob_key]),
            "home_won": int(p["home_won"]),
        }
        for p in pairs
    ]
    return compute_metrics(rows)



def compute_flip_analysis(
    pairs: Sequence[Dict[str, object]],
    baseline_key: str = "baseline_meta",
    candidate_key: str = "candidate_meta",
) -> Dict[str, int]:
    flips = 0
    net_correct_change = 0

    for row in pairs:
        b_prob = float(row[baseline_key])
        c_prob = float(row[candidate_key])
        y = int(row["home_won"])

        b_pick = 1 if b_prob >= 0.5 else 0
        c_pick = 1 if c_prob >= 0.5 else 0
        if b_pick == c_pick:
            continue

        flips += 1
        b_correct = 1 if b_pick == y else 0
        c_correct = 1 if c_pick == y else 0
        net_correct_change += c_correct - b_correct

    return {
        "side_flips": flips,
        "net_correct_change_from_flips": net_correct_change,
    }



def _fmt(v: Optional[float], places: int = 4) -> str:
    if v is None:
        return "—"
    return f"{v:.{places}f}"



def _fmt_delta(v: float, places: int = 4) -> str:
    return f"{v:+.{places}f}"



def _build_exploratory_shadow(
    pairs: Sequence[Dict[str, object]],
    filtered_rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    """NON-TRUSTED exploratory section.

    Recomputes selected submodels live against current DB state. This is
    intentionally exploratory and not leak-safe by construction.
    """
    try:
        from models.predictor_db import Predictor
    except Exception:
        return []

    baseline_probs = {
        model_name: _index_first_by_game(filtered_rows, model_name)
        for model_name in EXPLORATORY_MODELS
    }

    results: List[Dict[str, object]] = []
    for model_name in EXPLORATORY_MODELS:
        try:
            predictor = Predictor(model=model_name)
        except Exception:
            continue

        compared = 0
        sum_delta = 0.0
        side_flips = 0
        net_correct = 0
        base_correct = 0
        live_correct = 0

        for pair in pairs:
            gid = str(pair["game_id"])
            if gid not in baseline_probs[model_name]:
                continue

            row = baseline_probs[model_name][gid]
            baseline_prob = float(row["predicted_home_prob"])
            y = int(pair["home_won"])

            try:
                pred = predictor.predict_game(
                    str(pair["home_team_id"]),
                    str(pair["away_team_id"]),
                    neutral_site=bool(pair["is_neutral_site"]),
                )
                live_prob = clamp_prob(float(pred["home_win_probability"]))
            except Exception:
                continue

            compared += 1
            sum_delta += live_prob - baseline_prob

            b_pick = 1 if baseline_prob >= 0.5 else 0
            l_pick = 1 if live_prob >= 0.5 else 0
            if b_pick != l_pick:
                side_flips += 1
            b_correct = 1 if b_pick == y else 0
            l_correct = 1 if l_pick == y else 0
            net_correct += l_correct - b_correct
            base_correct += b_correct
            live_correct += l_correct

        if compared == 0:
            continue

        results.append(
            {
                "model": model_name,
                "n": compared,
                "mean_live_minus_stored_prob": sum_delta / compared,
                "stored_acc": base_correct / compared,
                "live_acc": live_correct / compared,
                "acc_delta": (live_correct - base_correct) / compared,
                "side_flips": side_flips,
                "net_correct_change_from_flips": net_correct,
            }
        )

    return results



def build_report_markdown(
    start_date: str,
    end_date: str,
    cohort: ReplayCohort,
    pairs: Sequence[Dict[str, object]],
    baseline_metrics: Dict[str, Optional[float]],
    candidate_metrics: Dict[str, Optional[float]],
    flips: Dict[str, int],
    exploratory_rows: Optional[Sequence[Dict[str, object]]] = None,
) -> str:
    generated = datetime.now().isoformat(timespec="seconds")

    lines: List[str] = []
    lines.append("# P1.3 Trusted Replay Uplift Benchmark")
    lines.append("")
    lines.append(f"- Generated: `{generated}`")
    lines.append(f"- Date range: `{start_date}` .. `{end_date}`")
    lines.append("")

    lines.append("## Cohort definition")
    lines.append("")
    lines.append("Leak-safe cohort filters (P0 semantics):")
    lines.append("- final games only with known winner and scores")
    lines.append("- exclude `prediction_source='backfill'`")
    lines.append("- enforce pregame timestamp cutoff (`predicted_at <= game_start - 5m`, fallback end-of-date)")
    lines.append("- baseline and candidate evaluated on the same game IDs (stored `meta_ensemble` rows)")
    lines.append("")
    lines.append("| Cohort stat | Value |")
    lines.append("|---|---:|")
    lines.append(f"| raw rows in date range | {len(cohort.raw_rows)} |")
    lines.append(f"| leak-safe kept rows | {len(cohort.filtered_rows)} |")
    lines.append(f"| excluded backfill | {cohort.filter_stats.excluded_backfill} |")
    lines.append(f"| excluded disallowed source | {cohort.filter_stats.excluded_source} |")
    lines.append(f"| excluded late/non-parseable timestamp | {cohort.filter_stats.excluded_late} |")
    lines.append(f"| trusted benchmark games (`n`) | {len(pairs)} |")

    lines.append("")
    lines.append("## Trusted uplift")
    lines.append("")
    lines.append("Baseline = stored pregame `meta_ensemble`; candidate = current meta model replayed on stored base probabilities only.")
    lines.append("")
    lines.append("| Metric | Baseline stored meta | Candidate replayed meta | Delta (candidate-baseline) |")
    lines.append("|---|---:|---:|---:|")

    for metric in ["n", "accuracy", "brier", "log_loss", "ece"]:
        b = baseline_metrics.get(metric)
        c = candidate_metrics.get(metric)
        if metric == "n":
            b_num = int(b or 0)
            c_num = int(c or 0)
            lines.append(f"| {metric} | {b_num} | {c_num} | {c_num - b_num:+d} |")
            continue
        delta = (c - b) if (b is not None and c is not None) else None
        lines.append(
            f"| {metric} | {_fmt(b)} | {_fmt(c)} | {(_fmt_delta(delta) if delta is not None else '—')} |"
        )

    lines.append("")
    lines.append("## Flip analysis")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| side flips count | {flips['side_flips']} |")
    lines.append(f"| net correct change from flips | {flips['net_correct_change_from_flips']} |")

    if exploratory_rows:
        lines.append("")
        lines.append("## Optional exploratory section (NON-TRUSTED)")
        lines.append("")
        lines.append("Directional shadow recomputation of live submodels (`lightgbm`, `xgboost`, `upset`) against their stored probabilities.")
        lines.append("")
        lines.append("| Model | n | Mean(live-stored prob) | Stored acc | Live acc | Acc delta | Side flips | Net correct change from flips |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for row in exploratory_rows:
            lines.append(
                f"| {row['model']} | {row['n']} | {row['mean_live_minus_stored_prob']:+.4f} | "
                f"{row['stored_acc']:.4f} | {row['live_acc']:.4f} | {row['acc_delta']:+.4f} | "
                f"{row['side_flips']} | {row['net_correct_change_from_flips']} |"
            )

    lines.append("")
    lines.append("## Interpretation / confidence limits")
    lines.append("")
    lines.append("- This is a replay-style observational benchmark, not a randomized experiment.")
    lines.append("- Trusted uplift isolates only meta-layer changes because base probabilities are held fixed from stored pregame rows.")
    lines.append("- Sample size and date-window composition can dominate deltas; confidence is limited for small `n` or low flip counts.")
    lines.append("- Exploratory shadow results (if present) are intentionally non-trusted and directional only.")

    return "\n".join(lines) + "\n"



def default_output_path(end_date: str) -> Path:
    return ROOT / "artifacts" / f"replay_uplift_{end_date}.md"



def run_benchmark(
    start_date: str,
    end_date: str,
    out_path: Optional[Path] = None,
    exploratory_shadow_submodels: bool = False,
) -> Path:
    cohort = build_replay_cohort(start_date, end_date)
    score_fn = build_meta_score_fn()
    pairs = build_trusted_comparison_rows(cohort.filtered_rows, score_fn)

    baseline_metrics = compute_probability_metrics(pairs, "baseline_meta")
    candidate_metrics = compute_probability_metrics(pairs, "candidate_meta")
    flips = compute_flip_analysis(pairs)

    exploratory_rows = None
    if exploratory_shadow_submodels:
        exploratory_rows = _build_exploratory_shadow(pairs, cohort.filtered_rows)

    destination = out_path or default_output_path(end_date)
    destination.parent.mkdir(parents=True, exist_ok=True)

    report = build_report_markdown(
        start_date=start_date,
        end_date=end_date,
        cohort=cohort,
        pairs=pairs,
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
        flips=flips,
        exploratory_rows=exploratory_rows,
    )
    destination.write_text(report, encoding="utf-8")
    return destination



def main() -> int:
    args = parse_args()
    out_path = Path(args.out) if args.out else None
    path = run_benchmark(
        start_date=args.start_date,
        end_date=args.end_date,
        out_path=out_path,
        exploratory_shadow_submodels=args.exploratory_shadow_submodels,
    )
    print(f"Wrote replay uplift benchmark: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
