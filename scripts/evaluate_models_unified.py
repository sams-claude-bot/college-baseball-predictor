#!/usr/bin/env python3
"""
Unified read-only evaluator for model comparisons on the same finalized game set.
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import database as dbmod


def _install_readonly_sqlite_guard() -> None:
    """Force project DB connections to open read-only to prevent accidental writes."""
    original_connect = sqlite3.connect
    target_db = str(dbmod.DB_PATH.resolve())

    def guarded_connect(database, *args, **kwargs):
        try:
            resolved = str(Path(database).resolve())
        except Exception:
            resolved = str(database)

        if resolved == target_db and not kwargs.get("uri"):
            uri = f"file:{target_db}?mode=ro"
            return original_connect(uri, *args, uri=True, **kwargs)
        return original_connect(database, *args, **kwargs)

    sqlite3.connect = guarded_connect


@dataclass(frozen=True)
class GameRow:
    game_id: str
    date: str
    home_team_id: str
    away_team_id: str
    home_score: int
    away_score: int
    neutral_site: bool


@dataclass
class EvalAccumulator:
    n_games: int = 0
    n_prob: int = 0
    n_totals: int = 0
    correct: int = 0
    brier_sum: float = 0.0
    log_loss_sum: float = 0.0
    totals_mae_sum: float = 0.0

    def as_row(self) -> Dict[str, object]:
        accuracy = (self.correct / self.n_games) if self.n_games else None
        brier = (self.brier_sum / self.n_prob) if self.n_prob else None
        log_loss = (self.log_loss_sum / self.n_prob) if self.n_prob else None
        totals_mae = (self.totals_mae_sum / self.n_totals) if self.n_totals else None
        return {
            "n_games": self.n_games,
            "win_accuracy": accuracy,
            "brier": brier,
            "log_loss": log_loss,
            "totals_mae": totals_mae,
            "n_prob": self.n_prob,
            "n_totals": self.n_totals,
        }


def _fmt_metric(value: Optional[float], decimals: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{decimals}f}"


def _clamp_prob(p: float) -> float:
    return max(1e-15, min(1.0 - 1e-15, float(p)))


def _binary_log_loss(y: int, p: float) -> float:
    p = _clamp_prob(p)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def _load_games(start_date: str, end_date: str) -> List[GameRow]:
    conn = dbmod.get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, date, home_team_id, away_team_id, home_score, away_score,
                   COALESCE(is_neutral_site, 0) AS is_neutral_site
            FROM games
            WHERE status = 'final'
              AND home_score IS NOT NULL
              AND away_score IS NOT NULL
              AND date >= ?
              AND date <= ?
            ORDER BY date, id
            """,
            (start_date, end_date),
        )
        rows = []
        for r in cur.fetchall():
            if int(r["home_score"]) == int(r["away_score"]):
                continue
            rows.append(
                GameRow(
                    game_id=r["id"],
                    date=r["date"],
                    home_team_id=r["home_team_id"],
                    away_team_id=r["away_team_id"],
                    home_score=int(r["home_score"]),
                    away_score=int(r["away_score"]),
                    neutral_site=bool(r["is_neutral_site"]),
                )
            )
        return rows
    finally:
        conn.close()


def _normalize_prediction(raw: Dict[str, object]) -> Dict[str, Optional[float]]:
    home_prob = raw.get("home_win_probability")
    if home_prob is None and "win_prob_a" in raw:
        home_prob = raw.get("win_prob_a")

    projected_total = raw.get("projected_total")
    if projected_total is None and "expected_total" in raw:
        projected_total = raw.get("expected_total")

    try:
        home_prob = None if home_prob is None else float(home_prob)
    except (TypeError, ValueError):
        home_prob = None
    try:
        projected_total = None if projected_total is None else float(projected_total)
    except (TypeError, ValueError):
        projected_total = None

    return {
        "home_win_probability": home_prob,
        "projected_total": projected_total,
    }


def _predict_poisson(game: GameRow):
    from models.poisson_model import predict as poisson_predict

    return poisson_predict(
        game.home_team_id,
        game.away_team_id,
        neutral_site=game.neutral_site,
        team_a_home=True,
        game_id=game.game_id,
    )


def _predict_elo(game: GameRow, _cache={"model": None}):
    if _cache["model"] is None:
        from models.elo_model import EloModel

        _cache["model"] = EloModel()
    return _cache["model"].predict_game(
        game.home_team_id, game.away_team_id, neutral_site=game.neutral_site
    )


def _predict_ensemble(game: GameRow, _cache={"model": None}):
    if _cache["model"] is None:
        from models.ensemble_model import EnsembleModel

        _cache["model"] = EnsembleModel()
    return _cache["model"].predict_game(
        game.home_team_id,
        game.away_team_id,
        neutral_site=game.neutral_site,
        game_id=game.game_id,
    )


MODEL_RUNNERS: Dict[str, Callable[[GameRow], Dict[str, object]]] = {
    "poisson": _predict_poisson,
    "elo": _predict_elo,
    "ensemble": _predict_ensemble,
}


def evaluate_model(model_name: str, games: Iterable[GameRow]) -> Dict[str, object]:
    runner = MODEL_RUNNERS[model_name]
    acc = EvalAccumulator()

    for game in games:
        raw = runner(game)
        pred = _normalize_prediction(raw)

        acc.n_games += 1
        home_won = 1 if game.home_score > game.away_score else 0
        actual_total = game.home_score + game.away_score

        if pred["home_win_probability"] is not None:
            p = _clamp_prob(pred["home_win_probability"])
            pred_home_won = 1 if p >= 0.5 else 0
            acc.correct += int(pred_home_won == home_won)
            acc.n_prob += 1
            acc.brier_sum += (p - home_won) ** 2
            acc.log_loss_sum += _binary_log_loss(home_won, p)
        else:
            # Keep accuracy denominator aligned to n_games by counting no-prob predictions as unavailable.
            pass

        if pred["projected_total"] is not None:
            acc.n_totals += 1
            acc.totals_mae_sum += abs(pred["projected_total"] - actual_total)

    row = acc.as_row()
    row["model"] = model_name
    return row


def build_markdown(
    rows: List[Dict[str, object]],
    start_date: str,
    end_date: str,
    n_games: int,
    models: List[str],
) -> str:
    lines = [
        "# Unified Model Evaluation",
        "",
        f"- Date window: `{start_date}` to `{end_date}`",
        f"- Finalized games evaluated: `{n_games}`",
        f"- Models: `{', '.join(models)}`",
        f"- Protocol: same game set, same date window, read-only evaluation",
        "",
        "| Model | Win Accuracy | Brier | Log Loss | Totals MAE | n_games |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {acc} | {brier} | {log_loss} | {totals_mae} | {n_games} |".format(
                model=row["model"],
                acc=_fmt_metric(row["win_accuracy"]),
                brier=_fmt_metric(row["brier"]),
                log_loss=_fmt_metric(row["log_loss"]),
                totals_mae=_fmt_metric(row["totals_mae"], decimals=3),
                n_games=row["n_games"],
            )
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified evaluation for selected models")
    parser.add_argument("--start-date", default="2026-01-01", help="YYYY-MM-DD")
    parser.add_argument("--end-date", default="2026-12-31", help="YYYY-MM-DD")
    parser.add_argument(
        "--models",
        default="poisson,ensemble,elo",
        help="Comma-separated model list (supported: poisson,ensemble,elo)",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "artifacts" / "model_eval_unified.md"),
        help="Markdown output path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _install_readonly_sqlite_guard()

    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    invalid = [m for m in models if m not in MODEL_RUNNERS]
    if invalid:
        print(
            f"Unsupported models: {', '.join(invalid)}. Supported: {', '.join(sorted(MODEL_RUNNERS))}",
            file=sys.stderr,
        )
        return 2

    games = _load_games(args.start_date, args.end_date)
    if not games:
        print("No finalized games found in the specified date window.", file=sys.stderr)
        return 1

    results = [evaluate_model(name, games) for name in models]
    markdown = build_markdown(results, args.start_date, args.end_date, len(games), models)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")

    print(markdown, end="")
    print(f"Saved report to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
