#!/usr/bin/env python3
"""
Coverage guardrail — verify every scheduled game has predictions from all 13 models.

Usage:
    python coverage_check.py              # Check today's scheduled games
    python coverage_check.py --date 2026-03-15   # Check specific date
    python coverage_check.py --fix        # Fill gaps via predict_and_track.py, then re-check
    python coverage_check.py --quiet      # Only output if gaps exist

Exit codes:
    0 = full coverage (all games have all 13 model predictions)
    1 = gaps remain
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import pytz

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.database import get_connection

MODEL_NAMES = [
    'pythagorean', 'elo', 'pitching', 'poisson', 'xgboost', 'lightgbm',
    'pear', 'quality', 'neural', 'venue', 'rest_travel', 'upset',
    'meta_ensemble',
]


def check_coverage(conn, date_str):
    """Return list of dicts with gap info for scheduled games on date_str.

    Each dict: {game_id, home_team, away_team, missing_count, missing_models}
    """
    cur = conn.cursor()

    # Get scheduled games for the date with team names
    games = cur.execute("""
        SELECT g.id, COALESCE(h.name, g.home_team_id) AS home,
               COALESCE(a.name, g.away_team_id) AS away
        FROM games g
        LEFT JOIN teams h ON g.home_team_id = h.id
        LEFT JOIN teams a ON g.away_team_id = a.id
        WHERE g.date = ? AND g.status = 'scheduled'
        ORDER BY g.id
    """, (date_str,)).fetchall()

    if not games:
        return []

    gaps = []
    for game in games:
        game_id, home, away = game['id'], game['home'], game['away']

        existing = {row['model_name'] for row in cur.execute(
            "SELECT model_name FROM model_predictions WHERE game_id = ?",
            (game_id,)
        ).fetchall()}

        missing = sorted(set(MODEL_NAMES) - existing)
        if missing:
            gaps.append({
                'game_id': game_id,
                'home_team': home,
                'away_team': away,
                'missing_count': len(missing),
                'missing_models': missing,
            })

    return gaps


def print_report(gaps, date_str, total_games):
    """Print human-readable coverage report."""
    if not gaps:
        print(f"Coverage OK: {total_games} scheduled game(s) on {date_str}, all fully covered.")
        return

    print(f"Coverage GAPS on {date_str}: {len(gaps)}/{total_games} game(s) missing predictions\n")
    for g in gaps:
        print(f"  {g['game_id']}  {g['away_team']} @ {g['home_team']}")
        print(f"    missing {g['missing_count']}/{len(MODEL_NAMES)}: {', '.join(g['missing_models'])}")
    print()


def run_fix(date_str):
    """Run predict_and_track.py to fill gaps for the given date."""
    script = Path(__file__).parent / "predict_and_track.py"
    cmd = [sys.executable, str(script), "predict", date_str]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env={**__import__('os').environ, 'PYTHONPATH': str(Path(__file__).parent.parent)})
    return result.returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check model prediction coverage for scheduled games")
    parser.add_argument("--date", help="Date to check (YYYY-MM-DD). Default: today Central Time.")
    parser.add_argument("--fix", action="store_true", help="Run predict_and_track.py to fill gaps, then re-check.")
    parser.add_argument("--quiet", action="store_true", help="Suppress output when fully covered.")
    args = parser.parse_args(argv)

    date_str = args.date or datetime.now(pytz.timezone("America/Chicago")).strftime("%Y-%m-%d")

    conn = get_connection()
    try:
        gaps = check_coverage(conn, date_str)
        total_games = conn.execute(
            "SELECT COUNT(*) FROM games WHERE date = ? AND status = 'scheduled'",
            (date_str,)
        ).fetchone()[0]

        if gaps and args.fix:
            print_report(gaps, date_str, total_games)
            run_fix(date_str)
            # Re-check after fix
            gaps = check_coverage(conn, date_str)

        if not args.quiet or gaps:
            print_report(gaps, date_str, total_games)
    finally:
        conn.close()

    return 1 if gaps else 0


if __name__ == "__main__":
    sys.exit(main())
