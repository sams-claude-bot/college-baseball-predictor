#!/usr/bin/env python3
"""
Update Elo ratings for completed games that haven't been processed yet.

Tracks processed games via elo_history table to avoid reprocessing.

Usage:
    python3 scripts/update_elo.py           # Update all unprocessed games
    python3 scripts/update_elo.py --date 2026-02-15  # Update specific date
    python3 scripts/update_elo.py --force   # Reprocess all games (rebuilds ratings)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.elo_model import EloModel
from scripts.database import get_connection
from scripts.run_utils import ScriptRunner


def update_elo_ratings(date=None, force=False, runner=None):
    """Update Elo ratings for completed games not yet in elo_history."""
    conn = get_connection()
    cur = conn.cursor()
    
    elo = EloModel()
    
    if force:
        # Force mode: clear elo_history and reprocess everything
        if runner:
            runner.warn("Force mode: rebuilding all Elo ratings from scratch")
        cur.execute("DELETE FROM elo_history")
        cur.execute("DELETE FROM elo_ratings")
        conn.commit()
        elo.ratings = {}  # Clear in-memory cache
    
    # Find completed games NOT already in elo_history
    # elo_history has 2 entries per game (one per team), so we check for either
    if date:
        cur.execute('''
            SELECT g.id, g.home_team_id, g.away_team_id, g.home_score, g.away_score, g.winner_id, g.date
            FROM games g
            WHERE g.date = ? 
              AND g.home_score IS NOT NULL
              AND g.status = 'final'
              AND g.id NOT IN (SELECT DISTINCT game_id FROM elo_history WHERE game_id IS NOT NULL)
            ORDER BY g.date, g.time
        ''', (date,))
    else:
        cur.execute('''
            SELECT g.id, g.home_team_id, g.away_team_id, g.home_score, g.away_score, g.winner_id, g.date
            FROM games g
            WHERE g.home_score IS NOT NULL
              AND g.status = 'final'
              AND g.id NOT IN (SELECT DISTINCT game_id FROM elo_history WHERE game_id IS NOT NULL)
            ORDER BY g.date, g.time
        ''')
    
    games = cur.fetchall()
    
    if not games:
        if runner:
            runner.info("No new games to process")
        conn.close()
        return 0
    
    if runner:
        runner.info(f"Processing {len(games)} unprocessed games...")
    updated = 0
    errors = 0
    
    for game_id, home_id, away_id, home_score, away_score, winner_id, game_date in games:
        try:
            # SQLite may return scores as TEXT depending on table history/migrations.
            hs = int(home_score)
            aws = int(away_score)
        except (TypeError, ValueError):
            if runner:
                runner.error(f"Invalid score values for {game_id}: home={home_score}, away={away_score}")
            errors += 1
            continue

        # Prefer winner_id when present (more robust for tournament home/away quirks).
        if winner_id == home_id:
            home_won = True
        elif winner_id == away_id:
            home_won = False
        else:
            home_won = hs > aws

        margin = abs(hs - aws)

        try:
            result = elo.update_ratings(home_id, away_id, home_won, margin=margin,
                                        game_id=game_id, game_date=game_date)
            updated += 1
        except Exception as e:
            if runner:
                runner.error(f"Error updating {game_id}: {e}")
            errors += 1
    
    conn.close()
    if runner:
        runner.info(f"Updated Elo ratings for {updated} games")
    
    # Also evaluate any pending predictions
    if updated > 0:
        try:
            from scripts.predict_and_track import evaluate_predictions
            evaluate_predictions()
        except Exception as e:
            if runner:
                runner.warn(f"Evaluation note: {e}")
    
    return updated, errors


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Update Elo ratings for completed games')
    parser.add_argument('--date', '-d', type=str, help='Process only games on this date (YYYY-MM-DD)')
    parser.add_argument('--force', '-f', action='store_true', 
                        help='Rebuild all ratings from scratch (clears elo_history)')
    args = parser.parse_args()
    
    runner = ScriptRunner("update_elo")
    
    result = update_elo_ratings(date=args.date, force=args.force, runner=runner)
    
    if result:
        updated, errors = result
        runner.add_stat("games_processed", updated)
        runner.add_stat("errors", errors)
    
    runner.finish()
