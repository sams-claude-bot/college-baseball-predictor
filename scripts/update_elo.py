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


def update_elo_ratings(date=None, force=False):
    """Update Elo ratings for completed games not yet in elo_history."""
    conn = get_connection()
    cur = conn.cursor()
    
    elo = EloModel()
    
    if force:
        # Force mode: clear elo_history and reprocess everything
        print("⚠️  Force mode: rebuilding all Elo ratings from scratch")
        cur.execute("DELETE FROM elo_history")
        cur.execute("DELETE FROM elo_ratings")
        conn.commit()
        elo.ratings = {}  # Clear in-memory cache
    
    # Find completed games NOT already in elo_history
    # elo_history has 2 entries per game (one per team), so we check for either
    if date:
        cur.execute('''
            SELECT g.id, g.home_team_id, g.away_team_id, g.home_score, g.away_score, g.date
            FROM games g
            WHERE g.date = ? 
              AND g.home_score IS NOT NULL
              AND g.status = 'final'
              AND g.id NOT IN (SELECT DISTINCT game_id FROM elo_history WHERE game_id IS NOT NULL)
            ORDER BY g.date, g.time
        ''', (date,))
    else:
        cur.execute('''
            SELECT g.id, g.home_team_id, g.away_team_id, g.home_score, g.away_score, g.date
            FROM games g
            WHERE g.home_score IS NOT NULL
              AND g.status = 'final'
              AND g.id NOT IN (SELECT DISTINCT game_id FROM elo_history WHERE game_id IS NOT NULL)
            ORDER BY g.date, g.time
        ''')
    
    games = cur.fetchall()
    
    if not games:
        print("✅ No new games to process")
        conn.close()
        return
    
    print(f"Processing {len(games)} unprocessed games...")
    updated = 0
    
    for game_id, home_id, away_id, home_score, away_score, game_date in games:
        home_won = home_score > away_score
        margin = abs(home_score - away_score)
        
        try:
            result = elo.update_ratings(home_id, away_id, home_won, margin=margin, 
                                        game_id=game_id, game_date=game_date)
            updated += 1
        except Exception as e:
            print(f"  Error updating {game_id}: {e}")
    
    conn.close()
    print(f"✅ Updated Elo ratings for {updated} games")
    
    # Also evaluate any pending predictions
    if updated > 0:
        try:
            from scripts.predict_and_track import evaluate_predictions
            evaluate_predictions()
        except Exception as e:
            print(f"  Evaluation note: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Update Elo ratings for completed games')
    parser.add_argument('--date', '-d', type=str, help='Process only games on this date (YYYY-MM-DD)')
    parser.add_argument('--force', '-f', action='store_true', 
                        help='Rebuild all ratings from scratch (clears elo_history)')
    args = parser.parse_args()
    
    update_elo_ratings(date=args.date, force=args.force)
