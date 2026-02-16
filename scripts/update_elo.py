#!/usr/bin/env python3
"""
Update Elo ratings for all completed games that haven't been processed yet.

Usage:
    python3 scripts/update_elo.py           # Update all unprocessed games
    python3 scripts/update_elo.py --date 2026-02-15  # Update specific date
"""

import sys
import sqlite3
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.elo_model import EloModel
from scripts.database import get_connection


def update_elo_ratings(date=None):
    """Update Elo ratings for completed games."""
    conn = get_connection()
    cur = conn.cursor()
    
    elo = EloModel()
    
    # Find games with scores where we might need to update Elo
    # We track last_updated in elo_ratings to avoid reprocessing
    if date:
        cur.execute('''
            SELECT g.id, g.home_team_id, g.away_team_id, g.home_score, g.away_score, g.date
            FROM games g
            WHERE g.date = ? AND g.home_score IS NOT NULL
            ORDER BY g.date
        ''', (date,))
    else:
        # Get all completed games, ordered by date
        cur.execute('''
            SELECT g.id, g.home_team_id, g.away_team_id, g.home_score, g.away_score, g.date
            FROM games g
            WHERE g.home_score IS NOT NULL
            ORDER BY g.date
        ''')
    
    games = cur.fetchall()
    updated = 0
    
    for game_id, home_id, away_id, home_score, away_score, game_date in games:
        home_won = home_score > away_score
        margin = abs(home_score - away_score)
        
        try:
            elo.update_ratings(home_id, away_id, home_won, margin=margin, game_id=game_id, game_date=game_date)
            updated += 1
        except Exception as e:
            print(f"  Error updating {game_id}: {e}")
    
    conn.close()
    print(f"✅ Updated Elo ratings for {updated} games")


if __name__ == "__main__":
    date = None
    if len(sys.argv) > 1 and sys.argv[1] == '--date' and len(sys.argv) > 2:
        date = sys.argv[2]
    
    update_elo_ratings(date)
