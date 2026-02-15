#!/usr/bin/env python3
"""
Load DraftKings odds from JSON file into database
"""

import json
import sqlite3
from datetime import datetime
import sys

def normalize_team_id(team_name):
    """Normalize team name to database format"""
    # Simple normalization - convert to lowercase with hyphens
    normalized = team_name.lower().strip()
    replacements = {
        " ": "-",
        ".": "",
        "&": "and",
        "'": "",
        "state": "st",
        "university": "univ",
        "college": "coll",
        "southern": "south",
        "northern": "north",
        "eastern": "east",
        "western": "west"
    }
    
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    
    return normalized

def init_betting_table():
    """Create betting lines table if it doesn't exist"""
    conn = sqlite3.connect('data/baseball.db')
    c = conn.cursor()
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS betting_lines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT,
            date TEXT NOT NULL,
            home_team_id TEXT NOT NULL,
            away_team_id TEXT NOT NULL,
            book TEXT DEFAULT 'draftkings',
            -- Moneyline (American odds)
            home_ml INTEGER,
            away_ml INTEGER,
            -- Run line
            home_spread REAL,
            home_spread_odds INTEGER,
            away_spread REAL,
            away_spread_odds INTEGER,
            -- Totals
            over_under REAL,
            over_odds INTEGER,
            under_odds INTEGER,
            -- Timestamps
            captured_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(game_id, book)
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✓ Betting lines table ready")

def load_odds_from_json(json_file):
    """Load odds from JSON file into database"""
    init_betting_table()
    
    with open(json_file, 'r') as f:
        games = json.load(f)
    
    conn = sqlite3.connect('data/baseball.db')
    c = conn.cursor()
    
    today = datetime.now().strftime('%Y-%m-%d')
    added_count = 0
    updated_count = 0
    
    for game in games:
        away_team = game['away_team']
        home_team = game['home_team']
        away_id = normalize_team_id(away_team)
        home_id = normalize_team_id(home_team)
        game_id = f"{today}_{away_id}_{home_id}"
        
        # Extract odds data
        away_ml = game.get('away_ml')
        home_ml = game.get('home_ml')
        run_line_spread = game.get('run_line_spread')  # This is the away team spread
        away_rl_odds = game.get('away_rl_odds')
        home_rl_odds = game.get('home_rl_odds')
        total = game.get('total')
        over_odds = game.get('over_odds')
        under_odds = game.get('under_odds')
        
        # Convert away spread to home spread
        home_spread = -run_line_spread if run_line_spread else None
        away_spread = run_line_spread
        
        # Try to insert, update if exists
        try:
            c.execute('''
                INSERT INTO betting_lines 
                (game_id, date, home_team_id, away_team_id, book,
                 home_ml, away_ml, home_spread, home_spread_odds,
                 away_spread, away_spread_odds, over_under, over_odds, under_odds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (game_id, today, home_id, away_id, 'draftkings',
                  home_ml, away_ml, home_spread, home_rl_odds,
                  away_spread, away_rl_odds, total, over_odds, under_odds))
            added_count += 1
            print(f"✓ Added: {away_team} @ {home_team}")
        
        except sqlite3.IntegrityError:
            # Update existing record
            c.execute('''
                UPDATE betting_lines SET
                home_ml = ?, away_ml = ?, home_spread = ?, home_spread_odds = ?,
                away_spread = ?, away_spread_odds = ?, over_under = ?, 
                over_odds = ?, under_odds = ?, captured_at = CURRENT_TIMESTAMP
                WHERE game_id = ? AND book = ?
            ''', (home_ml, away_ml, home_spread, home_rl_odds,
                  away_spread, away_rl_odds, total, over_odds, under_odds,
                  game_id, 'draftkings'))
            updated_count += 1
            print(f"✓ Updated: {away_team} @ {home_team}")
    
    conn.commit()
    conn.close()
    
    print(f"\n📊 Summary: {added_count} added, {updated_count} updated")
    return added_count + updated_count

def show_todays_lines():
    """Display today's betting lines"""
    conn = sqlite3.connect('data/baseball.db')
    c = conn.cursor()
    
    today = datetime.now().strftime('%Y-%m-%d')
    
    c.execute('''
        SELECT away_team_id, home_team_id, away_ml, home_ml, over_under, captured_at
        FROM betting_lines
        WHERE date = ? AND book = 'draftkings'
        ORDER BY captured_at DESC
    ''', (today,))
    
    rows = c.fetchall()
    conn.close()
    
    if not rows:
        print(f"No DraftKings lines for {today}")
        return
    
    print(f"\n📊 DraftKings Lines for {today}:")
    print("=" * 70)
    print(f"{'Matchup':<40} {'Away ML':<10} {'Home ML':<10} {'Total':<8} {'Time'}")
    print("-" * 70)
    
    for row in rows:
        away_id, home_id, away_ml, home_ml, total, captured_at = row
        away_display = away_id.replace('-', ' ').title()
        home_display = home_id.replace('-', ' ').title()
        matchup = f"{away_display} @ {home_display}"
        
        away_ml_str = f"+{away_ml}" if away_ml > 0 else str(away_ml) if away_ml else "—"
        home_ml_str = f"+{home_ml}" if home_ml > 0 else str(home_ml) if home_ml else "—"
        total_str = str(total) if total else "—"
        time_str = captured_at.split(' ')[1][:5] if captured_at else "—"
        
        print(f"{matchup:<40} {away_ml_str:<10} {home_ml_str:<10} {total_str:<8} {time_str}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "show":
        show_todays_lines()
    else:
        total_games = load_odds_from_json('draftkings_odds.json')
        print(f"\n🎯 Successfully processed {total_games} games")
        show_todays_lines()