#!/usr/bin/env python3
"""
Slowly collect box scores from ESPN with rate limiting.
Aggregates player stats to build team statistics.

Usage:
    python3 collect_boxscores_slow.py              # Collect today's games
    python3 collect_boxscores_slow.py --date 2026-02-13  # Specific date
    python3 collect_boxscores_slow.py --delay 20   # Custom delay (seconds)
    python3 collect_boxscores_slow.py --limit 5    # Max games to process
"""

import re
import sys
import time
import json
import argparse
import sqlite3
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'

# ESPN game ID mappings we've seen
KNOWN_GAME_IDS = {
    # Will be populated as we discover them
}

def get_connection():
    """Get database connection"""
    return sqlite3.connect(str(DATA_DIR / 'baseball.db'))


def init_boxscore_tables():
    """Create tables for box score data"""
    conn = get_connection()
    cur = conn.cursor()
    
    cur.execute('''
        CREATE TABLE IF NOT EXISTS game_boxscores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT NOT NULL,
            espn_game_id TEXT,
            date TEXT NOT NULL,
            home_team TEXT,
            away_team TEXT,
            home_score INTEGER,
            away_score INTEGER,
            innings INTEGER,
            raw_json TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(game_id)
        )
    ''')
    
    cur.execute('''
        CREATE TABLE IF NOT EXISTS player_boxscore_batting (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT NOT NULL,
            team_id TEXT NOT NULL,
            player_name TEXT NOT NULL,
            position TEXT,
            at_bats INTEGER,
            runs INTEGER,
            hits INTEGER,
            rbi INTEGER,
            home_runs INTEGER,
            walks INTEGER,
            strikeouts INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cur.execute('''
        CREATE TABLE IF NOT EXISTS player_boxscore_pitching (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT NOT NULL,
            team_id TEXT NOT NULL,
            player_name TEXT NOT NULL,
            innings_pitched REAL,
            hits_allowed INTEGER,
            runs_allowed INTEGER,
            earned_runs INTEGER,
            walks INTEGER,
            strikeouts INTEGER,
            home_runs_allowed INTEGER,
            pitch_count INTEGER,
            strikes INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    print("Box score tables initialized")


def fetch_boxscore(espn_game_id, delay=15):
    """Fetch box score from ESPN"""
    try:
        import requests
    except ImportError:
        print("Error: requests library required")
        return None
    
    url = f"https://www.espn.com/college-baseball/boxscore/_/gameId/{espn_game_id}"
    print(f"  Fetching: {url}")
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code == 200:
            time.sleep(delay)  # Rate limit
            return resp.text
        elif resp.status_code == 429:
            print(f"  Rate limited! Waiting {delay * 3}s...")
            time.sleep(delay * 3)
            return None
        else:
            print(f"  HTTP {resp.status_code}")
            return None
    except Exception as e:
        print(f"  Error: {e}")
        return None


def parse_boxscore_text(text):
    """Parse ESPN box score text content"""
    result = {
        'home_team': None,
        'away_team': None,
        'home_score': None,
        'away_score': None,
        'line_score': None,
        'batting': {'home': [], 'away': []},
        'pitching': {'home': [], 'away': []},
    }
    
    lines = text.split('\n')
    
    # Find team names and scores
    # Pattern: "TeamName\n\nN\n\nFinal" or similar
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Look for "Final" marker
        if line == 'Final' and i > 0:
            # Score is typically a few lines before
            for j in range(i-1, max(0, i-5), -1):
                if lines[j].strip().isdigit():
                    if result['away_score'] is None:
                        result['away_score'] = int(lines[j].strip())
                    elif result['home_score'] is None:
                        result['home_score'] = int(lines[j].strip())
                        break
    
    # Parse batting stats - look for "Hitting" sections
    current_section = None
    current_team = None
    
    batting_headers = ['AB', 'R', 'H', 'RBI', 'HR', 'BB', 'K']
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        if 'Hitting' in line:
            # Team hitting section
            team_name = line.replace('Hitting', '').strip()
            if result['away_team'] is None:
                result['away_team'] = team_name
                current_team = 'away'
            else:
                result['home_team'] = team_name
                current_team = 'home'
            current_section = 'batting'
            continue
        
        if 'Pitching' in line:
            team_name = line.replace('Pitching', '').strip()
            current_team = 'home' if team_name == result['home_team'] else 'away'
            current_section = 'pitching'
            continue
        
        # Parse stat lines
        if current_section == 'batting' and current_team:
            # Look for player stat patterns
            # Format: "ABRHRBIHRBBBK" - 7 numbers
            if re.match(r'^\d{7,}$', line):
                stats = list(line)
                # Need to parse carefully - each stat can be 1-2 digits
                # Simple approach for now
                pass
    
    return result


def find_espn_game_ids(date_str):
    """Find ESPN game IDs for a given date from scoreboard"""
    try:
        import requests
    except ImportError:
        return []
    
    date_param = date_str.replace('-', '')
    url = f"https://www.espn.com/college-baseball/scoreboard?date={date_param}"
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code != 200:
            return []
        
        # Extract game IDs from URLs in response
        game_ids = re.findall(r'/gameId/(\d+)', resp.text)
        return list(set(game_ids))
    except Exception as e:
        print(f"Error fetching game IDs: {e}")
        return []


def process_game(espn_game_id, date_str, delay=15):
    """Process a single game's box score"""
    print(f"\nProcessing game {espn_game_id}...")
    
    html = fetch_boxscore(espn_game_id, delay)
    if not html:
        return False
    
    # For now, just save the raw response
    conn = get_connection()
    cur = conn.cursor()
    
    try:
        cur.execute('''
            INSERT OR REPLACE INTO game_boxscores 
            (game_id, espn_game_id, date, raw_json)
            VALUES (?, ?, ?, ?)
        ''', (f"espn_{espn_game_id}", espn_game_id, date_str, html[:50000]))
        conn.commit()
        print(f"  Saved box score for game {espn_game_id}")
        return True
    except Exception as e:
        print(f"  Error saving: {e}")
        return False
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description='Slowly collect ESPN box scores')
    parser.add_argument('--date', help='Date in YYYY-MM-DD format')
    parser.add_argument('--delay', type=int, default=15, help='Delay between requests (seconds)')
    parser.add_argument('--limit', type=int, help='Max games to process')
    args = parser.parse_args()
    
    date_str = args.date or datetime.now().strftime('%Y-%m-%d')
    
    # Initialize tables
    init_boxscore_tables()
    
    print(f"\nCollecting box scores for {date_str}")
    print(f"Delay: {args.delay}s between requests")
    
    # Find game IDs
    print("\nFinding game IDs from ESPN scoreboard...")
    game_ids = find_espn_game_ids(date_str)
    print(f"Found {len(game_ids)} games")
    
    if args.limit:
        game_ids = game_ids[:args.limit]
        print(f"Processing first {args.limit} games")
    
    # Process each game
    success = 0
    for gid in game_ids:
        if process_game(gid, date_str, args.delay):
            success += 1
    
    print(f"\n=== Complete ===")
    print(f"Processed {success}/{len(game_ids)} games")


if __name__ == '__main__':
    main()
