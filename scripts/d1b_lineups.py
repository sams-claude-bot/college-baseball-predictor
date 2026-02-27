#!/usr/bin/env python3
"""
D1Baseball Lineup Scraper

Scrapes historical lineup data from D1Baseball team pages.
This provides actual starting pitcher data for rotation analysis.

URL pattern: https://d1baseball.com/team/{team_slug}/lineup/

Data captured:
- Starting pitcher for each game
- Batting lineup (positions 1-9)
- Defensive positions

Usage:
    python3 scripts/d1b_lineups.py --team mississippi-state
    python3 scripts/d1b_lineups.py --conference SEC
    python3 scripts/d1b_lineups.py --all
    python3 scripts/d1b_lineups.py --show mississippi-state
"""

import argparse
import re
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("Error: playwright not installed")
    sys.exit(1)

PROJECT_DIR = Path(__file__).parent.parent
DB_PATH = PROJECT_DIR / 'data' / 'baseball.db'
SLUGS_PATH = PROJECT_DIR / 'config' / 'd1bb_slugs.json'

sys.path.insert(0, str(PROJECT_DIR / 'scripts'))
from team_resolver import resolve_team as db_resolve_team

# Load D1BB slugs from config (311 teams)
import json
if SLUGS_PATH.exists():
    _slug_data = json.load(open(SLUGS_PATH))
    D1BB_SLUGS = _slug_data.get('team_id_to_d1bb_slug', _slug_data)
else:
    print(f"Warning: {SLUGS_PATH} not found, using empty slug map")
    D1BB_SLUGS = {}

# Timing
PAGE_LOAD_WAIT = 5
BETWEEN_TEAMS_DELAY = 3


def get_connection():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def get_d1bb_slug(team_id):
    """Get D1Baseball URL slug for a team."""
    if team_id in D1BB_SLUGS:
        return D1BB_SLUGS[team_id]
    # Try common transformations
    slug = team_id.replace('-', '').replace(' ', '')
    return slug


def parse_lineup_page(text, team_id):
    """
    Parse the lineup page content.
    
    Returns list of game lineup records:
    [
        {
            'date': '2026-02-13',
            'opponent': 'Hofstra',
            'result': 'W, 6-5',
            'starting_pitcher': 'Ryan McPherson',
            'lineup': ['James Nunnallee', 'Ace Reese', ...],  # batting order
            'positions': {'C': 'Kevin Milewski', '1B': 'Blake Bevis', ...}
        },
        ...
    ]
    """
    games = []
    lines = text.split('\n')
    
    # Find the defensive lineup section (has SP column)
    in_defensive_section = False
    current_game = None
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Look for game lines: "Fri, Feb 13 vs Hofstra (W, 6-5)"
        game_match = re.match(
            r'(Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s+(Jan|Feb|Mar|Apr|May|Jun)\s+(\d+)\s+'
            r'(vs|at|@)\s+(.+?)\s+\(([WL]),\s*([\d-]+)\)',
            line
        )
        
        if game_match:
            dow, month, day, home_away, opponent, result, score = game_match.groups()
            
            # Build date (assume 2026 season)
            month_num = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6}.get(month, 2)
            date_str = f"2026-{month_num:02d}-{int(day):02d}"
            
            current_game = {
                'date': date_str,
                'day_of_week': dow,
                'opponent': opponent.strip(),
                'home_away': 'home' if home_away == 'vs' else 'away',
                'result': result,
                'score': score,
                'starting_pitcher': None,
                'lineup': [],
                'positions': {}
            }
            games.append(current_game)
            continue
        
        # Look for SP column data - usually appears after position names
        # The defensive lineup shows: C, 1B, 2B, SS, 3B, LF, CF, RF, DH, SP
        if current_game and not current_game['starting_pitcher']:
            # Check if this line contains a player name that might be SP
            # After the game line, look for position data
            
            # Pattern: position assignments often appear tab-separated or in sequence
            # The SP is typically the last column in the defensive lineup row
            
            # Look for known pitcher names or just capture names after DH
            pass
    
    # More robust parsing: look for the table structure
    # The page has two tables:
    # 1. Batting order (1-9)
    # 2. Defensive positions (C, 1B, 2B, SS, 3B, LF, CF, RF, DH, SP)
    
    # Parse the raw text more carefully
    games = []
    
    # Split by game entries
    game_pattern = re.compile(
        r'((?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s+(?:Jan|Feb|Mar|Apr|May|Jun)\s+\d+\s+'
        r'(?:vs|at|@)\s+.+?\s+\([WL],\s*[\d-]+\))'
    )
    
    # Find all game headers
    game_headers = list(game_pattern.finditer(text))
    
    for idx, match in enumerate(game_headers):
        header = match.group(1)
        start_pos = match.end()
        end_pos = game_headers[idx + 1].start() if idx + 1 < len(game_headers) else len(text)
        
        game_section = text[start_pos:end_pos]
        
        # Parse header
        header_match = re.match(
            r'(Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s+(Jan|Feb|Mar|Apr|May|Jun)\s+(\d+)\s+'
            r'(vs|at|@)\s+(.+?)\s+\(([WL]),\s*([\d-]+)\)',
            header
        )
        
        if not header_match:
            continue
            
        dow, month, day, home_away, opponent, result, score = header_match.groups()
        month_num = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6}.get(month, 2)
        date_str = f"2026-{month_num:02d}-{int(day):02d}"
        
        game = {
            'date': date_str,
            'day_of_week': dow,
            'opponent': opponent.strip(),
            'home_away': 'home' if home_away == 'vs' else 'away',
            'result': result,
            'score': score,
            'starting_pitcher': None,
            'lineup': [],
        }
        
        # Extract names from the game section
        # Names appear as "Player Name - Position" or just "Player Name"
        names = re.findall(r'([A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', game_section)
        
        # The last name in a defensive position row should be SP
        # Look for the sequence ending in common pitcher indicators
        if names:
            game['lineup'] = names[:9]  # First 9 are batting order
            # Try to find SP - often last in the row
            if len(names) > 9:
                game['starting_pitcher'] = names[-1]  # Last name is often SP
        
        games.append(game)
    
    return games


def scrape_team_lineups(team_id, dry_run=False):
    """Scrape lineup history for a team from D1Baseball."""
    d1bb_slug = get_d1bb_slug(team_id)
    url = f"https://d1baseball.com/team/{d1bb_slug}/lineup/"
    
    print(f"\nScraping: {team_id} ({url})")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        try:
            page.goto(url, wait_until='domcontentloaded', timeout=30000)
            time.sleep(PAGE_LOAD_WAIT)
            
            # Scroll to load all content
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(2)
            
            text = page.inner_text('body')
            lines = text.split('\n')
            
            # Find the defensive section header (GAME C 1B 2B ... SP)
            def_start = None
            for i, line in enumerate(lines):
                if line.strip().startswith('GAME\tC\t1B'):
                    def_start = i
                    break
            
            if not def_start:
                print("  Could not find defensive lineup section")
                browser.close()
                return []
            
            # Parse games after the header
            games = []
            i = def_start + 1
            
            while i < len(lines):
                line = lines[i].strip()
                
                # Look for game header
                game_match = re.match(
                    r'^(Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s+(Jan|Feb|Mar|Apr|May|Jun)\s+(\d+)\s+'
                    r'(vs|at|@)\s+(.+?)\s+\(([WL]),\s*([\d-]+)\)',
                    line
                )
                
                if game_match:
                    dow, month, day, home_away, opponent, result, score = game_match.groups()
                    month_num = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6}.get(month, 2)
                    
                    # Collect next 10 names (C, 1B, 2B, SS, 3B, LF, CF, RF, DH, SP)
                    names = []
                    j = i + 1
                    while len(names) < 10 and j < len(lines):
                        name_line = lines[j].strip()
                        # Full name pattern
                        if re.match(r'^[A-Z][a-z]+ [A-Z][a-zA-Z]+', name_line):
                            names.append(name_line)
                        elif name_line.startswith(('Mon,', 'Tue,', 'Wed,', 'Thu,', 'Fri,', 'Sat,', 'Sun,')):
                            break  # Next game
                        elif 'Games' in name_line or 'Most' in name_line or 'About' in name_line:
                            break  # End of table
                        j += 1
                    
                    if len(names) == 10:
                        sp = names[9]  # 10th position is SP
                        games.append({
                            'date': f"2026-{month_num:02d}-{int(day):02d}",
                            'day_of_week': dow,
                            'opponent': opponent.strip(),
                            'home_away': 'home' if home_away == 'vs' else 'away',
                            'result': result,
                            'starting_pitcher': sp
                        })
                    
                    i = j
                else:
                    i += 1
                    # Stop if we hit footer content
                    if 'About Us' in line or 'Contact' in line:
                        break
            
            print(f"  Found {len(games)} games")
            for g in games:
                sp = g.get('starting_pitcher', 'Unknown')
                print(f"    {g['date']} ({g['day_of_week']}) vs {g['opponent']}: SP = {sp}")
            
            browser.close()
            
            if not dry_run and games:
                save_lineup_data(team_id, games)
            
            return games
            
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            browser.close()
            return []


def save_lineup_data(team_id, games):
    """Save lineup/starter data to database."""
    conn = get_connection()
    c = conn.cursor()
    
    # Create table if needed
    c.execute('''
        CREATE TABLE IF NOT EXISTS lineup_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_id TEXT NOT NULL,
            game_date TEXT NOT NULL,
            game_num INTEGER DEFAULT 1,
            day_of_week TEXT,
            opponent TEXT,
            result TEXT,
            starting_pitcher TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(team_id, game_date, opponent, game_num)
        )
    ''')
    
    inserted = 0
    # Track game numbers for doubleheaders
    date_opponent_count = {}
    for game in games:
        if not game.get('starting_pitcher'):
            continue
        
        key = (game['date'], game['opponent'])
        game_num = date_opponent_count.get(key, 0) + 1
        date_opponent_count[key] = game_num
        
        try:
            c.execute('''
                INSERT OR REPLACE INTO lineup_history 
                (team_id, game_date, game_num, day_of_week, opponent, result, starting_pitcher)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (team_id, game['date'], game_num, game['day_of_week'], 
                  game['opponent'], game['result'], game['starting_pitcher']))
            inserted += 1
        except Exception as e:
            print(f"    DB error: {e}")
    
    conn.commit()
    conn.close()
    print(f"  Saved {inserted} games to lineup_history")


def show_team_rotation(team_id):
    """Show rotation analysis for a team."""
    conn = get_connection()
    c = conn.cursor()
    
    c.execute('''
        SELECT game_date, day_of_week, opponent, result, starting_pitcher
        FROM lineup_history
        WHERE team_id = ?
        ORDER BY game_date
    ''', (team_id,))
    
    games = c.fetchall()
    
    if not games:
        print(f"No lineup data for {team_id}")
        return
    
    print(f"\n=== {team_id.upper()} Rotation Analysis ===")
    print(f"{'Date':<12} {'Day':<4} {'Opponent':<20} {'Result':<4} {'SP':<20}")
    print("-" * 65)
    
    by_dow = {}
    for g in games:
        print(f"{g['game_date']:<12} {g['day_of_week']:<4} {g['opponent']:<20} {g['result']:<4} {g['starting_pitcher']:<20}")
        dow = g['day_of_week']
        if dow not in by_dow:
            by_dow[dow] = []
        by_dow[dow].append(g['starting_pitcher'])
    
    print("\n--- By Day of Week ---")
    for dow in ['Fri', 'Sat', 'Sun', 'Tue', 'Wed', 'Thu', 'Mon']:
        if dow in by_dow:
            pitchers = by_dow[dow]
            from collections import Counter
            counts = Counter(pitchers)
            most_common = counts.most_common(1)[0] if counts else ('?', 0)
            print(f"  {dow}: {most_common[0]} ({most_common[1]}/{len(pitchers)} starts)")
    
    conn.close()


def main():
    parser = argparse.ArgumentParser(description='Scrape D1Baseball lineup data')
    parser.add_argument('--team', type=str, help='Single team to scrape')
    parser.add_argument('--conference', type=str, help='Conference to scrape (e.g., SEC)')
    parser.add_argument('--all', action='store_true', help='Scrape all tracked teams')
    parser.add_argument('--show', type=str, help='Show rotation analysis for team')
    parser.add_argument('--dry-run', action='store_true', help='Don\'t save to database')
    args = parser.parse_args()
    
    if args.show:
        show_team_rotation(args.show)
        return
    
    if args.team:
        scrape_team_lineups(args.team, dry_run=args.dry_run)
    elif args.conference:
        conn = get_connection()
        c = conn.cursor()
        c.execute('SELECT id FROM teams WHERE conference = ?', (args.conference,))
        teams = [r['id'] for r in c.fetchall()]
        conn.close()
        
        print(f"Scraping {len(teams)} {args.conference} teams...")
        for team_id in teams:
            scrape_team_lineups(team_id, dry_run=args.dry_run)
            time.sleep(BETWEEN_TEAMS_DELAY)
    elif args.all:
        conn = get_connection()
        c = conn.cursor()
        c.execute('SELECT id FROM teams ORDER BY conference, id')
        teams = [r['id'] for r in c.fetchall()]
        conn.close()
        
        print(f"Scraping all {len(teams)} teams...")
        success = 0
        failed = 0
        for i, team_id in enumerate(teams):
            print(f"\n[{i+1}/{len(teams)}] ", end='')
            try:
                games = scrape_team_lineups(team_id, dry_run=args.dry_run)
                if games:
                    success += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"  FAILED: {e}")
                failed += 1
            time.sleep(BETWEEN_TEAMS_DELAY)
        
        print(f"\n=== Done: {success} teams with data, {failed} failed/empty ===")
    else:
        print("Usage:")
        print("  --team mississippi-state  # Single team")
        print("  --conference SEC          # All SEC teams")
        print("  --all                     # All tracked teams")
        print("  --show mississippi-state  # View rotation analysis")


if __name__ == '__main__':
    main()
