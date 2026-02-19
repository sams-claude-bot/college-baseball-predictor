#!/usr/bin/env python3
"""
D1Baseball Lineup Scraper

Scrapes starting lineups from D1Baseball game preview pages.
Captures:
- Starting pitcher (most valuable for predictions)
- Batting lineup (9 batters with positions)

Usage:
    python3 scripts/d1bb_lineups.py --today              # Today's games
    python3 scripts/d1bb_lineups.py --date 2026-02-20    # Specific date
    python3 scripts/d1bb_lineups.py --dry-run            # Don't save to DB

D1Baseball shows lineups on the scoreboard page and individual game pages
when teams submit them (typically a few hours before game time).
"""

import argparse
import json
import re
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
DB_PATH = PROJECT_DIR / 'data' / 'baseball.db'
SNAPSHOT_DIR = PROJECT_DIR / 'data' / 'snapshots'

sys.path.insert(0, str(PROJECT_DIR / 'scripts'))
from team_resolver import resolve_team as db_resolve_team

# Timing
PAGE_LOAD_WAIT = 5
BETWEEN_PAGES_DELAY = 3


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def resolve_team(name):
    """Resolve team name to our team_id."""
    if not name:
        return None
    result = db_resolve_team(name)
    if result:
        return result
    # Fallback slugify
    slug = name.lower().strip()
    slug = re.sub(r'[^a-z0-9]+', '-', slug).strip('-')
    return slug


def find_player_id(conn, team_id, player_name):
    """Find player ID from player_stats table."""
    if not player_name or not team_id:
        return None
    
    c = conn.cursor()
    
    # Try exact match first
    c.execute('''
        SELECT id FROM player_stats 
        WHERE team_id = ? AND LOWER(name) = LOWER(?)
        LIMIT 1
    ''', (team_id, player_name.strip()))
    row = c.fetchone()
    if row:
        return row['id']
    
    # Try partial match (last name)
    parts = player_name.strip().split()
    if parts:
        last_name = parts[-1]
        c.execute('''
            SELECT id FROM player_stats 
            WHERE team_id = ? AND LOWER(name) LIKE ?
            LIMIT 1
        ''', (team_id, f'%{last_name.lower()}%'))
        row = c.fetchone()
        if row:
            return row['id']
    
    return None


def parse_lineup_from_snapshot(text, home_team, away_team):
    """
    Parse lineup data from a D1Baseball snapshot.
    
    Returns:
    {
        'home_starter': {'name': str, 'id': int or None},
        'away_starter': {'name': str, 'id': int or None},
        'home_lineup': [{'name': str, 'position': str}, ...],
        'away_lineup': [{'name': str, 'position': str}, ...],
    }
    """
    result = {
        'home_starter': None,
        'away_starter': None,
        'home_lineup': [],
        'away_lineup': [],
    }
    
    lines = text.split('\n')
    
    # Look for patterns like:
    # "Starting Pitcher" or "SP:" followed by player name
    # "Lineup" sections with numbered batters
    
    # This is a basic parser - D1Baseball's actual format may vary
    current_team = None
    in_lineup = False
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Detect team context
        if home_team and home_team.lower() in line.lower():
            current_team = 'home'
        elif away_team and away_team.lower() in line.lower():
            current_team = 'away'
        
        # Starting pitcher patterns
        sp_patterns = [
            r'(?:SP|Starting Pitcher|Probable)[:\s]+(.+)',
            r'(?:RHP|LHP)\s+(.+)',
        ]
        
        for pattern in sp_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match and current_team:
                name = match.group(1).strip()
                # Clean up name
                name = re.sub(r'\s*\(.*\)', '', name)  # Remove stats in parens
                name = re.sub(r'\s*#\d+', '', name)    # Remove jersey numbers
                if name and len(name) > 2:
                    result[f'{current_team}_starter'] = {'name': name}
                    break
        
        # Lineup patterns (numbered 1-9)
        lineup_match = re.match(r'^(\d)[.\)]\s*(.+?)\s+(C|1B|2B|3B|SS|LF|CF|RF|DH|P)', line, re.IGNORECASE)
        if lineup_match and current_team:
            order = int(lineup_match.group(1))
            name = lineup_match.group(2).strip()
            position = lineup_match.group(3).upper()
            result[f'{current_team}_lineup'].append({
                'order': order,
                'name': name,
                'position': position
            })
    
    return result


def scrape_lineups_playwright(date_str, dry_run=False):
    """
    Scrape lineup data from D1Baseball using Playwright.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Error: playwright not installed")
        return []
    
    conn = get_connection()
    c = conn.cursor()
    
    # Get games for this date
    c.execute('''
        SELECT g.id, g.home_team_id, g.away_team_id, 
               h.name as home_name, a.name as away_name
        FROM games g
        JOIN teams h ON g.home_team_id = h.id
        JOIN teams a ON g.away_team_id = a.id
        WHERE g.date = ? AND g.status = 'scheduled'
    ''', (date_str,))
    games = c.fetchall()
    
    if not games:
        print(f"No scheduled games found for {date_str}")
        return []
    
    print(f"Found {len(games)} scheduled games for {date_str}")
    
    SNAPSHOT_DIR.mkdir(exist_ok=True)
    results = []
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        
        # Go to D1Baseball scoreboard
        url = f"https://d1baseball.com/scores/?date={date_str}"
        print(f"Loading: {url}")
        
        try:
            page.goto(url, wait_until='domcontentloaded', timeout=30000)
            time.sleep(PAGE_LOAD_WAIT)
            
            # Scroll to load all content
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(2)
            
            # Get page content
            content = page.content()
            text_content = page.inner_text('body')
            
            # Save snapshot
            snapshot_file = SNAPSHOT_DIR / f"d1bb_lineups_{date_str}.txt"
            with open(snapshot_file, 'w') as f:
                f.write(f"URL: {url}\n")
                f.write(f"Date: {datetime.now().isoformat()}\n\n")
                f.write(text_content)
            print(f"Saved snapshot: {snapshot_file}")
            
            # Try to find lineup data for each game
            for game in games:
                game_id = game['id']
                home_name = game['home_name']
                away_name = game['away_name']
                home_id = game['home_team_id']
                away_id = game['away_team_id']
                
                print(f"\n  {away_name} @ {home_name}:")
                
                # Parse lineup from content
                lineup_data = parse_lineup_from_snapshot(text_content, home_name, away_name)
                
                home_starter = lineup_data.get('home_starter')
                away_starter = lineup_data.get('away_starter')
                
                if home_starter or away_starter:
                    # Try to find player IDs
                    home_starter_id = None
                    away_starter_id = None
                    home_starter_name = None
                    away_starter_name = None
                    
                    if home_starter:
                        home_starter_name = home_starter['name']
                        home_starter_id = find_player_id(conn, home_id, home_starter_name)
                        print(f"    Home SP: {home_starter_name}" + 
                              (f" (ID: {home_starter_id})" if home_starter_id else " (not in DB)"))
                    
                    if away_starter:
                        away_starter_name = away_starter['name']
                        away_starter_id = find_player_id(conn, away_id, away_starter_name)
                        print(f"    Away SP: {away_starter_name}" + 
                              (f" (ID: {away_starter_id})" if away_starter_id else " (not in DB)"))
                    
                    if not dry_run and (home_starter_name or away_starter_name):
                        # Update or insert pitching matchup
                        c.execute('SELECT id FROM pitching_matchups WHERE game_id = ?', (game_id,))
                        existing = c.fetchone()
                        
                        if existing:
                            c.execute('''
                                UPDATE pitching_matchups SET
                                    home_starter_id = COALESCE(?, home_starter_id),
                                    away_starter_id = COALESCE(?, away_starter_id),
                                    home_starter_name = COALESCE(?, home_starter_name),
                                    away_starter_name = COALESCE(?, away_starter_name),
                                    notes = 'd1baseball_lineup'
                                WHERE game_id = ?
                            ''', (home_starter_id, away_starter_id, 
                                  home_starter_name, away_starter_name, game_id))
                        else:
                            c.execute('''
                                INSERT INTO pitching_matchups 
                                (game_id, home_starter_id, away_starter_id, 
                                 home_starter_name, away_starter_name, notes)
                                VALUES (?, ?, ?, ?, ?, 'd1baseball_lineup')
                            ''', (game_id, home_starter_id, away_starter_id,
                                  home_starter_name, away_starter_name))
                        
                        results.append({
                            'game_id': game_id,
                            'home_starter': home_starter_name,
                            'away_starter': away_starter_name
                        })
                else:
                    print(f"    No lineup data found")
            
        except Exception as e:
            print(f"Error scraping: {e}")
        finally:
            browser.close()
    
    if not dry_run:
        conn.commit()
    conn.close()
    
    return results


def show_current_matchups(date_str=None):
    """Show pitching matchups we have for a date."""
    conn = get_connection()
    c = conn.cursor()
    
    if date_str is None:
        date_str = datetime.now().strftime('%Y-%m-%d')
    
    c.execute('''
        SELECT g.id, h.name as home, a.name as away,
               pm.home_starter_name, pm.away_starter_name, pm.notes
        FROM games g
        JOIN teams h ON g.home_team_id = h.id
        JOIN teams a ON g.away_team_id = a.id
        LEFT JOIN pitching_matchups pm ON g.id = pm.game_id
        WHERE g.date = ?
        ORDER BY g.time
    ''', (date_str,))
    
    print(f"\n=== Pitching Matchups for {date_str} ===")
    
    games = c.fetchall()
    with_matchups = 0
    
    for game in games:
        home_sp = game['home_starter_name'] or 'TBD'
        away_sp = game['away_starter_name'] or 'TBD'
        source = f"({game['notes']})" if game['notes'] else ''
        
        if game['home_starter_name'] or game['away_starter_name']:
            with_matchups += 1
        
        print(f"  {game['away']:20} @ {game['home']:20}  |  {away_sp:15} vs {home_sp:15} {source}")
    
    print(f"\n{with_matchups}/{len(games)} games have pitcher data")
    conn.close()


def main():
    parser = argparse.ArgumentParser(description='Scrape D1Baseball lineup data')
    parser.add_argument('--today', action='store_true', help='Scrape today\'s games')
    parser.add_argument('--date', type=str, help='Date to scrape (YYYY-MM-DD)')
    parser.add_argument('--show', action='store_true', help='Show current matchups only')
    parser.add_argument('--dry-run', action='store_true', help='Don\'t save to database')
    args = parser.parse_args()
    
    if args.show:
        date_str = args.date or datetime.now().strftime('%Y-%m-%d')
        show_current_matchups(date_str)
        return
    
    if args.today:
        date_str = datetime.now().strftime('%Y-%m-%d')
    elif args.date:
        date_str = args.date
    else:
        # Default to tomorrow (lineups usually posted day before)
        date_str = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    
    print(f"Scraping lineups for {date_str}")
    results = scrape_lineups_playwright(date_str, dry_run=args.dry_run)
    
    if results:
        print(f"\n✅ Found {len(results)} games with lineup data")
    else:
        print(f"\n⚠️  No lineup data found (may not be posted yet)")


if __name__ == '__main__':
    main()
