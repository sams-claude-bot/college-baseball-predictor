#!/usr/bin/env python3
"""
P4 Team Stats Scraper - Collects batting and pitching stats from team athletics sites.

Uses browser automation to scrape SIDEARM Sports pages (most P4 schools).
Designed for biweekly collection: Sunday nights and Thursday mornings.

Usage:
    python3 p4_stats_scraper.py                    # Scrape all P4 teams
    python3 p4_stats_scraper.py --conference SEC   # Scrape only SEC
    python3 p4_stats_scraper.py --team alabama     # Scrape single team
    python3 p4_stats_scraper.py --limit 10         # Scrape first N teams
    python3 p4_stats_scraper.py --resume           # Resume from last progress
    python3 p4_stats_scraper.py --dry-run          # Test without database updates
"""

import argparse
import json
import sys
import time
import re
import sqlite3
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.database import get_connection

# Constants
DATA_DIR = Path(__file__).parent.parent / 'data'
TEAM_URLS_FILE = DATA_DIR / 'p4_team_urls.json'
PROGRESS_FILE = DATA_DIR / 'stats_scraper_progress.json'

# Delay between teams to avoid rate limiting (seconds)
TEAM_DELAY = 15
PAGE_LOAD_DELAY = 3

# Column mappings for SIDEARM Sports tables
BATTING_COLUMNS = [
    'number', 'name', 'avg', 'ops', 'gp_gs', 'ab', 'r', 'h', 
    '2b', '3b', 'hr', 'rbi', 'tb', 'slg', 'bb', 'hbp', 'so', 
    'gdp', 'obp', 'sf', 'sh', 'sb_att'
]

PITCHING_COLUMNS = [
    'number', 'name', 'era', 'whip', 'w_l', 'app_gs', 'cg', 'sho', 
    'sv', 'ip', 'h', 'r', 'er', 'bb', 'so', '2b', '3b', 'hr', 
    'ab', 'bavg', 'wp', 'hbp', 'bk', 'sfa', 'sha'
]


def load_team_urls():
    """Load team URLs from JSON file."""
    if not TEAM_URLS_FILE.exists():
        print(f"Error: Team URLs file not found: {TEAM_URLS_FILE}")
        sys.exit(1)
    
    with open(TEAM_URLS_FILE) as f:
        data = json.load(f)
    return data.get('teams', {})


def load_progress():
    """Load scraping progress for resume."""
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE) as f:
                return json.load(f)
        except:
            pass
    return {'completed': [], 'last_run': None, 'failed': []}


def save_progress(progress):
    """Save scraping progress."""
    progress['last_run'] = datetime.now().isoformat()
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def get_teams_by_conference(team_urls: dict, conference: str) -> dict:
    """Filter teams by conference."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id FROM teams WHERE conference = ?", (conference,))
    team_ids = {row[0] for row in cur.fetchall()}
    conn.close()
    return {k: v for k, v in team_urls.items() if k in team_ids}


def parse_stat_value(value: str, stat_type: str = 'float'):
    """Parse a stat value from the table."""
    if not value or value == '-':
        return None
    
    value = value.strip().replace('*', '')  # Remove asterisk from names
    
    if stat_type == 'int':
        try:
            return int(value.replace(',', ''))
        except:
            return None
    elif stat_type == 'float':
        try:
            return float(value)
        except:
            return None
    elif stat_type == 'split':
        # Handle formats like "3 - 3" or "1-0"
        parts = value.replace(' ', '').split('-')
        if len(parts) == 2:
            try:
                return (int(parts[0]), int(parts[1]))
            except:
                pass
        return None
    else:
        return value


def parse_batting_row(cells: list, team_id: str) -> dict:
    """Parse a batting stats row."""
    if len(cells) < 20:
        return None
    
    # Skip totals and opponents rows
    if 'total' in cells[1].lower() or 'opponent' in cells[1].lower():
        return None
    
    name = cells[1].strip().replace('*', '').strip()
    if not name:
        return None
    
    # Parse games played - handle "3 - 3" format
    gp_gs = parse_stat_value(cells[4], 'split')
    games = gp_gs[0] if gp_gs else 0
    games_started = gp_gs[1] if gp_gs else 0
    
    # Parse stolen bases - handle "2 - 2" format
    sb_att = parse_stat_value(cells[21], 'split') if len(cells) > 21 else None
    sb = sb_att[0] if sb_att else 0
    cs = sb_att[1] - sb_att[0] if sb_att and sb_att[1] >= sb_att[0] else 0
    
    return {
        'team_id': team_id,
        'name': name,
        'number': parse_stat_value(cells[0], 'int'),
        'games': games,
        'at_bats': parse_stat_value(cells[5], 'int') or 0,
        'runs': parse_stat_value(cells[6], 'int') or 0,
        'hits': parse_stat_value(cells[7], 'int') or 0,
        'doubles': parse_stat_value(cells[8], 'int') or 0,
        'triples': parse_stat_value(cells[9], 'int') or 0,
        'home_runs': parse_stat_value(cells[10], 'int') or 0,
        'rbi': parse_stat_value(cells[11], 'int') or 0,
        'walks': parse_stat_value(cells[14], 'int') or 0,
        'strikeouts': parse_stat_value(cells[16], 'int') or 0,
        'stolen_bases': sb,
        'caught_stealing': cs,
        'batting_avg': parse_stat_value(cells[2], 'float') or 0,
        'obp': parse_stat_value(cells[18], 'float') or 0,
        'slg': parse_stat_value(cells[13], 'float') or 0,
        'ops': parse_stat_value(cells[3], 'float') or 0,
    }


def parse_pitching_row(cells: list, team_id: str) -> dict:
    """Parse a pitching stats row."""
    if len(cells) < 20:
        return None
    
    # Skip totals and opponents rows
    if 'total' in cells[1].lower() or 'opponent' in cells[1].lower():
        return None
    
    name = cells[1].strip().replace('*', '').strip()
    if not name:
        return None
    
    # Parse W-L record
    w_l = parse_stat_value(cells[4], 'split')
    wins = w_l[0] if w_l else 0
    losses = w_l[1] if w_l else 0
    
    # Parse App-GS (appearances - games started)
    app_gs = parse_stat_value(cells[5], 'split')
    games_pitched = app_gs[0] if app_gs else 0
    games_started = app_gs[1] if app_gs else 0
    
    # Parse innings pitched (handle "5.1" format)
    ip_str = cells[9].strip()
    try:
        if '.' in ip_str:
            parts = ip_str.split('.')
            innings = float(parts[0]) + float(parts[1]) / 3
        else:
            innings = float(ip_str)
    except:
        innings = 0
    
    return {
        'team_id': team_id,
        'name': name,
        'number': parse_stat_value(cells[0], 'int'),
        'wins': wins,
        'losses': losses,
        'era': parse_stat_value(cells[2], 'float') or 0,
        'games_pitched': games_pitched,
        'games_started': games_started,
        'saves': parse_stat_value(cells[8], 'int') or 0,
        'innings_pitched': innings,
        'hits_allowed': parse_stat_value(cells[10], 'int') or 0,
        'runs_allowed': parse_stat_value(cells[11], 'int') or 0,
        'earned_runs': parse_stat_value(cells[12], 'int') or 0,
        'walks_allowed': parse_stat_value(cells[13], 'int') or 0,
        'strikeouts_pitched': parse_stat_value(cells[14], 'int') or 0,
        'whip': parse_stat_value(cells[3], 'float') or 0,
    }


def extract_table_data(snapshot_text: str, table_type: str = 'batting') -> list:
    """Extract table rows from browser snapshot text."""
    rows = []
    
    # Look for table rows in the snapshot
    # Format: "row "3 Reese, Ace .417 1.379 3 - 3 12 6 5 3 0 1 4 11 .917 0 1 4 0 .462 0 0 0 - 0":"
    row_pattern = r'row "([^"]+)"'
    
    for match in re.finditer(row_pattern, snapshot_text):
        row_text = match.group(1)
        
        # Skip header rows
        if 'undefined' in row_text.lower():
            continue
        
        # Split by spaces, but handle "X - Y" patterns
        cells = re.split(r'\s+(?=\d|\*|[A-Za-z])', row_text)
        
        # Clean cells
        cells = [c.strip() for c in cells if c.strip()]
        
        if len(cells) >= 10:
            rows.append(cells)
    
    return rows


def update_player_stats(stats: dict, stat_type: str = 'batting', dry_run: bool = False):
    """Update player stats in database."""
    if dry_run:
        print(f"  [DRY RUN] Would update {stats['name']} ({stats['team_id']})")
        return True
    
    conn = get_connection()
    cur = conn.cursor()
    
    try:
        # Check if player exists
        cur.execute(
            "SELECT id FROM player_stats WHERE team_id = ? AND name = ?",
            (stats['team_id'], stats['name'])
        )
        existing = cur.fetchone()
        
        if stat_type == 'batting':
            if existing:
                cur.execute('''
                    UPDATE player_stats SET
                        number = ?, games = ?, at_bats = ?, runs = ?, hits = ?,
                        doubles = ?, triples = ?, home_runs = ?, rbi = ?, walks = ?,
                        strikeouts = ?, stolen_bases = ?, caught_stealing = ?,
                        batting_avg = ?, obp = ?, slg = ?, ops = ?, updated_at = ?
                    WHERE team_id = ? AND name = ?
                ''', (
                    stats['number'], stats['games'], stats['at_bats'], stats['runs'],
                    stats['hits'], stats['doubles'], stats['triples'], stats['home_runs'],
                    stats['rbi'], stats['walks'], stats['strikeouts'], stats['stolen_bases'],
                    stats['caught_stealing'], stats['batting_avg'], stats['obp'],
                    stats['slg'], stats['ops'], datetime.now().isoformat(),
                    stats['team_id'], stats['name']
                ))
            else:
                cur.execute('''
                    INSERT INTO player_stats (
                        team_id, name, number, games, at_bats, runs, hits,
                        doubles, triples, home_runs, rbi, walks, strikeouts,
                        stolen_bases, caught_stealing, batting_avg, obp, slg, ops, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    stats['team_id'], stats['name'], stats['number'], stats['games'],
                    stats['at_bats'], stats['runs'], stats['hits'], stats['doubles'],
                    stats['triples'], stats['home_runs'], stats['rbi'], stats['walks'],
                    stats['strikeouts'], stats['stolen_bases'], stats['caught_stealing'],
                    stats['batting_avg'], stats['obp'], stats['slg'], stats['ops'],
                    datetime.now().isoformat()
                ))
        
        elif stat_type == 'pitching':
            if existing:
                cur.execute('''
                    UPDATE player_stats SET
                        number = ?, wins = ?, losses = ?, era = ?, games_pitched = ?,
                        games_started = ?, saves = ?, innings_pitched = ?, hits_allowed = ?,
                        runs_allowed = ?, earned_runs = ?, walks_allowed = ?,
                        strikeouts_pitched = ?, whip = ?, updated_at = ?
                    WHERE team_id = ? AND name = ?
                ''', (
                    stats['number'], stats['wins'], stats['losses'], stats['era'],
                    stats['games_pitched'], stats['games_started'], stats['saves'],
                    stats['innings_pitched'], stats['hits_allowed'], stats['runs_allowed'],
                    stats['earned_runs'], stats['walks_allowed'], stats['strikeouts_pitched'],
                    stats['whip'], datetime.now().isoformat(),
                    stats['team_id'], stats['name']
                ))
            else:
                cur.execute('''
                    INSERT INTO player_stats (
                        team_id, name, number, wins, losses, era, games_pitched,
                        games_started, saves, innings_pitched, hits_allowed, runs_allowed,
                        earned_runs, walks_allowed, strikeouts_pitched, whip, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    stats['team_id'], stats['name'], stats['number'], stats['wins'],
                    stats['losses'], stats['era'], stats['games_pitched'],
                    stats['games_started'], stats['saves'], stats['innings_pitched'],
                    stats['hits_allowed'], stats['runs_allowed'], stats['earned_runs'],
                    stats['walks_allowed'], stats['strikeouts_pitched'], stats['whip'],
                    datetime.now().isoformat()
                ))
        
        conn.commit()
        return True
    except Exception as e:
        print(f"  Error updating {stats['name']}: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()


def scrape_team_stats(team_id: str, url: str, dry_run: bool = False) -> dict:
    """
    Scrape batting and pitching stats for a single team.
    
    This function is designed to be called by the cron job which
    handles the actual browser automation.
    
    Returns dict with 'batting' and 'pitching' counts.
    """
    print(f"\n{'='*60}")
    print(f"Scraping: {team_id}")
    print(f"URL: {url}")
    print(f"{'='*60}")
    
    # Note: Browser automation will be handled by the cron job
    # This script provides the parsing logic
    
    return {
        'team_id': team_id,
        'url': url,
        'status': 'pending',
        'message': 'Ready for browser scraping'
    }


def main():
    parser = argparse.ArgumentParser(description='Scrape P4 team player stats')
    parser.add_argument('--conference', choices=['SEC', 'Big Ten', 'ACC', 'Big 12'],
                        help='Scrape only teams from this conference')
    parser.add_argument('--team', help='Scrape a single team by ID')
    parser.add_argument('--limit', type=int, help='Limit number of teams to scrape')
    parser.add_argument('--resume', action='store_true', help='Resume from last progress')
    parser.add_argument('--reset', action='store_true', help='Reset progress and start fresh')
    parser.add_argument('--dry-run', action='store_true', help='Test without database updates')
    parser.add_argument('--list', action='store_true', help='List teams and exit')
    args = parser.parse_args()
    
    # Load team URLs
    team_urls = load_team_urls()
    print(f"Loaded {len(team_urls)} team URLs")
    
    # Filter by conference
    if args.conference:
        team_urls = get_teams_by_conference(team_urls, args.conference)
        print(f"Filtered to {len(team_urls)} {args.conference} teams")
    
    # Single team
    if args.team:
        if args.team not in team_urls:
            print(f"Error: Team '{args.team}' not found")
            print(f"Available: {', '.join(sorted(team_urls.keys())[:10])}...")
            sys.exit(1)
        team_urls = {args.team: team_urls[args.team]}
    
    # List mode
    if args.list:
        print("\nTeam URLs:")
        for team_id, url in sorted(team_urls.items()):
            print(f"  {team_id}: {url}")
        print(f"\nTotal: {len(team_urls)} teams")
        return
    
    # Load/reset progress
    if args.reset:
        progress = {'completed': [], 'last_run': None, 'failed': []}
        save_progress(progress)
        print("Progress reset")
    else:
        progress = load_progress()
        if args.resume and progress['completed']:
            print(f"Resuming from last run: {progress['last_run']}")
            print(f"  Completed: {len(progress['completed'])} teams")
            print(f"  Failed: {len(progress.get('failed', []))} teams")
            team_urls = {k: v for k, v in team_urls.items() 
                        if k not in progress['completed']}
    
    # Apply limit
    if args.limit:
        team_urls = dict(list(team_urls.items())[:args.limit])
    
    if not team_urls:
        print("No teams to scrape")
        return
    
    print(f"\nReady to scrape {len(team_urls)} teams")
    print("This script provides parsing logic. Browser automation is handled by cron job.\n")
    
    # Output team list for cron job
    for team_id, url in team_urls.items():
        info = scrape_team_stats(team_id, url, args.dry_run)
        print(f"  {info['team_id']}: {info['status']}")


if __name__ == '__main__':
    main()
