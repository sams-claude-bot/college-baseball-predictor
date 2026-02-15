#!/usr/bin/env python3
"""
Parse ESPN college baseball scoreboard and update game results.
Designed to work with web_fetch text output.

Usage:
    python3 parse_espn_scoreboard.py          # Fetch and parse today's scores
    python3 parse_espn_scoreboard.py --date 2026-02-13  # Specific date
"""

import re
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.database import get_connection

# Team name normalizations
TEAM_ALIASES = {
    'miss state': 'mississippi-state',
    'mississippi state': 'mississippi-state',
    'ole miss': 'ole-miss',
    'texas a&m': 'texas-am',
    'south carolina': 'south-carolina',
    'nc state': 'nc-state',
    'georgia tech': 'georgia-tech',
    'florida st': 'florida-state',
    'florida state': 'florida-state',
    'oregon st': 'oregon-state',
    'oregon state': 'oregon-state',
    'wake forest': 'wake-forest',
    'virginia tech': 'virginia-tech',
    'east carolina': 'east-carolina',
    'coastal caro': 'coastal-carolina',
    'coastal carolina': 'coastal-carolina',
    'southern miss': 'southern-miss',
    'arizona st': 'arizona-state',
    'arizona state': 'arizona-state',
    'uc san diego': 'uc-san-diego',
    'uc davis': 'uc-davis',
    'uc santa barbara': 'uc-santa-barbara',
    'uc riverside': 'uc-riverside',
    'uc irvine': 'uc-irvine',
    'app state': 'appalachian-state',
    'penn state': 'penn-state',
    'ohio state': 'ohio-state',
    'michigan state': 'michigan-state',
    'ball state': 'ball-state',
    'kent state': 'kent-state',
    'boise state': 'boise-state',
    'fresno state': 'fresno-state',
    'san jose state': 'san-jose-state',
    'san diego state': 'san-diego-state',
    'north carolina a&t': 'north-carolina-at',
    'florida gulf coast': 'florida-gulf-coast',
    'boston college': 'boston-college',
    'unc wilmington': 'unc-wilmington',
    'unc greensboro': 'unc-greensboro',
    'north florida': 'north-florida',
    'south florida': 'south-florida',
    'south alabama': 'south-alabama',
    'north alabama': 'north-alabama',
    'middle tennessee': 'middle-tennessee',
    'tennessee tech': 'tennessee-tech',
    'western kentucky': 'western-kentucky',
    'northern kentucky': 'northern-kentucky',
    'eastern kentucky': 'eastern-kentucky',
    'western michigan': 'western-michigan',
    'central michigan': 'central-michigan',
    'northern illinois': 'northern-illinois',
    'southern indiana': 'southern-indiana',
    'austin peay': 'austin-peay',
    'dallas baptist': 'dallas-baptist',
    'grand canyon': 'grand-canyon',
    'west virginia': 'west-virginia',
    'georgia southern': 'georgia-southern',
    'georgia state': 'georgia-state',
    'louisiana tech': 'louisiana-tech',
    'ul monroe': 'louisiana-monroe',
    'arkansas state': 'arkansas-state',
    'central arkansas': 'central-arkansas',
    'texas state': 'texas-state',
    'texas tech': 'texas-tech',
    'oklahoma state': 'oklahoma-state',
    'new mexico state': 'new-mexico-state',
    'ut rio grande valley': 'utrgv',
    'cal state fullerton': 'cal-state-fullerton',
    'cal state bakersfield': 'cal-state-bakersfield',
    'cal state northridge': 'cal-state-northridge',
    'cal poly': 'cal-poly',
    'long beach state': 'long-beach-state',
    'sacramento state': 'sacramento-state',
    'sam houston': 'sam-houston',
    'ut martin': 'ut-martin',
    'south carolina upstate': 'usc-upstate',
    'florida atlantic': 'florida-atlantic',
    'florida international': 'fiu',
    'st. johns': 'st-johns',
    "st. john's": 'st-johns',
    'mount st. marys': 'mount-st-marys',
    "mount st. mary's": 'mount-st-marys',
    'seattle u': 'seattle',
    'wichita state': 'wichita-state',
    'new orleans': 'new-orleans',
    'old dominion': 'old-dominion',
    'queens university': 'queens-nc',
    'stony brook': 'stony-brook',
    'siu edwardsville': 'siue',
    'missouri state': 'missouri-state',
    'kansas state': 'kansas-state',
    'wright state': 'wright-state',
    'youngstown state': 'youngstown-state',
    'north dakota state': 'north-dakota-state',
    'south dakota state': 'south-dakota-state',
    'bowling green': 'bowling-green',
    'james madison': 'james-madison',
    'west georgia': 'west-georgia',
    'kennesaw state': 'kennesaw-state',
    'jacksonville state': 'jacksonville-state',
    'loyola marymount': 'loyola-marymount',
    'utah tech': 'utah-tech',
    'houston christian': 'houston-christian',
    'california baptist': 'cal-baptist',
    'northern colorado': 'northern-colorado',
    'alabama state': 'alabama-state',
    'norfolk state': 'norfolk-state',
}


def normalize_team_id(name):
    """Convert team name to standardized team_id format"""
    if not name:
        return None
    
    lower = name.lower().strip()
    
    # Check aliases first
    if lower in TEAM_ALIASES:
        return TEAM_ALIASES[lower]
    
    # Clean up: remove parentheses, periods, extra spaces
    result = lower.replace('(', '').replace(')', '').replace('.', '')
    result = re.sub(r'\s+', '-', result.strip())
    return result


def parse_rhe(s):
    """Parse R/H/E from end of string. Returns (runs, hits, errors)."""
    if not s or len(s) < 3:
        return None, None, None
    
    e = int(s[-1])  # Last digit is errors
    rh = s[:-1]     # Rest is runs + hits
    
    if len(rh) == 2:
        r, h = int(rh[0]), int(rh[1])
    elif len(rh) == 3:
        # Try both splits
        r1, h1 = int(rh[0]), int(rh[1:])  # 1+2 split
        r2, h2 = int(rh[:2]), int(rh[2])  # 2+1 split
        
        # Heuristics:
        # 1. If 2-digit runs (10-19) and 1-digit alternative is low, pick 2-digit
        # 2. Can't have 0 hits
        # 3. Runs <= hits is typical
        if r2 >= 10 and r1 <= 2:
            r, h = r2, h2
        elif h2 == 0:
            r, h = r1, h1
        elif r1 <= h1:
            r, h = r1, h1
        else:
            r, h = r2, h2
    elif len(rh) == 4:
        r, h = int(rh[:2]), int(rh[2:])
    else:
        return None, None, None
    
    return r, h, e


def parse_scoreboard_text(text):
    """
    Parse ESPN scoreboard text extracted by web_fetch.
    
    Returns list of dicts: {away_team, away_score, home_team, home_score, tournament}
    """
    games = []
    lines = text.split('\n')
    
    # Pattern for team lines - captures team name and RHE digits at end
    team_pattern = re.compile(
        r'^-?\s*(\d{1,2})?'                    # optional rank
        r'([A-Za-z][A-Za-z0-9 &\'\.\-]+?)'     # team name  
        r'(?:\((\d+-\d+)\))?'                  # optional record
        r'(\d{3,5})$'                          # RHE digits (3-5 total)
    )
    
    i = 0
    current_game = {'away': None, 'home': None, 'tournament': None}
    tournament = None
    
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        
        # Skip empty lines
        if not line:
            continue
            
        # Skip header lines
        if line in ('R', 'H', 'E', 'RHE'):
            continue
            
        # Detect tournament names
        if any(x in line for x in ['Challenge', 'Classic', 'Invitational', 'Showdown', 'Series', 'Desert']):
            tournament = line
            continue
        
        # Skip link format entries: [RHE...](url) - they have messy concatenation
        if line.startswith('[RHE') or line.startswith('['):
            continue
        
        # Try to match team line
        clean_line = line.lstrip('- ')
        match = team_pattern.match(clean_line)
        
        if match:
            rank = int(match.group(1)) if match.group(1) else None
            team_name = match.group(2).strip()
            record = match.group(3)
            rhe_str = match.group(4)
            runs, hits, errors = parse_rhe(rhe_str)
            
            team_data = {
                'name': team_name,
                'team_id': normalize_team_id(team_name),
                'rank': rank,
                'record': record,
                'runs': runs,
                'hits': hits,
                'errors': errors
            }
            
            if current_game['away'] is None:
                current_game['away'] = team_data
                current_game['tournament'] = tournament
            else:
                current_game['home'] = team_data
                games.append(current_game.copy())
                current_game = {'away': None, 'home': None, 'tournament': None}
    
    return games


def update_game_results(games, game_date):
    """Update database with game results"""
    conn = get_connection()
    cur = conn.cursor()
    
    updated = 0
    not_found = []
    
    for game in games:
        away = game['away']
        home = game['home']
        
        if not away or not home:
            continue
        
        # Try to find the game in database
        # Match on date and teams (either direction since we might have them swapped)
        cur.execute('''
            SELECT id, home_team_id, away_team_id FROM games 
            WHERE date = ? AND (
                (home_team_id = ? AND away_team_id = ?)
                OR (home_team_id = ? AND away_team_id = ?)
            )
        ''', (game_date, home['team_id'], away['team_id'], 
              away['team_id'], home['team_id']))
        
        row = cur.fetchone()
        
        if row:
            game_id, db_home, db_away = row
            
            # Determine correct score assignment
            if db_home == home['team_id']:
                home_score, away_score = home['runs'], away['runs']
            else:
                home_score, away_score = away['runs'], home['runs']
            
            cur.execute('''
                UPDATE games 
                SET home_score = ?, away_score = ?, status = 'final'
                WHERE id = ?
            ''', (home_score, away_score, game_id))
            updated += 1
            print(f"  Updated: {away['name']} {away['runs']} @ {home['name']} {home['runs']}")
        else:
            not_found.append(f"{away['name']} @ {home['name']}")
    
    conn.commit()
    conn.close()
    
    print(f"\nUpdated {updated} games")
    if not_found:
        print(f"Not found in DB ({len(not_found)}):")
        for nf in not_found[:10]:
            print(f"  - {nf}")
        if len(not_found) > 10:
            print(f"  ... and {len(not_found) - 10} more")
    
    return updated


def fetch_and_parse(date_str=None):
    """Fetch ESPN scoreboard and parse results"""
    try:
        import requests
    except ImportError:
        print("Error: requests library required")
        return []
    
    # ESPN scoreboard URL - can add date parameter
    url = "https://www.espn.com/college-baseball/scoreboard"
    if date_str:
        # ESPN uses YYYYMMDD format
        date_param = date_str.replace('-', '')
        url = f"{url}?date={date_param}"
    
    print(f"Fetching: {url}")
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    resp = requests.get(url, headers=headers, timeout=30)
    
    if resp.status_code != 200:
        print(f"Error: HTTP {resp.status_code}")
        return []
    
    # For ESPN, we need the raw HTML or use web_fetch
    # For now, just demonstrate with sample parsing
    print(f"Got {len(resp.text)} bytes")
    
    return []


def main():
    parser = argparse.ArgumentParser(description='Parse ESPN scoreboard')
    parser.add_argument('--date', help='Date in YYYY-MM-DD format')
    parser.add_argument('--text', help='Path to saved web_fetch text output')
    parser.add_argument('--dry-run', action='store_true', help='Parse only, no DB update')
    args = parser.parse_args()
    
    game_date = args.date or datetime.now().strftime('%Y-%m-%d')
    
    if args.text:
        # Read from saved file
        with open(args.text) as f:
            text = f.read()
        games = parse_scoreboard_text(text)
    else:
        # For now, we need web_fetch output
        print("Note: For best results, use --text with web_fetch output")
        print("Example: Save output from web_fetch to a file and pass with --text")
        games = fetch_and_parse(game_date)
    
    print(f"Parsed {len(games)} games")
    
    for g in games[:5]:
        away = g['away']
        home = g['home']
        print(f"  {away['name']} {away['runs']} @ {home['name']} {home['runs']}")
    
    if games and not args.dry_run:
        update_game_results(games, game_date)


if __name__ == '__main__':
    main()
