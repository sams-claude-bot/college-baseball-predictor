#!/usr/bin/env python3
"""
Import box scores from ESPN API.
Works for all D1 teams (not just SEC).

Usage:
    python import_espn_boxscore.py 401847541  # ESPN game ID
    python import_espn_boxscore.py --today     # All completed games today
"""

import sys
import sqlite3
import requests
from datetime import datetime
from pathlib import Path

HEADERS = {'User-Agent': 'Mozilla/5.0'}
DB_PATH = Path(__file__).parent.parent / 'data' / 'baseball.db'

def get_stat_value(stats, stat_name):
    """Extract a stat value from ESPN stat list"""
    for s in stats:
        if s.get('name') == stat_name or s.get('abbreviation') == stat_name:
            return s.get('value', 0)
    return 0

def fetch_espn_boxscore(game_id):
    """Fetch box score data from ESPN"""
    url = f'https://site.api.espn.com/apis/site/v2/sports/baseball/college-baseball/summary?event={game_id}'
    resp = requests.get(url, headers=HEADERS, timeout=30)
    return resp.json()

def import_boxscore(game_id, dry_run=False):
    """Import a single game's box score"""
    print(f"\nFetching ESPN game {game_id}...")
    data = fetch_espn_boxscore(game_id)
    
    # Get team rosters with stats
    rosters = data.get('rosters', [])
    if not rosters:
        print("  No roster data found")
        return False
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.now().isoformat()
    
    updates = 0
    for roster in rosters:
        team_info = roster.get('team', {})
        team_name = team_info.get('displayName', 'Unknown')
        
        # Try to find team_id from our database
        c.execute("SELECT id FROM teams WHERE name = ?", (team_name,))
        row = c.fetchone()
        if not row:
            # Try fuzzy match
            c.execute("SELECT id, name FROM teams WHERE name LIKE ?", (f"%{team_name.split()[0]}%",))
            row = c.fetchone()
        
        if not row:
            print(f"  Team not in DB: {team_name}")
            continue
            
        team_id = row[0]
        print(f"  Processing {team_name} ({team_id})...")
        
        players = roster.get('roster', [])
        for p in players:
            athlete = p.get('athlete', {})
            name = athlete.get('displayName', '')
            stats = p.get('stats', [])
            
            if not stats:
                continue
            
            # Extract batting stats
            ab = get_stat_value(stats, 'atBats')
            h = get_stat_value(stats, 'hits')
            rbi = get_stat_value(stats, 'RBIs')
            hr = get_stat_value(stats, 'homeRuns')
            sb = get_stat_value(stats, 'stolenBases')
            avg = get_stat_value(stats, 'avg')
            
            if ab > 0 or h > 0:
                if not dry_run:
                    c.execute('''
                        UPDATE player_stats 
                        SET at_bats = COALESCE(at_bats, 0) + ?,
                            hits = COALESCE(hits, 0) + ?,
                            rbi = COALESCE(rbi, 0) + ?,
                            home_runs = COALESCE(home_runs, 0) + ?,
                            stolen_bases = COALESCE(stolen_bases, 0) + ?,
                            batting_avg = ?,
                            games = COALESCE(games, 0) + 1,
                            updated_at = ?
                        WHERE team_id = ? AND LOWER(name) = LOWER(?)
                    ''', (ab, h, rbi, hr, sb, avg, now, team_id, name))
                    
                    if c.rowcount > 0:
                        updates += 1
                        print(f"    {name}: {h}-{int(ab)}, {int(rbi)} RBI")
    
    if not dry_run:
        conn.commit()
    conn.close()
    
    print(f"  Updated {updates} player stats")
    return updates > 0

def get_completed_games_today():
    """Get all completed games from ESPN for today"""
    today = datetime.now().strftime('%Y%m%d')
    url = f'https://site.api.espn.com/apis/site/v2/sports/baseball/college-baseball/scoreboard?dates={today}&limit=200'
    resp = requests.get(url, headers=HEADERS, timeout=30)
    data = resp.json()
    
    game_ids = []
    for event in data.get('events', []):
        status = event.get('status', {}).get('type', {}).get('name', '')
        if status == 'STATUS_FINAL':
            game_ids.append(event.get('id'))
    
    return game_ids

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python import_espn_boxscore.py <game_id> | --today")
        sys.exit(1)
    
    if sys.argv[1] == '--today':
        game_ids = get_completed_games_today()
        print(f"Found {len(game_ids)} completed games today")
        for gid in game_ids:
            import_boxscore(gid)
    else:
        import_boxscore(sys.argv[1])
