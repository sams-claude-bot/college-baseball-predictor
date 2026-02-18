#!/usr/bin/env python3
"""
ESPN Box Score Parser

Extracts player stats from ESPN's college baseball API.
Not all games have detailed stats - ESPN covers ~20% of D1 games.

Usage:
    python espn_boxscore.py 401852666          # Single game
    python espn_boxscore.py --date 2026-02-13  # All games on date
"""

import sys
import sqlite3
import requests
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / 'data' / 'baseball.db'
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}


def get_espn_scoreboard(date_str):
    """Get all games for a date from ESPN scoreboard API"""
    date_compact = date_str.replace('-', '')
    url = f'https://site.api.espn.com/apis/site/v2/sports/baseball/college-baseball/scoreboard?dates={date_compact}&limit=200'
    
    resp = requests.get(url, headers=HEADERS, timeout=30)
    data = resp.json()
    
    games = []
    for event in data.get('events', []):
        status = event.get('status', {}).get('type', {}).get('name', '')
        if status == 'STATUS_FINAL':
            games.append({
                'espn_id': event.get('id'),
                'name': event.get('shortName', ''),
                'date': date_str
            })
    return games


def get_espn_boxscore(espn_game_id):
    """Fetch detailed box score from ESPN summary API"""
    url = f'https://site.api.espn.com/apis/site/v2/sports/baseball/college-baseball/summary?event={espn_game_id}'
    
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        return resp.json()
    except:
        return None


def parse_boxscore(data):
    """Parse ESPN boxscore data into standardized format"""
    result = {
        'home_team': None,
        'away_team': None,
        'home_batting': [],
        'away_batting': [],
        'home_pitching': [],
        'away_pitching': [],
    }
    
    boxscore = data.get('boxscore', {})
    players = boxscore.get('players', [])
    
    if not players:
        return None
    
    # Get home/away from header if available
    header = data.get('header', {})
    competitions = header.get('competitions', [{}])
    comp = competitions[0] if competitions else {}
    competitors = comp.get('competitors', [])
    
    home_team_name = None
    away_team_name = None
    for c in competitors:
        if c.get('homeAway') == 'home':
            home_team_name = c.get('team', {}).get('displayName')
        else:
            away_team_name = c.get('team', {}).get('displayName')
    
    for i, team_data in enumerate(players):
        team_info = team_data.get('team', {})
        team_name = team_info.get('displayName', '')
        
        # Determine if home or away by team name or index (away is usually first)
        is_home = (team_name == home_team_name) if home_team_name else (i == 1)
        
        # Determine team keys
        team_key_bat = 'home_batting' if is_home else 'away_batting'
        team_key_pitch = 'home_pitching' if is_home else 'away_pitching'
        
        if is_home:
            result['home_team'] = team_name
        else:
            result['away_team'] = team_name
        
        # Process statistics categories
        for stat_cat in team_data.get('statistics', []):
            athletes = stat_cat.get('athletes', [])
            
            if not athletes:
                continue
            
            # Detect category by first stat format
            first_stat = athletes[0].get('stats', [])
            if not first_stat:
                continue
            
            # If first stat contains "-" like "1-4", it's batting (H-AB format)
            # If first stat contains "." like "5.0", it's pitching (IP)
            first_val = str(first_stat[0])
            is_batting = '-' in first_val and '.' not in first_val
            is_pitching = '.' in first_val or (first_val.isdigit() and len(first_stat) >= 6)
            
            for athlete_data in athletes:
                athlete = athlete_data.get('athlete', {})
                name = athlete.get('displayName', '')
                
                if not name or name.lower() == 'totals':
                    continue
                
                stats = athlete_data.get('stats', [])
                
                if is_batting:
                    batting = parse_batting_stats(name, stats)
                    if batting:
                        result[team_key_bat].append(batting)
                
                elif is_pitching:
                    pitching = parse_pitching_stats(name, stats)
                    if pitching:
                        result[team_key_pitch].append(pitching)
    
    # Check if we got meaningful data
    if not result['home_batting'] and not result['away_batting']:
        return None
    
    return result


def parse_batting_stats(name, stats):
    """Parse batting stats array"""
    try:
        # ESPN format varies:
        # Option 1: [H-AB, AB, R, RBI, BB, SO] where H-AB is like "1-4"
        # Option 2: [AB, R, H, RBI, BB, SO]
        if len(stats) < 5:
            return None
        
        first = str(stats[0])
        
        if '-' in first:
            # Format: H-AB (like "1-4" means 1 hit in 4 at-bats)
            parts = first.split('-')
            hits = safe_int(parts[0])
            ab = safe_int(parts[1])
            # Remaining: [AB, R, RBI, BB, SO, ...]
            return {
                'name': name,
                'ab': ab,
                'runs': safe_int(stats[1]),  # Position column
                'hits': hits,
                'rbi': safe_int(stats[2]),
                'bb': safe_int(stats[3]),
                'so': safe_int(stats[4]) if len(stats) > 4 else 0,
            }
        else:
            # Standard format: [AB, R, H, RBI, BB, SO]
            return {
                'name': name,
                'ab': safe_int(stats[0]),
                'runs': safe_int(stats[1]),
                'hits': safe_int(stats[2]),
                'rbi': safe_int(stats[3]),
                'bb': safe_int(stats[4]),
                'so': safe_int(stats[5]) if len(stats) > 5 else 0,
            }
    except:
        return None


def parse_pitching_stats(name, stats):
    """Parse pitching stats array"""
    try:
        # ESPN format: [IP, H, R, ER, BB, SO, HR, ERA] (varies)
        if len(stats) < 6:
            return None
        
        return {
            'name': name,
            'ip': parse_innings(stats[0]),
            'hits': safe_int(stats[1]),
            'runs': safe_int(stats[2]),
            'er': safe_int(stats[3]),
            'bb': safe_int(stats[4]),
            'so': safe_int(stats[5]),
        }
    except:
        return None


def safe_int(val):
    """Safely convert to int"""
    try:
        if isinstance(val, str):
            val = val.replace('-', '0')
        return int(float(val))
    except:
        return 0


def parse_innings(ip_str):
    """Parse innings pitched (6.2 = 6 2/3 innings)"""
    try:
        ip_str = str(ip_str).strip()
        if '.' in ip_str:
            whole, frac = ip_str.split('.')
            thirds = int(frac) if frac else 0
            return float(whole) + (thirds / 3.0)
        return float(ip_str)
    except:
        return 0.0


def normalize_team_id(team_name):
    """Convert ESPN team name to our database team_id"""
    if not team_name:
        return None
    
    # Common mappings
    mappings = {
        'Mississippi State Bulldogs': 'mississippi-state',
        'Ole Miss Rebels': 'ole-miss',
        'Texas A&M Aggies': 'texas-am',
        'South Carolina Gamecocks': 'south-carolina',
        'Georgia Tech Yellow Jackets': 'georgia-tech',
        'Florida State Seminoles': 'florida-state',
        'NC State Wolfpack': 'nc-state',
        'Boston College Eagles': 'boston-college',
        'Miami Hurricanes': 'miami-fl',
        'Virginia Tech Hokies': 'virginia-tech',
        'Wake Forest Demon Deacons': 'wake-forest',
        'Oklahoma State Cowboys': 'oklahoma-state',
        'Kansas State Wildcats': 'kansas-state',
        'West Virginia Mountaineers': 'west-virginia',
        'Texas Tech Red Raiders': 'texas-tech',
        'Arizona State Sun Devils': 'arizona-state',
        'Oregon State Beavers': 'oregon-state',
        'Penn State Nittany Lions': 'penn-state',
        'Ohio State Buckeyes': 'ohio-state',
        'Michigan State Spartans': 'michigan-state',
        'North Carolina Tar Heels': 'north-carolina',
    }
    
    for key, val in mappings.items():
        if key in team_name or team_name in key:
            return val
    
    # Default: extract first word, lowercase, hyphenate
    return team_name.split()[0].lower() if team_name else None


def update_player_stats(team_id, batting, pitching, date_str):
    """Update player stats in database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    updated = 0
    
    for batter in batting:
        name = batter['name']
        # Try exact match then fuzzy
        c.execute("""
            SELECT id FROM player_stats 
            WHERE team_id = ? AND (name = ? OR name LIKE ?)
        """, (team_id, name, f"%{name.split()[-1]}%"))
        
        row = c.fetchone()
        if row:
            c.execute("""
                UPDATE player_stats SET
                    games = games + 1,
                    at_bats = at_bats + ?,
                    runs = runs + ?,
                    hits = hits + ?,
                    rbi = rbi + ?,
                    walks = walks + ?,
                    strikeouts = strikeouts + ?,
                    updated_at = ?
                WHERE id = ?
            """, (batter['ab'], batter['runs'], batter['hits'],
                  batter['rbi'], batter['bb'], batter['so'],
                  datetime.now().isoformat(), row[0]))
            updated += 1
    
    for pitcher in pitching:
        name = pitcher['name']
        c.execute("""
            SELECT id FROM player_stats 
            WHERE team_id = ? AND (name = ? OR name LIKE ?)
        """, (team_id, name, f"%{name.split()[-1]}%"))
        
        row = c.fetchone()
        if row:
            c.execute("""
                UPDATE player_stats SET
                    games_pitched = games_pitched + 1,
                    innings_pitched = innings_pitched + ?,
                    hits_allowed = hits_allowed + ?,
                    runs_allowed = runs_allowed + ?,
                    earned_runs = earned_runs + ?,
                    walks_allowed = walks_allowed + ?,
                    strikeouts_pitched = strikeouts_pitched + ?,
                    updated_at = ?
                WHERE id = ?
            """, (pitcher['ip'], pitcher['hits'], pitcher['runs'],
                  pitcher['er'], pitcher['bb'], pitcher['so'],
                  datetime.now().isoformat(), row[0]))
            updated += 1
    
    conn.commit()
    conn.close()
    return updated


def collect_espn_boxscore(espn_game_id, dry_run=False):
    """Collect and store box score for a single ESPN game"""
    print(f"\n📊 Fetching ESPN game {espn_game_id}")
    
    data = get_espn_boxscore(espn_game_id)
    if not data:
        print("   ⊘ Failed to fetch data")
        return False
    
    boxscore = parse_boxscore(data)
    if not boxscore:
        print("   ⊘ No detailed stats available")
        return False
    
    home = boxscore['home_team']
    away = boxscore['away_team']
    print(f"   {away} @ {home}")
    print(f"   ✓ {len(boxscore['home_batting'])} home batters, {len(boxscore['away_batting'])} away batters")
    print(f"     {len(boxscore['home_pitching'])} home pitchers, {len(boxscore['away_pitching'])} away pitchers")
    
    if dry_run:
        print("   [DRY RUN]")
        return True
    
    # Update database
    total_updated = 0
    
    home_id = normalize_team_id(home)
    away_id = normalize_team_id(away)
    
    if home_id:
        updated = update_player_stats(home_id, boxscore['home_batting'], boxscore['home_pitching'], '')
        total_updated += updated
    
    if away_id:
        updated = update_player_stats(away_id, boxscore['away_batting'], boxscore['away_pitching'], '')
        total_updated += updated
    
    print(f"   ✓ Updated {total_updated} player records")
    return True


def collect_date(date_str, dry_run=False):
    """Collect all ESPN box scores for a date"""
    print(f"\n{'='*60}")
    print(f"Collecting ESPN box scores for {date_str}")
    print(f"{'='*60}")
    
    games = get_espn_scoreboard(date_str)
    print(f"Found {len(games)} completed games")
    
    collected = 0
    no_data = 0
    
    for game in games:
        result = collect_espn_boxscore(game['espn_id'], dry_run)
        if result:
            collected += 1
        else:
            no_data += 1
    
    print(f"\n{'='*60}")
    print(f"ESPN Collection Complete:")
    print(f"  With stats: {collected}")
    print(f"  No data:    {no_data}")
    print(f"{'='*60}")
    
    return {'collected': collected, 'no_data': no_data}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python espn_boxscore.py GAME_ID")
        print("  python espn_boxscore.py --date 2026-02-13")
        sys.exit(1)
    
    dry_run = '--dry' in sys.argv
    
    if sys.argv[1] == '--date':
        date_str = sys.argv[2] if len(sys.argv) > 2 else datetime.now().strftime('%Y-%m-%d')
        collect_date(date_str, dry_run)
    else:
        collect_espn_boxscore(sys.argv[1], dry_run)
