#!/usr/bin/env python3
"""
Fetch schedules for Tier 2 conference teams from ESPN.

This script fetches the full 2026 schedule for each team in the specified
Tier 2 conferences and adds any missing games to the database.

Usage:
    python3 scripts/fetch_tier2_schedules.py --conference "Sun Belt"
    python3 scripts/fetch_tier2_schedules.py --all
    python3 scripts/fetch_tier2_schedules.py --dry-run
"""

import argparse
import json
import re
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError

BASE_DIR = Path(__file__).parent.parent
DB_PATH = BASE_DIR / "data" / "baseball.db"
ESPN_TEAMS_PATH = BASE_DIR / "data" / "espn_team_ids.json"
PROGRESS_FILE = BASE_DIR / "data" / "tier2_schedule_progress.json"

TEAM_SCHEDULE_URL = "https://site.api.espn.com/apis/site/v2/sports/baseball/college-baseball/teams/{espn_id}/schedule"

TIER2_CONFERENCES = [
    "Sun Belt", "AAC", "A-10", "CAA", "WCC", "MVC", "Big East", "C-USA", "ASUN"
]


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def api_get(url):
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except URLError as e:
        print(f"    API error: {e}")
        return None


def load_espn_id_map():
    """Returns {our_team_id: espn_id_str}"""
    if ESPN_TEAMS_PATH.exists():
        with open(ESPN_TEAMS_PATH) as f:
            return json.load(f)
    return {}


def espn_date_to_local(date_str):
    """Convert ESPN UTC date to local date string."""
    if not date_str:
        return None
    try:
        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d')
    except:
        return None


# Cache for team lookups
_team_cache = {}

# Manual overrides for ESPN names that don't match our IDs
ESPN_NAME_OVERRIDES = {
    "ole miss rebels": "ole-miss",
    "texas a&m aggies": "texas-am",
    "miami hurricanes": "miami-fl",
    "ucf knights": "ucf",
    "lsu tigers": "lsu",
    "usc trojans": "usc",
    "ucla bruins": "ucla",
    "smu mustangs": "smu",
    "tcu horned frogs": "tcu",
    "byu cougars": "byu",
    "uconn huskies": "uconn",
    "umass minutemen": "umass",
    "unc wilmington seahawks": "uncw",
    "app state mountaineers": "appalachian-state",
    "ualbany great danes": "albany",
    "fiu panthers": "fiu",
    "ut rio grande valley vaqueros": "utrgv",
    "uc santa barbara gauchos": "uc-santa-barbara",
    "tarleton state texans": "tarleton-state",
    "stephen f. austin lumberjacks": "stephen-f-austin",
    "texas a&m-corpus christi islanders": "texas-aandm-corpus-christi",
    "lmu lions": "loyola-marymount",
    "saint mary's gaels": "saint-marys",
    "st. john's red storm": "st-johns",
    "queens royals": "queens-university",
    "hawaii rainbow warriors": "hawaii",
    "sfa lumberjacks": "stephen-f-austin",
    "little rock trojans": "little-rock",
    "seattle u redhawks": "seattle",
    "ualr trojans": "little-rock",
}

def load_team_cache():
    """Load all team IDs and names into cache for matching."""
    global _team_cache
    if _team_cache:
        return _team_cache
    
    conn = get_conn()
    teams = conn.execute("SELECT id, name, nickname FROM teams").fetchall()
    conn.close()
    
    for t in teams:
        tid = t['id'].lower()
        name = t['name'].lower() if t['name'] else ''
        nickname = t['nickname'].lower() if t['nickname'] else ''
        
        _team_cache[tid] = t['id']
        if name:
            _team_cache[name] = t['id']
            # Also add without common words
            simple = re.sub(r'\s+(state|university|college)\s*$', '', name, flags=re.I)
            _team_cache[simple] = t['id']
    
    return _team_cache


def normalize_team_id(name):
    """Convert a display name to our team_id format, matching existing teams."""
    if not name:
        return None
    
    name_lower = name.lower().strip()
    
    # Check manual overrides first
    if name_lower in ESPN_NAME_OVERRIDES:
        return ESPN_NAME_OVERRIDES[name_lower]
    
    # Load team cache
    cache = load_team_cache()
    
    # Direct match on name
    if name_lower in cache:
        return cache[name_lower]
    
    # Try without mascot
    # Common mascots to remove
    mascots = [
        'aggies', 'anteaters', 'aztecs', 'badgers', 'bears', 'beavers', 'bearcats',
        'bengals', 'bison', 'blazers', 'blue devils', 'bobcats', 'boilermakers',
        'braves', 'broncos', 'bruins', 'buccaneers', 'buckeyes', 'buffaloes', 
        'bulldogs', 'bulls', 'cardinals', 'catamounts', 'cavaliers', 'chanticleers',
        'chippewas', 'colonels', 'commodores', 'cornhuskers', 'cougars', 'cowboys',
        'crimson tide', 'crusaders', 'cyclones', 'deacons', 'demon deacons',
        'dirtbags', 'dolphins', 'dons', 'ducks', 'dukes', 'eagles', 'explorers',
        'falcons', 'fighting camels', 'fighting illini', 'fighting irish', 'flames',
        'flyers', 'friars', 'gaels', 'gamecocks', 'gators', 'golden bears',
        'golden eagles', 'golden flashes', 'governors', 'great danes', 'green wave',
        'grizzlies', 'hawks', 'hilltoppers', 'hokies', 'hoosiers', 'hornets',
        'hoyas', 'huskies', 'hurricanes', 'islanders', 'jaguars', 'jaspers',
        'jayhawks', 'knights', 'lancers', 'leathernecks', 'lions', 'lobos',
        'longhorns', 'lumberjacks', 'matadors', 'mavericks', 'midshipmen', 'miners',
        'minutemen', 'monarchs', 'mountaineers', 'musketeers', 'mustangs',
        'nittany lions', 'norse', 'orange', 'orangemen', 'owls', 'paladins',
        'panthers', 'patriots', 'peacocks', 'penguins', 'phoenix', 'pilots',
        'pioneers', 'pirates', 'privateers', 'purple aces', 'purple eagles',
        'quakers', 'raiders', 'ragin cajuns', 'rainbow warriors', 'rams', 'rattlers',
        'razorbacks', 'rebels', 'red raiders', 'red storm', 'red wolves', 'redhawks',
        'retrievers', 'revolutionaries', 'roadrunners', 'rockets', 'royals',
        'running rebels', 'salukis', 'scarlet knights', 'seahawks', 'seawolves',
        'seminoles', 'shockers', 'skyhawks', 'sooners', 'spartans', 'spiders',
        'stags', 'sun devils', 'tar heels', 'terrapins', 'terriers', 'texans',
        'thundering herd', 'tigers', 'titans', 'toreros', 'trojans', 'tribe',
        'tritons', 'utes', 'vaqueros', 'vikings', 'volunteers', 'warhawks',
        'warriors', 'waves', 'wildcats', 'wolfpack', 'wolverines', 'wolves',
        'yellow jackets', 'zips', '49ers', 'bonnies', 'demons', 'golden', 'purple'
    ]
    
    for mascot in sorted(mascots, key=len, reverse=True):
        if name_lower.endswith(' ' + mascot):
            school = name_lower[:-len(mascot)-1].strip()
            if school in cache:
                return cache[school]
            # Try converting to hyphenated ID
            school_id = re.sub(r'[^\w\s]', '', school)
            school_id = re.sub(r'\s+', '-', school_id).strip('-')
            if school_id in cache:
                return cache[school_id]
    
    # Last resort: convert to ID format and hope for match
    team_id = re.sub(r'[^\w\s]', '', name_lower)
    team_id = re.sub(r'\s+', '-', team_id).strip('-')
    
    # Remove mascot suffixes from ID
    for mascot in sorted(mascots, key=len, reverse=True):
        mascot_id = mascot.replace(' ', '-')
        if team_id.endswith('-' + mascot_id):
            team_id = team_id[:-len(mascot_id)-1]
            break
    
    if team_id in cache:
        return cache[team_id]
    
    # Return the generated ID even if not found (will create new team)
    return team_id


def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"processed_teams": [], "games_added": 0}


def save_progress(progress):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def get_tier2_teams(conference=None):
    """Get all teams from Tier 2 conferences."""
    conn = get_conn()
    
    if conference:
        teams = conn.execute("""
            SELECT id, name, conference FROM teams 
            WHERE conference = ?
            ORDER BY name
        """, (conference,)).fetchall()
    else:
        placeholders = ','.join('?' * len(TIER2_CONFERENCES))
        teams = conn.execute(f"""
            SELECT id, name, conference FROM teams 
            WHERE conference IN ({placeholders})
            ORDER BY conference, name
        """, TIER2_CONFERENCES).fetchall()
    
    conn.close()
    return teams


def ensure_team_exists(conn, team_id, name=None):
    """Make sure a team exists in the database."""
    existing = conn.execute(
        "SELECT id FROM teams WHERE id = ?", (team_id,)
    ).fetchone()
    
    if existing:
        return True
    
    # Create the team
    conn.execute(
        "INSERT INTO teams (id, name, conference) VALUES (?, ?, ?)",
        (team_id, name or team_id.replace('-', ' ').title(), "Unknown")
    )
    print(f"      Created new team: {team_id}")
    return True


def game_exists(conn, game_id):
    """Check if a game already exists."""
    result = conn.execute(
        "SELECT id FROM games WHERE id = ?", (game_id,)
    ).fetchone()
    return result is not None


def add_game(conn, game_id, date, home_team, away_team, home_score=None, away_score=None, status='scheduled'):
    """Add a game to the database."""
    # Ensure both teams exist
    ensure_team_exists(conn, home_team)
    ensure_team_exists(conn, away_team)
    
    conn.execute("""
        INSERT INTO games (id, date, home_team_id, away_team_id, home_score, away_score, status)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (game_id, date, home_team, away_team, home_score, away_score, status))
    return True


def fetch_team_schedule(espn_id, team_id):
    """Fetch and process schedule for a single team."""
    url = TEAM_SCHEDULE_URL.format(espn_id=espn_id)
    data = api_get(url)
    
    if not data:
        return []
    
    games = []
    events = data.get('events', [])
    
    for event in events:
        date_str = event.get('date')
        date = espn_date_to_local(date_str)
        if not date:
            continue
        
        # Get competitors
        competitions = event.get('competitions', [])
        if not competitions:
            continue
        
        comp = competitions[0]
        competitors = comp.get('competitors', [])
        if len(competitors) != 2:
            continue
        
        home_team = None
        away_team = None
        home_score = None
        away_score = None
        
        for c in competitors:
            team_info = c.get('team', {})
            team_name = team_info.get('displayName', '')
            is_home = c.get('homeAway') == 'home'
            
            # Get score
            score = c.get('score')
            if isinstance(score, dict):
                score = score.get('displayValue')
            if score:
                try:
                    score = int(score)
                except:
                    score = None
            
            # Normalize team ID
            tid = normalize_team_id(team_name)
            
            if is_home:
                home_team = tid
                home_score = score
            else:
                away_team = tid
                away_score = score
        
        if home_team and away_team:
            game_id = f"{date}-{home_team}-vs-{away_team}"
            status = 'completed' if home_score is not None else 'scheduled'
            
            games.append({
                'id': game_id,
                'date': date,
                'home_team': home_team,
                'away_team': away_team,
                'home_score': home_score,
                'away_score': away_score,
                'status': status
            })
    
    return games


def process_conference(conference, dry_run=False, resume=True):
    """Process all teams in a conference."""
    print(f"\n{'='*60}")
    print(f"Processing {conference} Conference")
    print(f"{'='*60}")
    
    teams = get_tier2_teams(conference)
    espn_ids = load_espn_id_map()
    progress = load_progress() if resume else {"processed_teams": [], "games_added": 0}
    
    conn = get_conn()
    total_added = 0
    
    for team in teams:
        team_id = team['id']
        team_name = team['name']
        
        if team_id in progress['processed_teams'] and resume:
            print(f"  · {team_name}: already processed (skipping)")
            continue
        
        # Find ESPN ID
        espn_id = espn_ids.get(team_id)
        if not espn_id:
            print(f"  ⚠ {team_name}: no ESPN ID found")
            continue
        
        print(f"  → {team_name} (ESPN ID: {espn_id})")
        
        # Fetch schedule
        games = fetch_team_schedule(espn_id, team_id)
        print(f"    Found {len(games)} games")
        
        added = 0
        for game in games:
            if not game_exists(conn, game['id']):
                if not dry_run:
                    add_game(conn, game['id'], game['date'], game['home_team'], 
                            game['away_team'], game['home_score'], game['away_score'],
                            game['status'])
                added += 1
        
        if added > 0:
            print(f"    Added {added} new games")
            total_added += added
            if not dry_run:
                conn.commit()
        
        # Update progress
        progress['processed_teams'].append(team_id)
        progress['games_added'] += added
        if not dry_run:
            save_progress(progress)
        
        # Rate limiting - 5 second delay between teams
        time.sleep(5)
    
    conn.close()
    print(f"\n{conference}: Added {total_added} new games total")
    return total_added


def main():
    parser = argparse.ArgumentParser(description="Fetch Tier 2 conference schedules from ESPN")
    parser.add_argument('--conference', type=str, help='Specific conference to process')
    parser.add_argument('--all', action='store_true', help='Process all Tier 2 conferences')
    parser.add_argument('--dry-run', action='store_true', help='Preview without writing')
    parser.add_argument('--no-resume', action='store_true', help='Start fresh, ignore progress')
    args = parser.parse_args()
    
    if args.all:
        total = 0
        for conf in TIER2_CONFERENCES:
            total += process_conference(conf, args.dry_run, not args.no_resume)
        print(f"\n{'='*60}")
        print(f"TOTAL: Added {total} new games across all Tier 2 conferences")
    elif args.conference:
        process_conference(args.conference, args.dry_run, not args.no_resume)
    else:
        print("Usage: specify --conference 'Name' or --all")
        print(f"\nAvailable conferences: {', '.join(TIER2_CONFERENCES)}")


if __name__ == '__main__':
    main()
