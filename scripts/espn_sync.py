#!/usr/bin/env python3
"""
ESPN Sync — Master script for syncing all D1 baseball data from ESPN.

Usage:
    python3 scripts/espn_sync.py --today
    python3 scripts/espn_sync.py --yesterday
    python3 scripts/espn_sync.py --date 2026-02-15
    python3 scripts/espn_sync.py --range 2026-02-13 2026-02-15
    python3 scripts/espn_sync.py --sync-teams          # Fetch all D1 teams from ESPN
"""

import argparse
import json
import re
import sys
import time
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError

BASE_DIR = Path(__file__).parent.parent
DB_PATH = BASE_DIR / "data" / "baseball.db"
ESPN_TEAMS_PATH = BASE_DIR / "data" / "espn_team_ids.json"

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/baseball/college-baseball/scoreboard"
TEAMS_URL = "https://site.api.espn.com/apis/site/v2/sports/baseball/college-baseball/teams"

# ESPN displayName -> our team_id overrides (for tricky names)
NAME_OVERRIDES = {
    "Ole Miss Rebels": "ole-miss",
    "Texas A&M Aggies": "texas-am",
    "Miami Hurricanes": "miami-fl",
    "UCF Knights": "ucf",
    "LSU Tigers": "lsu",
    "USC Trojans": "usc",
    "UCLA Bruins": "ucla",
    "SMU Mustangs": "smu",
    "TCU Horned Frogs": "tcu",
    "BYU Cougars": "byu",
    "UNLV Rebels": "unlv",
    "UTSA Roadrunners": "utsa",
    "UTEP Miners": "utep",
    "UAB Blazers": "uab",
    "UMass Minutemen": "umass",
    "UConn Huskies": "uconn",
    "VCU Rams": "vcu",
    "FIU Panthers": "fiu",
    "LIU Sharks": "liu",
    "SIUE Cougars": "siue",
    "SIU Salukis": "southern-illinois",
    "UIC Flames": "uic",
    "NJIT Highlanders": "njit",
    "IUPUI Jaguars": "iupui",
    "UT Arlington Mavericks": "ut-arlington",
    "UT Rio Grande Valley Vaqueros": "utrgv",
    "UNC Wilmington Seahawks": "unc-wilmington",
    "UNC Greensboro Spartans": "unc-greensboro",
    "UNC Asheville Bulldogs": "unc-asheville",
    "USC Upstate Spartans": "usc-upstate",
    "Cal State Fullerton Titans": "cal-state-fullerton",
    "Cal State Bakersfield Roadrunners": "cal-state-bakersfield",
    "Cal State Northridge Matadors": "cal-state-northridge",
    "Cal Poly Mustangs": "cal-poly",
    "Milwaukee Panthers": "milwaukee",
    "UMBC Retrievers": "umbc",
    "UAlbany Great Danes": "albany",
    "LMU Lions": "loyola-marymount",
    "Saint Mary's Gaels": "saint-marys",
    "St. John's Red Storm": "st-johns",
    "Mount St. Mary's Mountaineers": "mount-st-marys",
    "Queens Royals": "queens-nc",
    "Hawai'i Rainbow Warriors": "hawaii",
    "SFA Lumberjacks": "stephen-f-austin",
    "Omaha Mavericks": "nebraska-omaha",
    "Little Rock Trojans": "little-rock",
    "Tarleton State Texans": "tarleton-state",
    "Seattle U Redhawks": "seattle",
    "Purdue Fort Wayne Mastodons": "purdue-fort-wayne",
    "Southern University Jaguars": "southern",
    "Grambling Tigers": "grambling",
    "Prairie View A&M Panthers": "prairie-view",
    "Texas Southern Tigers": "texas-southern",
    "Alabama A&M Bulldogs": "alabama-am",
    "Bethune-Cookman Wildcats": "bethune-cookman",
    "Alcorn State Braves": "alcorn-state",
    "North Carolina A&T Aggies": "north-carolina-at",
    "Florida A&M Rattlers": "florida-am",
    "Coppin State Eagles": "coppin-state",
    "Maryland-Eastern Shore Hawks": "maryland-eastern-shore",
    "Delaware State Hornets": "delaware-state",
    "Mississippi Valley State Delta Devils": "mississippi-valley-state",
    "Arkansas-Pine Bluff Golden Lions": "arkansas-pine-bluff",
    "Jackson State Tigers": "jackson-state",
    "Central Connecticut Blue Devils": "central-connecticut",
    "Le Moyne Dolphins": "le-moyne",
    "Stonehill Skyhawks": "stonehill",
    "Lindenwood Lions": "lindenwood",
    "Southern Indiana Screaming Eagles": "southern-indiana",
    "West Georgia Wolves": "west-georgia",
}

# Reverse lookup: ESPN ID -> our team_id (built at runtime)
_espn_id_cache = {}


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def api_get(url):
    """Fetch JSON from ESPN API with rate limiting."""
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except URLError as e:
        print(f"  API error: {e}")
        return None


def espn_display_to_slug(display_name):
    """Convert ESPN displayName to our slug-style team_id."""
    if display_name in NAME_OVERRIDES:
        return NAME_OVERRIDES[display_name]
    
    # Remove mascot: "Mississippi State Bulldogs" -> "Mississippi State"
    # Strategy: ESPN displayName = location + " " + name (mascot)
    # We want the location part, slugified
    # But some are tricky — "North Carolina Tar Heels" should be "north-carolina"
    # For now, strip common suffixes and slugify
    
    # Use everything before the last word if it looks like a mascot
    # But handle multi-word mascots: "Tar Heels", "Blue Devils", "Red Storm", etc.
    location = display_name
    
    # Common multi-word mascots to strip
    multi_mascots = [
        # Two-word mascots (alphabetical for maintainability)
        "Black Bears", "Black Knights", "Blue Demons", "Blue Devils",
        "Blue Hens", "Blue Hose", "Blue Jays", "Blue Raiders",
        "Crimson Tide", "Delta Devils", "Demon Deacons",
        "Fighting Camels", "Fighting Hawks", "Fighting Illini", "Fighting Irish",
        "Golden Eagles", "Golden Flashes", "Golden Gophers", "Golden Griffins",
        "Golden Grizzlies", "Golden Hurricane", "Golden Lions",
        "Green Wave", "Horned Frogs",
        "Mean Green", "Mountain Hawks",
        "Nittany Lions",
        "Purple Eagles",
        "Ragin' Cajuns", "Rainbow Warriors", "Red Flash", "Red Foxes",
        "Red Raiders", "Red Storm", "River Hawks", "Runnin' Bulldogs",
        "Scarlet Knights", "Screaming Eagles", "Sun Devils",
        "Tar Heels", "Thundering Herd", "Wolf Pack",
        "Yellow Jackets",
        # Single-word but commonly confused
        "Aztecs", "Beach", "Blazers", "Boilermakers", "Buckeyes",
        "Cornhuskers", "Hawkeyes", "Hilltoppers", "Hoosiers",
        "Leathernecks", "Redbirds", "Saluki", "Spartans",
        "Volunteers", "Wolverines", "Wildcats", "Badgers", "49ers",
    ]
    
    for mascot in multi_mascots:
        if display_name.endswith(mascot):
            location = display_name[:-len(mascot)].strip()
            break
    else:
        # Single-word mascot: strip last word
        parts = display_name.rsplit(" ", 1)
        if len(parts) == 2:
            location = parts[0]
    
    # Slugify
    slug = location.lower().strip()
    slug = slug.replace("&", "and").replace("'", "").replace(".", "")
    slug = re.sub(r'[^a-z0-9\s-]', '', slug)
    slug = re.sub(r'\s+', '-', slug.strip())
    slug = re.sub(r'-+', '-', slug)
    return slug


def load_espn_id_map():
    """Load ESPN ID -> our team_id mapping."""
    global _espn_id_cache
    if ESPN_TEAMS_PATH.exists():
        with open(ESPN_TEAMS_PATH) as f:
            data = json.load(f)
        # data is {our_id: espn_id_str}
        _espn_id_cache = {v: k for k, v in data.items()}
    return _espn_id_cache


def save_espn_id_map(mapping):
    """Save our_id -> espn_id mapping."""
    with open(ESPN_TEAMS_PATH, 'w') as f:
        json.dump(mapping, f, indent=2, sort_keys=True)


def resolve_team(espn_id, display_name, abbreviation, conn):
    """
    Resolve an ESPN team to our team_id. Creates team if needed.
    Returns team_id.
    """
    espn_id_str = str(espn_id)
    
    # Check cache first
    if espn_id_str in _espn_id_cache:
        return _espn_id_cache[espn_id_str]
    
    # Check if we have this team by ESPN ID in our mapping file
    load_espn_id_map()
    if espn_id_str in _espn_id_cache:
        return _espn_id_cache[espn_id_str]
    
    # Generate slug from display name
    team_id = espn_display_to_slug(display_name)
    
    # Check if this team_id already exists in DB
    cur = conn.cursor()
    cur.execute("SELECT id FROM teams WHERE id = ?", (team_id,))
    exists = cur.fetchone()
    
    if not exists:
        # Create the team
        # Extract location (everything before mascot)
        name = display_name.rsplit(" ", 1)[0] if " " in display_name else display_name
        for mascot in ["Nittany Lions", "Horned Frogs", "Golden Eagles", "Blue Devils",
                       "Demon Deacons", "Yellow Jackets", "Crimson Tide", "Fighting Irish",
                       "Tar Heels", "Red Raiders", "Sun Devils", "Red Storm", "Screaming Eagles",
                       "Golden Lions", "Delta Devils", "Rainbow Warriors", "Ragin' Cajuns",
                       "Golden Flashes", "Golden Gophers", "Scarlet Knights", "River Hawks"]:
            if display_name.endswith(mascot):
                name = display_name[:-len(mascot)].strip()
                break
        
        nickname = display_name.replace(name, "").strip() if name != display_name else None
        
        cur.execute('''
            INSERT INTO teams (id, name, nickname, created_at, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        ''', (team_id, name, nickname))
        
        # Initialize Elo at 1500
        cur.execute('''
            INSERT OR IGNORE INTO elo_ratings (team_id, rating, games_played)
            VALUES (?, 1500, 0)
        ''', (team_id,))
        
        conn.commit()
    
    # Cache it
    _espn_id_cache[espn_id_str] = team_id
    
    return team_id


def sync_teams():
    """Fetch all D1 teams from ESPN and ensure they're in our DB."""
    print("Fetching all D1 teams from ESPN...")
    
    data = api_get(f"{TEAMS_URL}?limit=500")
    if not data:
        print("Failed to fetch teams")
        return 0
    
    teams = data.get('sports', [{}])[0].get('leagues', [{}])[0].get('teams', [])
    print(f"ESPN reports {len(teams)} teams")
    
    conn = get_conn()
    created = 0
    mapping = {}
    
    # Load existing mapping
    if ESPN_TEAMS_PATH.exists():
        with open(ESPN_TEAMS_PATH) as f:
            mapping = json.load(f)
    
    for t_wrapper in teams:
        t = t_wrapper['team']
        espn_id = t['id']
        display_name = t['displayName']
        abbreviation = t['abbreviation']
        
        team_id = resolve_team(espn_id, display_name, abbreviation, conn)
        
        # Check if this was newly created
        if team_id not in mapping.values() and str(espn_id) not in {str(v) for v in mapping.values()}:
            # It's a new mapping
            if team_id not in mapping:
                created += 1
        
        mapping[team_id] = str(espn_id)
    
    conn.close()
    save_espn_id_map(mapping)
    
    print(f"Teams synced: {len(mapping)} total, {created} newly created")
    return created


def sync_date(date_str):
    """Sync all games for a given date from ESPN scoreboard."""
    date_param = date_str.replace('-', '')
    url = f"{SCOREBOARD_URL}?dates={date_param}&limit=200"
    
    print(f"\nSyncing {date_str}...")
    data = api_get(url)
    if not data:
        print(f"  Failed to fetch scoreboard for {date_str}")
        return {"games_created": 0, "games_updated": 0, "teams_created": 0}
    
    events = data.get('events', [])
    print(f"  ESPN shows {len(events)} events")
    
    conn = get_conn()
    cur = conn.cursor()
    
    stats = {"games_created": 0, "games_updated": 0, "teams_created": 0, "skipped": 0}
    
    # Track teams before to count new ones
    cur.execute("SELECT COUNT(*) FROM teams")
    teams_before = cur.fetchone()[0]
    
    for event in events:
        comp = event['competitions'][0]
        status_name = comp.get('status', {}).get('type', {}).get('name', '')
        
        # Skip cancelled/postponed
        if status_name in ('STATUS_CANCELED', 'STATUS_POSTPONED'):
            stats["skipped"] += 1
            continue
        
        competitors = comp['competitors']
        home_comp = next((c for c in competitors if c['homeAway'] == 'home'), None)
        away_comp = next((c for c in competitors if c['homeAway'] == 'away'), None)
        
        if not home_comp or not away_comp:
            continue
        
        home_team = home_comp['team']
        away_team = away_comp['team']
        
        # Resolve teams (creates if needed)
        home_id = resolve_team(home_team['id'], home_team['displayName'], 
                              home_team['abbreviation'], conn)
        away_id = resolve_team(away_team['id'], away_team['displayName'],
                              away_team['abbreviation'], conn)
        
        # Build game_id — also check for swapped version from schedule loader
        game_id = f"{date_str}_{away_id}_{home_id}"
        swapped_id = f"{date_str}_{home_id}_{away_id}"
        
        # Get scores if final
        home_score = None
        away_score = None
        game_status = 'scheduled'
        
        if status_name == 'STATUS_FINAL':
            home_score = int(home_comp.get('score', 0))
            away_score = int(away_comp.get('score', 0))
            game_status = 'final'
        elif status_name == 'STATUS_IN_PROGRESS':
            home_score = int(home_comp.get('score', 0))
            away_score = int(away_comp.get('score', 0))
            game_status = 'in_progress'
        
        winner_id = None
        if game_status == 'final' and home_score is not None:
            winner_id = home_id if home_score > away_score else away_id
        
        # Check venue
        venue = comp.get('venue', {}).get('fullName', None)
        neutral = 1 if comp.get('neutralSite', False) else 0
        conf_game = 1 if comp.get('conferenceCompetition', False) else 0
        
        # Detect innings from linescores
        innings = 9
        if home_comp.get('linescores'):
            innings = len(home_comp['linescores'])
        
        # Check if game exists — also check swapped home/away from schedule loader
        cur.execute("SELECT id, status FROM games WHERE id = ?", (game_id,))
        existing = cur.fetchone()
        
        if not existing:
            # Check for swapped version (schedule loader may have home/away reversed)
            cur.execute("SELECT id, status FROM games WHERE id = ? OR id LIKE ?", 
                       (swapped_id, swapped_id + '_g%'))
            swapped = cur.fetchone()
            if swapped:
                # Update the existing swapped game with correct home/away from ESPN
                cur.execute('''
                    UPDATE games SET home_team_id = ?, away_team_id = ?,
                    home_score = ?, away_score = ?, winner_id = ?,
                    status = ?, innings = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (home_id, away_id, home_score, away_score, winner_id,
                      game_status, innings, swapped['id']))
                if game_status == 'final':
                    stats["games_updated"] += 1
                continue  # Skip to next game
        
        # If game exists and is final with different scores, it's a doubleheader
        if existing and existing['status'] == 'final' and game_status == 'final':
            cur.execute("SELECT home_score, away_score FROM games WHERE id = ?", (game_id,))
            ex_scores = cur.fetchone()
            if ex_scores and (ex_scores['home_score'] != home_score or ex_scores['away_score'] != away_score):
                # Different scores = different game (doubleheader)
                # First check if this score already exists in any suffix
                already_exists = False
                for check_num in range(1, 5):
                    check_id = f"{game_id}_g{check_num}"
                    cur.execute("SELECT home_score, away_score FROM games WHERE id = ?", (check_id,))
                    check_row = cur.fetchone()
                    if check_row and check_row['home_score'] == home_score and check_row['away_score'] == away_score:
                        # This game already recorded under a suffix
                        already_exists = True
                        game_id = check_id
                        existing = cur.execute("SELECT id, status FROM games WHERE id = ?", (game_id,)).fetchone()
                        break
                
                if not already_exists:
                    # Find next available suffix
                    for suffix_num in range(1, 5):
                        dh_id = f"{game_id}_g{suffix_num}"
                        cur.execute("SELECT id FROM games WHERE id = ?", (dh_id,))
                        if not cur.fetchone():
                            game_id = dh_id
                            existing = None
                            break
        
        if existing:
            # Update if we have new score data
            if game_status == 'final' and existing['status'] != 'final':
                cur.execute('''
                    UPDATE games SET home_score=?, away_score=?, winner_id=?, 
                    status='final', innings=?, updated_at=CURRENT_TIMESTAMP
                    WHERE id=?
                ''', (home_score, away_score, winner_id, innings, game_id))
                stats["games_updated"] += 1
            elif game_status == 'final' and existing['status'] == 'final':
                pass  # Already final, skip
            else:
                pass  # Not final yet
        else:
            # Create game
            cur.execute('''
                INSERT INTO games (id, date, home_team_id, away_team_id, home_score, away_score,
                    winner_id, innings, is_conference_game, is_neutral_site, venue, status,
                    created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ''', (game_id, date_str, home_id, away_id, home_score, away_score,
                  winner_id, innings, conf_game, neutral, venue, game_status))
            stats["games_created"] += 1
    
    conn.commit()
    
    # Count new teams
    cur.execute("SELECT COUNT(*) FROM teams")
    teams_after = cur.fetchone()[0]
    stats["teams_created"] = teams_after - teams_before
    
    conn.close()
    
    print(f"  Created: {stats['games_created']} games, Updated: {stats['games_updated']}, "
          f"New teams: {stats['teams_created']}, Skipped: {stats['skipped']}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='ESPN D1 Baseball Sync')
    parser.add_argument('--today', action='store_true', help='Sync today')
    parser.add_argument('--yesterday', action='store_true', help='Sync yesterday')
    parser.add_argument('--date', help='Sync specific date (YYYY-MM-DD)')
    parser.add_argument('--range', nargs=2, metavar=('START', 'END'), help='Sync date range')
    parser.add_argument('--sync-teams', action='store_true', help='Sync all D1 teams from ESPN')
    parser.add_argument('--backfill', action='store_true', help='After sync, backfill missing scores via team schedules')
    args = parser.parse_args()
    
    # Load existing ESPN ID mapping
    load_espn_id_map()
    
    if args.sync_teams:
        sync_teams()
        return
    
    # Determine dates to sync
    dates = []
    if args.today:
        dates = [datetime.now().strftime('%Y-%m-%d')]
    elif args.yesterday:
        dates = [(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')]
    elif args.date:
        dates = [args.date]
    elif args.range:
        start = datetime.strptime(args.range[0], '%Y-%m-%d')
        end = datetime.strptime(args.range[1], '%Y-%m-%d')
        d = start
        while d <= end:
            dates.append(d.strftime('%Y-%m-%d'))
            d += timedelta(days=1)
    else:
        # Default: yesterday
        dates = [(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')]
    
    totals = {"games_created": 0, "games_updated": 0, "teams_created": 0}
    
    for date_str in dates:
        stats = sync_date(date_str)
        for k in totals:
            totals[k] += stats.get(k, 0)
        if len(dates) > 1:
            time.sleep(1.5)  # Rate limit between dates
    
    # Save updated ESPN ID mapping
    if _espn_id_cache:
        mapping = {v: k for k, v in _espn_id_cache.items()}  # flip: our_id -> espn_id
        # Merge with existing
        existing = {}
        if ESPN_TEAMS_PATH.exists():
            with open(ESPN_TEAMS_PATH) as f:
                existing = json.load(f)
        existing.update(mapping)
        save_espn_id_map(existing)
    
    print(f"\n{'='*50}")
    print(f"TOTAL: {totals['games_created']} games created, {totals['games_updated']} updated, "
          f"{totals['teams_created']} new teams")
    
    # Run backfill if requested — catches games the scoreboard API missed
    if args.backfill:
        print("\n--- Running backfill for missing scores ---")
        from espn_backfill import backfill
        backfill()


if __name__ == '__main__':
    main()
