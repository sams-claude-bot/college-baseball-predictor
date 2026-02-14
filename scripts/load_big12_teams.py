#!/usr/bin/env python3
"""
Load Big 12 Conference teams - rosters and full schedules

Big 12 Teams (16):
Arizona, Arizona State, BYU, Baylor, Cincinnati, Colorado, Houston, Iowa State,
Kansas, Kansas State, Oklahoma State, TCU, Texas Tech, UCF, Utah, West Virginia
"""

import sys
import re
import time
import sqlite3
import json
from pathlib import Path
from datetime import datetime

import requests
from bs4 import BeautifulSoup

_scripts_dir = Path(__file__).parent
# sys.path.insert(0, str(_scripts_dir))  # Removed by cleanup

from scripts.database import get_connection, add_team
from scripts.player_stats import add_player, init_player_tables

DB_PATH = Path(__file__).parent.parent / "data" / "baseball.db"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

# Big 12 Teams with ESPN IDs and roster URLs
BIG12_TEAMS = {
    "arizona": {
        "name": "Arizona",
        "espn_id": 12,
        "roster_url": "https://arizonawildcats.com/sports/baseball/roster",
        "site_type": "sidearm"
    },
    "arizona-state": {
        "name": "Arizona State",
        "espn_id": 9,
        "roster_url": "https://thesundevils.com/sports/baseball/roster",
        "site_type": "sidearm"
    },
    "byu": {
        "name": "BYU",
        "espn_id": 252,
        "roster_url": "https://byucougars.com/sports/baseball/roster",
        "site_type": "sidearm"
    },
    "baylor": {
        "name": "Baylor",
        "espn_id": 239,
        "roster_url": "https://baylorbears.com/sports/baseball/roster",
        "site_type": "sidearm"
    },
    "cincinnati": {
        "name": "Cincinnati",
        "espn_id": 2132,
        "roster_url": "https://gobearcats.com/sports/baseball/roster",
        "site_type": "sidearm"
    },
    "colorado": {
        "name": "Colorado",
        "espn_id": 38,
        "roster_url": "https://cubuffs.com/sports/baseball/roster",
        "site_type": "sidearm"
    },
    "houston": {
        "name": "Houston",
        "espn_id": 248,
        "roster_url": "https://uhcougars.com/sports/baseball/roster",
        "site_type": "sidearm"
    },
    "iowa-state": {
        "name": "Iowa State",
        "espn_id": 66,
        "roster_url": "https://cyclones.com/sports/baseball/roster",
        "site_type": "sidearm"
    },
    "kansas": {
        "name": "Kansas",
        "espn_id": 2305,
        "roster_url": "https://kuathletics.com/sports/baseball/roster",
        "site_type": "sidearm"
    },
    "kansas-state": {
        "name": "Kansas State",
        "espn_id": 2306,
        "roster_url": "https://kstatesports.com/sports/baseball/roster",
        "site_type": "sidearm"
    },
    "oklahoma-state": {
        "name": "Oklahoma State",
        "espn_id": 197,
        "roster_url": "https://okstate.com/sports/baseball/roster",
        "site_type": "sidearm"
    },
    "tcu": {
        "name": "TCU",
        "espn_id": 2628,
        "roster_url": "https://gofrogs.com/sports/baseball/roster",
        "site_type": "sidearm"
    },
    "texas-tech": {
        "name": "Texas Tech",
        "espn_id": 2641,
        "roster_url": "https://texastech.com/sports/baseball/roster",
        "site_type": "sidearm"
    },
    "ucf": {
        "name": "UCF",
        "espn_id": 2116,
        "roster_url": "https://ucfknights.com/sports/baseball/roster",
        "site_type": "sidearm"
    },
    "utah": {
        "name": "Utah",
        "espn_id": 254,
        "roster_url": "https://utahutes.com/sports/baseball/roster",
        "site_type": "sidearm"
    },
    "west-virginia": {
        "name": "West Virginia",
        "espn_id": 277,
        "roster_url": "https://wvusports.com/sports/baseball/roster",
        "site_type": "sidearm"
    }
}

# Common team name mappings for opponent parsing
TEAM_NAME_MAP = {
    # Big 12 teams
    'arizona': 'arizona',
    'arizona wildcats': 'arizona',
    'arizona state': 'arizona-state',
    'asu': 'arizona-state',
    'sun devils': 'arizona-state',
    'byu': 'byu',
    'brigham young': 'byu',
    'baylor': 'baylor',
    'baylor bears': 'baylor',
    'cincinnati': 'cincinnati',
    'bearcats': 'cincinnati',
    'colorado': 'colorado',
    'buffaloes': 'colorado',
    'houston': 'houston',
    'cougars': 'houston',
    'iowa state': 'iowa-state',
    'cyclones': 'iowa-state',
    'kansas': 'kansas',
    'jayhawks': 'kansas',
    'kansas state': 'kansas-state',
    'k-state': 'kansas-state',
    'oklahoma state': 'oklahoma-state',
    'ok state': 'oklahoma-state',
    'osu': 'oklahoma-state',
    'cowboys': 'oklahoma-state',
    'tcu': 'tcu',
    'horned frogs': 'tcu',
    'texas tech': 'texas-tech',
    'red raiders': 'texas-tech',
    'ucf': 'ucf',
    'central florida': 'ucf',
    'knights': 'ucf',
    'utah': 'utah',
    'utes': 'utah',
    'west virginia': 'west-virginia',
    'wvu': 'west-virginia',
    'mountaineers': 'west-virginia',
    # Additional mappings from SEC script
    'texas': 'texas',
    'oklahoma': 'oklahoma',
    'lsu': 'lsu',
    'florida': 'florida',
    'georgia': 'georgia',
    'alabama': 'alabama',
    'arkansas': 'arkansas',
    'tennessee': 'tennessee',
    'vanderbilt': 'vanderbilt',
    'ole miss': 'ole-miss',
    'mississippi state': 'mississippi-state',
    'auburn': 'auburn',
    'kentucky': 'kentucky',
    'missouri': 'missouri',
    'south carolina': 'south-carolina',
    'texas a&m': 'texas-am',
}


def normalize_team_name(name):
    """Convert team name to standard ID."""
    if not name:
        return None
    
    name_lower = name.lower().strip()
    # Remove rankings like "#2" or "No. 14"
    name_lower = re.sub(r'^#?\d+\s+', '', name_lower)
    name_lower = re.sub(r'^no\.\s*\d+\s+', '', name_lower)
    name_lower = name_lower.strip()
    
    if name_lower in TEAM_NAME_MAP:
        return TEAM_NAME_MAP[name_lower]
    
    # Try to create an ID from the name
    team_id = re.sub(r'[^a-z0-9]+', '-', name_lower).strip('-')
    return team_id


def normalize_year(year_str):
    """Normalize academic year to standard format"""
    if not year_str:
        return None
    year_str = year_str.strip()
    
    mappings = {
        'freshman': 'Fr.', 'fr': 'Fr.', 'fr.': 'Fr.',
        'sophomore': 'So.', 'so': 'So.', 'so.': 'So.',
        'junior': 'Jr.', 'jr': 'Jr.', 'jr.': 'Jr.',
        'senior': 'Sr.', 'sr': 'Sr.', 'sr.': 'Sr.',
        'graduate': 'Gr.', 'gr': 'Gr.', 'gr.': 'Gr.',
        'redshirt freshman': 'R-Fr.', 'r-fr': 'R-Fr.', 'r-fr.': 'R-Fr.',
        'redshirt sophomore': 'R-So.', 'r-so': 'R-So.', 'r-so.': 'R-So.',
        'redshirt junior': 'R-Jr.', 'r-jr': 'R-Jr.', 'r-jr.': 'R-Jr.',
        'redshirt senior': 'R-Sr.', 'r-sr': 'R-Sr.', 'r-sr.': 'R-Sr.',
    }
    
    for key, val in mappings.items():
        if key in year_str.lower():
            return val
    
    return year_str


def normalize_position(pos_str):
    """Normalize position to standard abbreviations"""
    if not pos_str:
        return None
    pos_str = pos_str.strip()
    
    mappings = [
        ('right-handed pitcher', 'RHP'),
        ('left-handed pitcher', 'LHP'),
        ('right handed pitcher', 'RHP'),
        ('left handed pitcher', 'LHP'),
        ('catcher', 'C'),
        ('first base', '1B'),
        ('second base', '2B'),
        ('third base', '3B'),
        ('shortstop', 'SS'),
        ('outfielder', 'OF'),
        ('outfield', 'OF'),
        ('infielder', 'INF'),
        ('infield', 'INF'),
        ('designated hitter', 'DH'),
        ('utility', 'UTL'),
        ('pitcher', 'P'),
    ]
    
    result = pos_str.lower()
    for long_form, abbrev in mappings:
        result = result.replace(long_form, abbrev)
    
    result = result.upper()
    result = result.replace('RIGHT-HANDED', 'RHP').replace('LEFT-HANDED', 'LHP')
    result = re.sub(r'\s*/\s*', '/', result.strip())
    
    return result


def parse_bats_throws(bt_str):
    """Parse bats/throws string like 'R/R' or 'L/R'"""
    if not bt_str:
        return None, None
    
    match = re.search(r'([RLSB])\s*/\s*([RL])', bt_str.upper())
    if match:
        return match.group(1), match.group(2)
    
    return None, None


def parse_height(height_str):
    """Parse height string to standard format"""
    if not height_str:
        return None
    
    match = re.search(r"(\d+)['\-]?\s*(\d+)?", height_str)
    if match:
        feet = match.group(1)
        inches = match.group(2) or '0'
        return f"{feet}' {inches}''"
    
    return height_str


def parse_weight(weight_str):
    """Parse weight string to integer"""
    if not weight_str:
        return None
    
    match = re.search(r'(\d+)', str(weight_str))
    if match:
        return int(match.group(1))
    return None


def fetch_page(url, timeout=30):
    """Fetch a web page with error handling"""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        print(f"  ✗ Error fetching {url}: {e}")
        return None


def parse_sidearm_roster(html, team_id):
    """
    Parse roster from Sidearm Sports platform (most NCAA teams use this)
    """
    soup = BeautifulSoup(html, 'html.parser')
    players = []
    
    text = soup.get_text('|', strip=True)
    
    # Method 1: Pattern with Custom Field 1 (B/T)
    pattern_with_bt = r'Jersey Number\|(\d+)\|([^|]+)\|Position\|([^|]+)\|Academic Year\|([^|]+)\|Height\|([^|]+)\|Weight\|([^|]+)\|Custom Field 1\|([^|]+)'
    
    for match in re.finditer(pattern_with_bt, text):
        number = int(match.group(1))
        name = match.group(2).strip()
        position = normalize_position(match.group(3).strip())
        year = normalize_year(match.group(4).strip())
        height = parse_height(match.group(5).strip())
        weight = parse_weight(match.group(6).strip())
        bt = match.group(7).strip()
        
        bats, throws = parse_bats_throws(bt)
        
        players.append({
            'name': name,
            'number': number,
            'position': position,
            'year': year,
            'height': height,
            'weight': weight,
            'bats': bats,
            'throws': throws
        })
    
    if players:
        return players
    
    # Method 2: Pattern without B/T
    pattern_no_bt = r'Jersey Number\|(\d+)\|([^|]+)\|Position\|([^|]+)\|Academic Year\|([^|]+)\|Height\|([^|]+)\|Weight\|([^|]+)'
    
    for match in re.finditer(pattern_no_bt, text):
        number = int(match.group(1))
        name = match.group(2).strip()
        position = normalize_position(match.group(3).strip())
        year = normalize_year(match.group(4).strip())
        height = parse_height(match.group(5).strip())
        weight = parse_weight(match.group(6).strip())
        
        players.append({
            'name': name,
            'number': number,
            'position': position,
            'year': year,
            'height': height,
            'weight': weight,
            'bats': None,
            'throws': None
        })
    
    if players:
        return players
    
    # Method 3: Look for roster cards (older Sidearm)
    cards = soup.find_all('li', class_=re.compile(r'sidearm-roster-player'))
    if not cards:
        cards = soup.find_all('div', class_=re.compile(r'sidearm-roster-player'))
    
    for card in cards:
        player = {}
        
        name_el = card.find('a', class_=re.compile(r'name')) or card.find(class_=re.compile(r'name'))
        if name_el:
            player['name'] = name_el.get_text(strip=True)
        
        number_el = card.find(class_=re.compile(r'number|jersey'))
        if number_el:
            num_text = number_el.get_text(strip=True)
            num_match = re.search(r'\d+', num_text)
            if num_match:
                player['number'] = int(num_match.group())
        
        pos_el = card.find(class_=re.compile(r'position'))
        if pos_el:
            player['position'] = normalize_position(pos_el.get_text(strip=True))
        
        year_el = card.find(class_=re.compile(r'academic|year|class'))
        if year_el:
            player['year'] = normalize_year(year_el.get_text(strip=True))
        
        height_el = card.find(class_=re.compile(r'height'))
        if height_el:
            player['height'] = parse_height(height_el.get_text(strip=True))
        
        weight_el = card.find(class_=re.compile(r'weight'))
        if weight_el:
            player['weight'] = parse_weight(weight_el.get_text(strip=True))
        
        bt_el = card.find(class_=re.compile(r'custom|b-t|bats'))
        if bt_el:
            bats, throws = parse_bats_throws(bt_el.get_text(strip=True))
            player['bats'] = bats
            player['throws'] = throws
        
        if player.get('name'):
            players.append(player)
    
    return players


def scrape_team_roster(team_id):
    """Scrape roster for a Big 12 team"""
    if team_id not in BIG12_TEAMS:
        print(f"  ✗ Unknown team: {team_id}")
        return []
    
    config = BIG12_TEAMS[team_id]
    url = config['roster_url']
    
    print(f"\n📥 Fetching {config['name']} roster from {url}")
    
    html = fetch_page(url)
    if not html:
        return []
    
    players = parse_sidearm_roster(html, team_id)
    print(f"  ✓ Found {len(players)} players")
    
    return players


def save_roster_to_db(team_id, players):
    """Save roster to database"""
    if not players:
        print(f"  ⊘ No players to save for {team_id}")
        return 0
    
    init_player_tables()
    added = 0
    
    for player in players:
        if not player.get('name'):
            continue
        
        try:
            add_player(
                team_id=team_id,
                name=player['name'],
                number=player.get('number'),
                position=player.get('position'),
                year=player.get('year'),
                bats=player.get('bats'),
                throws=player.get('throws'),
                height=player.get('height'),
                weight=player.get('weight')
            )
            added += 1
        except Exception as e:
            print(f"  ✗ Error saving {player['name']}: {e}")
    
    print(f"  📝 Database: {added} players saved")
    return added


def parse_sidearm_schedule(html, team_id):
    """Parse schedule from Sidearm Sports schedule page (HTML structure)"""
    soup = BeautifulSoup(html, 'html.parser')
    games = []
    
    # Find all game list items with data-game-id attribute
    game_elements = soup.find_all('li', {'data-game-id': True})
    
    if not game_elements:
        print(f"  ⚠ Could not find game elements for {team_id}")
        return games
    
    for g in game_elements:
        classes = ' '.join(g.get('class', []))
        
        # Determine home/away/neutral
        is_home = 'home-game' in classes
        is_away = 'away-game' in classes  
        is_neutral = 'neutral-game' in classes
        
        # Get date text
        date_div = g.find(class_=re.compile('date'))
        date_text = date_div.get_text(strip=True) if date_div else ''
        
        # Parse date - format: "Feb 13 (Fri)" or "Mar 1 (Sun)6:00 PM"
        date_match = re.search(r'(Jan|Feb|Mar|Apr|May|Jun)\s+(\d{1,2})', date_text)
        if not date_match:
            continue
        
        month_str = date_match.group(1).lower()
        day = int(date_match.group(2))
        
        month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6}
        month = month_map.get(month_str)
        if not month:
            continue
        
        # Only include regular season (Feb-May)
        if month < 2:
            continue
        
        date = f"2026-{month:02d}-{day:02d}"
        
        # Get opponent name
        opp_div = g.find(class_=re.compile('opponent-name'))
        opponent_name = opp_div.get_text(strip=True) if opp_div else ''
        
        if not opponent_name:
            continue
        
        # Skip exhibition/non-games
        if 'meet the team' in opponent_name.lower():
            continue
        if 'exhibition' in opponent_name.lower():
            continue
        
        # Normalize opponent ID
        opponent_id = normalize_team_name(opponent_name)
        if not opponent_id:
            opponent_id = re.sub(r'[^a-z0-9]+', '-', opponent_name.lower()).strip('-')
        
        # Get time if present
        time_match = re.search(r'(\d{1,2}:\d{2}\s*(?:AM|PM)?)', date_text, re.IGNORECASE)
        time_str = time_match.group(1) if time_match else ''
        
        # Check if conference game (Big 12 opponent or has conference marker)
        is_conference = opponent_id in BIG12_TEAMS
        conf_marker = g.find(class_=re.compile('conference'))
        if conf_marker and 'big 12' in conf_marker.get_text(strip=True).lower():
            is_conference = True
        
        games.append({
            'date': date,
            'opponent_name': opponent_name,
            'opponent_id': opponent_id,
            'is_home': is_home,
            'is_neutral': is_neutral,
            'is_conference': is_conference,
            'time': time_str
        })
    
    return games


def fetch_schedule(team_id):
    """Fetch schedule from athletics site for a Big 12 team"""
    if team_id not in BIG12_TEAMS:
        print(f"  ✗ Unknown team: {team_id}")
        return []
    
    config = BIG12_TEAMS[team_id]
    # Get schedule URL from roster URL (replace /roster with /schedule)
    schedule_url = config['roster_url'].replace('/roster', '/schedule')
    
    print(f"\n📅 Fetching {config['name']} schedule...")
    
    html = fetch_page(schedule_url)
    if not html:
        return []
    
    games = parse_sidearm_schedule(html, team_id)
    print(f"  ✓ Found {len(games)} games")
    
    return games


def save_schedule_to_db(team_id, games):
    """Save schedule games to database"""
    if not games:
        return 0
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    added = 0
    teams_added = set()
    
    for game in games:
        opponent_id = game['opponent_id']
        
        # Ensure opponent team exists
        cursor.execute("SELECT id FROM teams WHERE id = ?", (opponent_id,))
        if not cursor.fetchone():
            teams_added.add(opponent_id)
            cursor.execute("""
                INSERT OR IGNORE INTO teams (id, name) VALUES (?, ?)
            """, (opponent_id, game['opponent_name']))
        
        # Determine home/away - for neutral site, use the team we're loading as "home" 
        # to keep consistent game IDs
        is_neutral = game.get('is_neutral', False)
        if game['is_home'] or is_neutral:
            home_team_id = team_id
            away_team_id = opponent_id
        else:
            home_team_id = opponent_id
            away_team_id = team_id
        
        # Create game ID
        game_id = f"{game['date']}_{away_team_id}_{home_team_id}"
        
        # Check if it's a conference game (both teams in Big 12)
        is_conference = 1 if opponent_id in BIG12_TEAMS else 0
        # Also check the JSON flag
        if game.get('is_conference'):
            is_conference = 1
        
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO games 
                (id, date, time, home_team_id, away_team_id, is_conference_game, is_neutral_site, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'scheduled')
            """, (game_id, game['date'], game.get('time'), home_team_id, away_team_id, 
                  is_conference, 1 if is_neutral else 0))
            
            if cursor.rowcount > 0:
                added += 1
        except sqlite3.IntegrityError:
            pass
    
    conn.commit()
    conn.close()
    
    if teams_added:
        print(f"  📝 Added {len(teams_added)} new opponent teams")
    print(f"  📝 Database: {added} games added")
    
    return added


def add_big12_teams():
    """Add all Big 12 teams to database"""
    print("\n" + "=" * 60)
    print("  Adding Big 12 Teams to Database")
    print("=" * 60)
    
    for team_id, config in BIG12_TEAMS.items():
        add_team(
            team_id=team_id,
            name=config['name'],
            conference='Big 12'
        )
        print(f"  ✓ {config['name']}")
    
    print(f"\n  Added {len(BIG12_TEAMS)} Big 12 teams")


def load_all_rosters(delay=1.5):
    """Scrape and load rosters for all Big 12 teams"""
    print("\n" + "=" * 60)
    print("  Loading Big 12 Rosters")
    print("=" * 60)
    
    total_players = 0
    
    for team_id in BIG12_TEAMS:
        players = scrape_team_roster(team_id)
        saved = save_roster_to_db(team_id, players)
        total_players += saved
        time.sleep(delay)
    
    print(f"\n  Total: {total_players} players loaded")
    return total_players


def load_all_schedules(delay=1.5):
    """Fetch and load schedules for all Big 12 teams"""
    print("\n" + "=" * 60)
    print("  Loading Big 12 Schedules")
    print("=" * 60)
    
    total_games = 0
    
    for team_id in BIG12_TEAMS:
        games = fetch_schedule(team_id)
        saved = save_schedule_to_db(team_id, games)
        total_games += saved
        time.sleep(delay)
    
    print(f"\n  Total: {total_games} unique games loaded")
    return total_games


def show_status():
    """Show current Big 12 status in database"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    print("\n📊 Big 12 Status:")
    print("-" * 50)
    
    # Teams
    cursor.execute("""
        SELECT COUNT(*) FROM teams WHERE conference = 'Big 12'
    """)
    team_count = cursor.fetchone()[0]
    print(f"  Teams: {team_count}")
    
    # Players per team
    cursor.execute("""
        SELECT t.name, COUNT(p.id) as player_count
        FROM teams t
        LEFT JOIN player_stats p ON t.id = p.team_id
        WHERE t.conference = 'Big 12'
        GROUP BY t.id
        ORDER BY player_count DESC
    """)
    print("\n  Players per team:")
    for row in cursor.fetchall():
        print(f"    {row['name']}: {row['player_count']}")
    
    # Games
    cursor.execute("""
        SELECT COUNT(DISTINCT g.id) as game_count
        FROM games g
        JOIN teams t ON g.home_team_id = t.id OR g.away_team_id = t.id
        WHERE t.conference = 'Big 12'
    """)
    game_count = cursor.fetchone()[0]
    print(f"\n  Total games involving Big 12: {game_count}")
    
    # Conference games
    cursor.execute("""
        SELECT COUNT(*) FROM games
        WHERE home_team_id IN (SELECT id FROM teams WHERE conference = 'Big 12')
        AND away_team_id IN (SELECT id FROM teams WHERE conference = 'Big 12')
    """)
    conf_games = cursor.fetchone()[0]
    print(f"  Conference games: {conf_games}")
    
    conn.close()


def main():
    if len(sys.argv) < 2:
        # Default: load everything
        print("\n🏈 Big 12 Conference Loader")
        print("=" * 60)
        
        add_big12_teams()
        load_all_rosters()
        load_all_schedules()
        show_status()
        
    elif sys.argv[1] == '--teams':
        add_big12_teams()
    elif sys.argv[1] == '--rosters':
        load_all_rosters()
    elif sys.argv[1] == '--schedules':
        load_all_schedules()
    elif sys.argv[1] == '--status':
        show_status()
    elif sys.argv[1] in BIG12_TEAMS:
        # Single team
        team_id = sys.argv[1]
        config = BIG12_TEAMS[team_id]
        print(f"\n🏈 Loading {config['name']}...")
        
        add_team(team_id=team_id, name=config['name'], conference='Big 12')
        
        players = scrape_team_roster(team_id)
        save_roster_to_db(team_id, players)
        
        games = fetch_schedule(team_id)
        save_schedule_to_db(team_id, games)
    else:
        print("Usage:")
        print("  python load_big12_teams.py              # Load all teams, rosters, schedules")
        print("  python load_big12_teams.py --teams      # Add teams only")
        print("  python load_big12_teams.py --rosters    # Load rosters only")
        print("  python load_big12_teams.py --schedules  # Load schedules only")
        print("  python load_big12_teams.py --status     # Show current status")
        print("  python load_big12_teams.py <team-id>    # Load single team")


if __name__ == "__main__":
    main()
