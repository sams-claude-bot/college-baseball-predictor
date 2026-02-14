#!/usr/bin/env python3
"""
Load Big Ten Conference teams — rosters + full schedules

18 Teams:
Illinois, Indiana, Iowa, Maryland, Michigan, Michigan State, Minnesota, Nebraska,
Northwestern, Ohio State, Oregon, Oregon State, Penn State, Purdue, Rutgers,
UCLA, USC, Washington
"""

import sqlite3
import re
import time
import sys
from datetime import datetime
from pathlib import Path

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

# Big Ten teams with ESPN IDs
BIG_TEN_TEAMS = {
    'illinois': {'name': 'Illinois', 'espn_id': 356},
    'indiana': {'name': 'Indiana', 'espn_id': 84},
    'iowa': {'name': 'Iowa', 'espn_id': 2294},
    'maryland': {'name': 'Maryland', 'espn_id': 120},
    'michigan': {'name': 'Michigan', 'espn_id': 130},
    'michigan-state': {'name': 'Michigan State', 'espn_id': 127},
    'minnesota': {'name': 'Minnesota', 'espn_id': 135},
    'nebraska': {'name': 'Nebraska', 'espn_id': 158},
    'northwestern': {'name': 'Northwestern', 'espn_id': 77},
    'ohio-state': {'name': 'Ohio State', 'espn_id': 194},
    'oregon': {'name': 'Oregon', 'espn_id': 2483},
    'oregon-state': {'name': 'Oregon State', 'espn_id': 204},
    'penn-state': {'name': 'Penn State', 'espn_id': 213},
    'purdue': {'name': 'Purdue', 'espn_id': 2509},
    'rutgers': {'name': 'Rutgers', 'espn_id': 164},
    'ucla': {'name': 'UCLA', 'espn_id': 26},
    'usc': {'name': 'USC', 'espn_id': 30},
    'washington': {'name': 'Washington', 'espn_id': 264},
}

# Roster URLs for Big Ten teams (athletics sites using Sidearm Sports)
BIG_TEN_ROSTER_URLS = {
    'illinois': 'https://fightingillini.com/sports/baseball/roster',
    'indiana': 'https://iuhoosiers.com/sports/baseball/roster',
    'iowa': 'https://hawkeyesports.com/sports/baseball/roster',
    'maryland': 'https://umterps.com/sports/baseball/roster',
    'michigan': 'https://mgoblue.com/sports/baseball/roster',
    'michigan-state': 'https://msuspartans.com/sports/baseball/roster',
    'minnesota': 'https://gophersports.com/sports/baseball/roster',
    'nebraska': 'https://huskers.com/sports/baseball/roster',
    'northwestern': 'https://nusports.com/sports/baseball/roster',
    'ohio-state': 'https://ohiostatebuckeyes.com/sports/baseball/roster',
    'oregon': 'https://goducks.com/sports/baseball/roster',
    'oregon-state': 'https://osubeavers.com/sports/baseball/roster',
    'penn-state': 'https://gopsusports.com/sports/baseball/roster',
    'purdue': 'https://purduesports.com/sports/baseball/roster',
    'rutgers': 'https://scarletknights.com/sports/baseball/roster',
    'ucla': 'https://uclabruins.com/sports/baseball/roster',
    'usc': 'https://usctrojans.com/sports/baseball/roster',
    'washington': 'https://gohuskies.com/sports/baseball/roster',
}

# Team name normalization map
TEAM_NAME_MAP = {
    # Big Ten teams
    'illinois': 'illinois',
    'indiana': 'indiana',
    'iowa': 'iowa',
    'maryland': 'maryland',
    'michigan': 'michigan',
    'michigan state': 'michigan-state',
    'michigan st.': 'michigan-state',
    'michigan st': 'michigan-state',
    'minnesota': 'minnesota',
    'nebraska': 'nebraska',
    'northwestern': 'northwestern',
    'ohio state': 'ohio-state',
    'ohio st.': 'ohio-state',
    'ohio st': 'ohio-state',
    'oregon': 'oregon',
    'oregon state': 'oregon-state',
    'oregon st.': 'oregon-state',
    'oregon st': 'oregon-state',
    'penn state': 'penn-state',
    'penn st.': 'penn-state',
    'penn st': 'penn-state',
    'purdue': 'purdue',
    'rutgers': 'rutgers',
    'ucla': 'ucla',
    'usc': 'usc',
    'southern california': 'usc',
    'washington': 'washington',
    'uw': 'washington',
    # Common opponents
    'gonzaga': 'gonzaga',
    'stanford': 'stanford',
    'cal': 'california',
    'california': 'california',
    'uc berkeley': 'california',
    'arizona': 'arizona',
    'arizona state': 'arizona-state',
    'arizona st.': 'arizona-state',
    'san diego state': 'san-diego-state',
    'san diego st.': 'san-diego-state',
    'san diego': 'san-diego',
    'fresno state': 'fresno-state',
    'fresno st.': 'fresno-state',
    'long beach state': 'long-beach-state',
    'long beach st.': 'long-beach-state',
    'cal state fullerton': 'cal-state-fullerton',
    'fullerton': 'cal-state-fullerton',
    'uc irvine': 'uc-irvine',
    'uci': 'uc-irvine',
    'uc riverside': 'uc-riverside',
    'ucr': 'uc-riverside',
    'uc santa barbara': 'uc-santa-barbara',
    'ucsb': 'uc-santa-barbara',
    'uc davis': 'uc-davis',
    'ucd': 'uc-davis',
    'santa clara': 'santa-clara',
    'san jose state': 'san-jose-state',
    'san jose st.': 'san-jose-state',
    'pacific': 'pacific',
    'pepperdine': 'pepperdine',
    'lmu': 'loyola-marymount',
    'loyola marymount': 'loyola-marymount',
    'csun': 'cal-state-northridge',
    'cal state northridge': 'cal-state-northridge',
    'hawaii': 'hawaii',
    "hawai'i": 'hawaii',
    'seattle': 'seattle',
    'seattle u': 'seattle',
    'portland': 'portland',
    'portland st.': 'portland-state',
    'oregon state': 'oregon-state',
    'byu': 'byu',
    'brigham young': 'byu',
    'utah': 'utah',
    'new mexico': 'new-mexico',
    'new mexico state': 'new-mexico-state',
    'unlv': 'unlv',
    'tcu': 'tcu',
    'texas tech': 'texas-tech',
    'baylor': 'baylor',
    'oklahoma': 'oklahoma',
    'oklahoma state': 'oklahoma-state',
    'oklahoma st.': 'oklahoma-state',
    'kansas': 'kansas',
    'kansas state': 'kansas-state',
    'kansas st.': 'kansas-state',
    'wichita state': 'wichita-state',
    'wichita st.': 'wichita-state',
    'tulsa': 'tulsa',
    'oral roberts': 'oral-roberts',
    'oru': 'oral-roberts',
    'creighton': 'creighton',
    'dallas baptist': 'dallas-baptist',
    'dbu': 'dallas-baptist',
    'notre dame': 'notre-dame',
    'xavier': 'xavier',
    'cincinnati': 'cincinnati',
    'louisville': 'louisville',
    'kentucky': 'kentucky',
    'vanderbilt': 'vanderbilt',
    'tennessee': 'tennessee',
    'alabama': 'alabama',
    'auburn': 'auburn',
    'lsu': 'lsu',
    'ole miss': 'ole-miss',
    'mississippi state': 'mississippi-state',
    'miss state': 'mississippi-state',
    'mississippi': 'ole-miss',
    'arkansas': 'arkansas',
    'missouri': 'missouri',
    'texas': 'texas',
    'texas a&m': 'texas-am',
    'texas am': 'texas-am',
    'florida': 'florida',
    'georgia': 'georgia',
    'south carolina': 'south-carolina',
    'clemson': 'clemson',
    'duke': 'duke',
    'north carolina': 'north-carolina',
    'unc': 'north-carolina',
    'nc state': 'nc-state',
    'wake forest': 'wake-forest',
    'virginia': 'virginia',
    'virginia tech': 'virginia-tech',
    'miami': 'miami-fl',
    'miami (fl)': 'miami-fl',
    'florida state': 'florida-state',
    'fsu': 'florida-state',
    'georgia tech': 'georgia-tech',
    'pittsburgh': 'pittsburgh',
    'pitt': 'pittsburgh',
    'west virginia': 'west-virginia',
    'wvu': 'west-virginia',
    'east carolina': 'east-carolina',
    'ecu': 'east-carolina',
    'coastal carolina': 'coastal-carolina',
    'south florida': 'usf',
    'usf': 'usf',
    'ucf': 'ucf',
    'central florida': 'ucf',
    'uconn': 'uconn',
    'connecticut': 'uconn',
    'boston college': 'boston-college',
    'northeastern': 'northeastern',
    'army': 'army',
    'navy': 'navy',
    'air force': 'air-force',
    'ball state': 'ball-state',
    'bowling green': 'bowling-green',
    'kent state': 'kent-state',
    'ohio': 'ohio',
    'miami (oh)': 'miami-oh',
    'toledo': 'toledo',
    'western michigan': 'western-michigan',
    'central michigan': 'central-michigan',
    'eastern michigan': 'eastern-michigan',
    'northern illinois': 'northern-illinois',
    'wright state': 'wright-state',
    'youngstown state': 'youngstown-state',
    'dayton': 'dayton',
    'valparaiso': 'valparaiso',
    'butler': 'butler',
    'indiana state': 'indiana-state',
    'illinois state': 'illinois-state',
    'southern illinois': 'southern-illinois',
    'siu': 'southern-illinois',
    'missouri state': 'missouri-state',
    'evansville': 'evansville',
    'bradley': 'bradley',
    'northern iowa': 'northern-iowa',
    'uni': 'northern-iowa',
    'north dakota state': 'north-dakota-state',
    'ndsu': 'north-dakota-state',
    'south dakota state': 'south-dakota-state',
    'sdsu': 'south-dakota-state',
    'omaha': 'omaha',
    'nebraska-omaha': 'omaha',
    'st. thomas': 'st-thomas',
    'st thomas': 'st-thomas',
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


def fetch_page(url, timeout=30):
    """Fetch a web page with error handling"""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        print(f"  ✗ Error fetching {url}: {e}")
        return None


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
        '5th-year senior': 'Gr.', '5th year': 'Gr.',
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


def parse_sidearm_roster(html, team_id):
    """
    Parse roster from Sidearm Sports (most college teams use this platform)
    """
    soup = BeautifulSoup(html, 'html.parser')
    players = []
    
    text = soup.get_text('|', strip=True)
    
    # Method 1: Modern Sidearm with B/T
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
    
    # Method 2: Sidearm without B/T
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
    
    # Method 3: Alternate pattern
    pos_pattern = r'(RHP|LHP|C|1B|2B|3B|SS|OF|INF|UTL|IF|P|Right-handed Pitcher|Left-handed Pitcher|Catcher|Infielder|Outfielder|Pitcher|First Base|Second Base|Third Base|Shortstop)'
    alt_pattern = r'\|(\d{1,2})\|([A-Z][a-zA-Z\'\-\s\.]+)\|' + pos_pattern + r'\|(\d+[\-\']\d+)\|(\d+)\s*(?:lbs?\.)?\|(Freshman|Sophomore|Junior|Senior|Graduate|Redshirt Freshman|Redshirt Sophomore|Redshirt Junior|Redshirt Senior|Fr\.|So\.|Jr\.|Sr\.|R-Fr\.|R-So\.|R-Jr\.|R-Sr\.)\|'
    
    for match in re.finditer(alt_pattern, text):
        number = int(match.group(1))
        name = match.group(2).strip()
        position = normalize_position(match.group(3).strip())
        height = parse_height(match.group(4).strip())
        weight = parse_weight(match.group(5).strip())
        year = normalize_year(match.group(6).strip())
        
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
    
    # Method 4: Roster cards
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
        
        if player.get('name'):
            players.append(player)
    
    return players


def scrape_team_roster(team_id):
    """Scrape roster for a Big Ten team"""
    if team_id not in BIG_TEN_ROSTER_URLS:
        print(f"  ✗ No roster URL for: {team_id}")
        return []
    
    url = BIG_TEN_ROSTER_URLS[team_id]
    team_name = BIG_TEN_TEAMS[team_id]['name']
    
    print(f"\n📥 Fetching {team_name} roster from {url}")
    
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
    conn = get_connection()
    c = conn.cursor()
    
    c.execute("SELECT name FROM player_stats WHERE team_id = ?", (team_id,))
    existing = {row['name'].lower() for row in c.fetchall()}
    conn.close()
    
    added = 0
    updated = 0
    
    for player in players:
        if not player.get('name'):
            continue
        
        name = player['name']
        is_update = name.lower() in existing
        
        try:
            add_player(
                team_id=team_id,
                name=name,
                number=player.get('number'),
                position=player.get('position'),
                year=player.get('year'),
                bats=player.get('bats'),
                throws=player.get('throws'),
                height=player.get('height'),
                weight=player.get('weight')
            )
            
            if is_update:
                updated += 1
            else:
                added += 1
                
        except Exception as e:
            print(f"  ✗ Error saving {name}: {e}")
    
    print(f"  📝 Database: {added} added, {updated} updated")
    return added + updated


def parse_espn_schedule(html, team_id, espn_id):
    """Parse schedule from ESPN team schedule page"""
    soup = BeautifulSoup(html, 'html.parser')
    games = []
    
    # Look for schedule table rows
    rows = soup.find_all('tr', class_=re.compile(r'Table__TR'))
    
    for row in rows:
        cells = row.find_all('td')
        if len(cells) < 2:
            continue
        
        # Try to find date and opponent
        date_text = None
        opponent_text = None
        result_text = None
        is_home = True
        
        for i, cell in enumerate(cells):
            text = cell.get_text(strip=True)
            
            # Date pattern (e.g., "Fri, Feb 14")
            date_match = re.search(r'(Mon|Tue|Wed|Thu|Fri|Sat|Sun),?\s*(Jan|Feb|Mar|Apr|May|Jun)\s*(\d+)', text)
            if date_match:
                month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6}
                month = month_map[date_match.group(2)]
                day = int(date_match.group(3))
                date_text = f"2026-{month:02d}-{day:02d}"
            
            # Opponent (look for @ or vs)
            if '@' in text or 'vs' in text.lower():
                is_home = 'vs' in text.lower() or text.startswith('vs')
                # Extract opponent name
                opp_text = re.sub(r'^(vs\.?|@)\s*', '', text, flags=re.IGNORECASE)
                opp_text = re.sub(r'#\d+\s*', '', opp_text)  # Remove rankings
                opponent_text = opp_text.strip()
            
            # Check for opponent link
            opp_link = cell.find('a', href=re.compile(r'/college-baseball/team/'))
            if opp_link:
                opponent_text = opp_link.get_text(strip=True)
                # Check cell text for @ indicator
                cell_full = cell.get_text()
                is_home = '@' not in cell_full
            
            # Result (W/L with score)
            result_match = re.match(r'([WL])\s*(\d+)-(\d+)', text)
            if result_match:
                result_text = text
        
        if date_text and opponent_text:
            games.append({
                'date': date_text,
                'opponent': opponent_text,
                'is_home': is_home,
                'result': result_text
            })
    
    return games


def fetch_espn_schedule(team_id, espn_id):
    """Fetch schedule from ESPN"""
    url = f"https://www.espn.com/college-baseball/team/schedule/_/id/{espn_id}"
    team_name = BIG_TEN_TEAMS[team_id]['name']
    
    print(f"\n📅 Fetching {team_name} schedule from ESPN (ID: {espn_id})")
    
    html = fetch_page(url)
    if not html:
        return []
    
    games = parse_espn_schedule(html, team_id, espn_id)
    print(f"  ✓ Found {len(games)} games")
    
    return games


def add_team_to_db(team_id, team_name, conference='Big Ten'):
    """Add a team to the database"""
    conn = get_connection()
    c = conn.cursor()
    
    c.execute("""
        INSERT INTO teams (id, name, conference) VALUES (?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET 
            name = COALESCE(excluded.name, teams.name),
            conference = COALESCE(excluded.conference, teams.conference),
            updated_at = CURRENT_TIMESTAMP
    """, (team_id, team_name, conference))
    
    conn.commit()
    conn.close()


def insert_game(date, home_team_id, away_team_id, is_conference=0, time=None, status='scheduled'):
    """Insert a game with INSERT OR IGNORE"""
    conn = get_connection()
    c = conn.cursor()
    
    game_id = f"{date}_{away_team_id}_{home_team_id}"
    
    c.execute("""
        INSERT OR IGNORE INTO games 
        (id, date, time, home_team_id, away_team_id, is_conference_game, status)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (game_id, date, time, home_team_id, away_team_id, is_conference, status))
    
    inserted = c.rowcount > 0
    conn.commit()
    conn.close()
    
    return inserted


def load_big_ten_schedules_hardcoded():
    """Load Big Ten schedules from hardcoded data (more reliable than scraping)"""
    
    # Big Ten 2026 schedules (key games - will be updated with full data)
    # Format: team_id -> [(date, opponent, is_home), ...]
    
    SCHEDULES = {
        'illinois': [
            ('Feb 13', 'Belmont', True, '3:00 PM'),
            ('Feb 14', 'Belmont', True, '1:00 PM'),
            ('Feb 15', 'Belmont', True, '12:00 PM'),
            ('Feb 20', 'Eastern Illinois', True, '3:00 PM'),
            ('Feb 21', 'Eastern Illinois', True, '1:00 PM'),
            ('Feb 22', 'Eastern Illinois', True, '12:00 PM'),
            ('Feb 27', 'UNLV', False, '5:00 PM'),
            ('Feb 28', 'UNLV', False, '5:00 PM'),
            ('Mar 1', 'UNLV', False, '12:00 PM'),
            ('Mar 6', 'Evansville', True, '3:00 PM'),
            ('Mar 7', 'Evansville', True, '1:00 PM'),
            ('Mar 8', 'Evansville', True, '12:00 PM'),
            ('Mar 13', 'Wright State', True, '3:00 PM'),
            ('Mar 14', 'Wright State', True, '1:00 PM'),
            ('Mar 15', 'Wright State', True, '12:00 PM'),
            ('Mar 20', 'Northwestern', True, '3:00 PM'),
            ('Mar 21', 'Northwestern', True, '1:00 PM'),
            ('Mar 22', 'Northwestern', True, '12:00 PM'),
            ('Mar 24', 'Missouri', True, '6:00 PM'),
            ('Mar 27', 'Indiana', False, '5:00 PM'),
            ('Mar 28', 'Indiana', False, '1:00 PM'),
            ('Mar 29', 'Indiana', False, '12:00 PM'),
            ('Apr 3', 'Purdue', True, '3:00 PM'),
            ('Apr 4', 'Purdue', True, '1:00 PM'),
            ('Apr 5', 'Purdue', True, '12:00 PM'),
            ('Apr 10', 'Ohio State', False, '5:00 PM'),
            ('Apr 11', 'Ohio State', False, '1:00 PM'),
            ('Apr 12', 'Ohio State', False, '12:00 PM'),
            ('Apr 17', 'Michigan', True, '6:00 PM'),
            ('Apr 18', 'Michigan', True, '2:00 PM'),
            ('Apr 19', 'Michigan', True, '12:00 PM'),
            ('Apr 24', 'Penn State', False, '5:00 PM'),
            ('Apr 25', 'Penn State', False, '1:00 PM'),
            ('Apr 26', 'Penn State', False, '12:00 PM'),
            ('May 1', 'Maryland', True, '6:00 PM'),
            ('May 2', 'Maryland', True, '2:00 PM'),
            ('May 3', 'Maryland', True, '12:00 PM'),
            ('May 8', 'Nebraska', False, '7:00 PM'),
            ('May 9', 'Nebraska', False, '2:00 PM'),
            ('May 10', 'Nebraska', False, '1:00 PM'),
            ('May 15', 'Rutgers', True, '6:00 PM'),
            ('May 16', 'Rutgers', True, '2:00 PM'),
            ('May 17', 'Rutgers', True, '12:00 PM'),
        ],
        'indiana': [
            ('Feb 13', 'Cal State Northridge', False, '7:00 PM'),
            ('Feb 14', 'Cal State Northridge', False, '4:00 PM'),
            ('Feb 15', 'Cal State Northridge', False, '12:00 PM'),
            ('Feb 20', 'Notre Dame', False, '11:00 AM'),
            ('Feb 21', 'LSU', False, '1:00 PM'),
            ('Feb 22', 'UCF', False, '2:00 PM'),
            ('Feb 27', 'Seattle', True, '4:00 PM'),
            ('Feb 28', 'Seattle', True, '1:00 PM'),
            ('Mar 1', 'Seattle', True, '12:00 PM'),
            ('Mar 6', 'Eastern Michigan', True, '4:00 PM'),
            ('Mar 7', 'Eastern Michigan', True, '1:00 PM'),
            ('Mar 8', 'Eastern Michigan', True, '12:00 PM'),
            ('Mar 13', 'Southern Illinois', True, '4:00 PM'),
            ('Mar 14', 'Southern Illinois', True, '1:00 PM'),
            ('Mar 15', 'Southern Illinois', True, '12:00 PM'),
            ('Mar 17', 'Vanderbilt', False, '6:00 PM'),
            ('Mar 20', 'Maryland', False, '5:00 PM'),
            ('Mar 21', 'Maryland', False, '1:00 PM'),
            ('Mar 22', 'Maryland', False, '12:00 PM'),
            ('Mar 27', 'Illinois', True, '5:00 PM'),
            ('Mar 28', 'Illinois', True, '1:00 PM'),
            ('Mar 29', 'Illinois', True, '12:00 PM'),
            ('Apr 3', 'Rutgers', False, '3:00 PM'),
            ('Apr 4', 'Rutgers', False, '12:00 PM'),
            ('Apr 5', 'Rutgers', False, '11:00 AM'),
            ('Apr 10', 'Michigan State', True, '5:00 PM'),
            ('Apr 11', 'Michigan State', True, '1:00 PM'),
            ('Apr 12', 'Michigan State', True, '12:00 PM'),
            ('Apr 17', 'Iowa', False, '6:00 PM'),
            ('Apr 18', 'Iowa', False, '2:00 PM'),
            ('Apr 19', 'Iowa', False, '1:00 PM'),
            ('Apr 24', 'Michigan', True, '5:00 PM'),
            ('Apr 25', 'Michigan', True, '1:00 PM'),
            ('Apr 26', 'Michigan', True, '12:00 PM'),
            ('May 1', 'Ohio State', False, '5:00 PM'),
            ('May 2', 'Ohio State', False, '1:00 PM'),
            ('May 3', 'Ohio State', False, '12:00 PM'),
            ('May 8', 'Penn State', True, '5:00 PM'),
            ('May 9', 'Penn State', True, '1:00 PM'),
            ('May 10', 'Penn State', True, '12:00 PM'),
            ('May 15', 'Purdue', False, '6:00 PM'),
            ('May 16', 'Purdue', False, '2:00 PM'),
            ('May 17', 'Purdue', False, '1:00 PM'),
        ],
        'iowa': [
            ('Feb 14', 'Southeastern Louisiana', False, '1:00 PM'),
            ('Feb 15', 'New Orleans', False, '11:00 AM'),
            ('Feb 16', 'McNeese', False, '11:00 AM'),
            ('Feb 20', 'Arizona State', False, '6:00 PM'),
            ('Feb 21', 'Arizona State', False, '6:00 PM'),
            ('Feb 22', 'Arizona State', False, '1:00 PM'),
            ('Feb 27', 'West Virginia', False, '12:00 PM'),
            ('Feb 28', 'West Virginia', False, '12:00 PM'),
            ('Mar 1', 'West Virginia', False, '12:00 PM'),
            ('Mar 6', 'Valparaiso', True, '4:00 PM'),
            ('Mar 7', 'Valparaiso', True, '2:00 PM'),
            ('Mar 8', 'Valparaiso', True, '1:00 PM'),
            ('Mar 13', 'Milwaukee', True, '4:00 PM'),
            ('Mar 14', 'Milwaukee', True, '2:00 PM'),
            ('Mar 15', 'Milwaukee', True, '1:00 PM'),
            ('Mar 20', 'Nebraska', False, '7:00 PM'),
            ('Mar 21', 'Nebraska', False, '2:00 PM'),
            ('Mar 22', 'Nebraska', False, '1:00 PM'),
            ('Mar 27', 'Minnesota', True, '6:00 PM'),
            ('Mar 28', 'Minnesota', True, '2:00 PM'),
            ('Mar 29', 'Minnesota', True, '1:00 PM'),
            ('Apr 3', 'Michigan', False, '5:00 PM'),
            ('Apr 4', 'Michigan', False, '1:00 PM'),
            ('Apr 5', 'Michigan', False, '12:00 PM'),
            ('Apr 10', 'Northwestern', True, '6:00 PM'),
            ('Apr 11', 'Northwestern', True, '2:00 PM'),
            ('Apr 12', 'Northwestern', True, '1:00 PM'),
            ('Apr 17', 'Indiana', True, '6:00 PM'),
            ('Apr 18', 'Indiana', True, '2:00 PM'),
            ('Apr 19', 'Indiana', True, '1:00 PM'),
            ('Apr 24', 'Purdue', False, '6:00 PM'),
            ('Apr 25', 'Purdue', False, '2:00 PM'),
            ('Apr 26', 'Purdue', False, '1:00 PM'),
            ('May 1', 'Michigan State', True, '6:00 PM'),
            ('May 2', 'Michigan State', True, '2:00 PM'),
            ('May 3', 'Michigan State', True, '1:00 PM'),
            ('May 8', 'Ohio State', True, '6:00 PM'),
            ('May 9', 'Ohio State', True, '2:00 PM'),
            ('May 10', 'Ohio State', True, '1:00 PM'),
            ('May 15', 'Penn State', False, '5:00 PM'),
            ('May 16', 'Penn State', False, '1:00 PM'),
            ('May 17', 'Penn State', False, '12:00 PM'),
        ],
        'maryland': [
            ('Feb 13', 'Ole Miss', False, '4:00 PM'),
            ('Feb 14', 'Ole Miss', False, '2:00 PM'),
            ('Feb 15', 'Ole Miss', False, '1:00 PM'),
            ('Feb 20', 'Virginia Tech', False, '4:00 PM'),
            ('Feb 21', 'Virginia Tech', False, '1:00 PM'),
            ('Feb 22', 'Virginia Tech', False, '12:00 PM'),
            ('Feb 27', 'UMES', True, '3:00 PM'),
            ('Feb 28', 'UMES', True, '1:00 PM'),
            ('Mar 1', 'UMES', True, '1:00 PM'),
            ('Mar 6', 'Towson', True, '3:00 PM'),
            ('Mar 7', 'Towson', True, '1:00 PM'),
            ('Mar 8', 'Towson', True, '1:00 PM'),
            ('Mar 13', 'Navy', True, '3:00 PM'),
            ('Mar 14', 'Navy', True, '1:00 PM'),
            ('Mar 15', 'Navy', True, '1:00 PM'),
            ('Mar 20', 'Indiana', True, '5:00 PM'),
            ('Mar 21', 'Indiana', True, '1:00 PM'),
            ('Mar 22', 'Indiana', True, '12:00 PM'),
            ('Mar 27', 'Rutgers', False, '3:00 PM'),
            ('Mar 28', 'Rutgers', False, '12:00 PM'),
            ('Mar 29', 'Rutgers', False, '11:00 AM'),
            ('Apr 3', 'Penn State', True, '5:00 PM'),
            ('Apr 4', 'Penn State', True, '1:00 PM'),
            ('Apr 5', 'Penn State', True, '12:00 PM'),
            ('Apr 10', 'Michigan', True, '5:00 PM'),
            ('Apr 11', 'Michigan', True, '1:00 PM'),
            ('Apr 12', 'Michigan', True, '12:00 PM'),
            ('Apr 17', 'Northwestern', False, '5:00 PM'),
            ('Apr 18', 'Northwestern', False, '1:00 PM'),
            ('Apr 19', 'Northwestern', False, '12:00 PM'),
            ('Apr 24', 'Ohio State', True, '5:00 PM'),
            ('Apr 25', 'Ohio State', True, '1:00 PM'),
            ('Apr 26', 'Ohio State', True, '12:00 PM'),
            ('May 1', 'Illinois', False, '6:00 PM'),
            ('May 2', 'Illinois', False, '2:00 PM'),
            ('May 3', 'Illinois', False, '12:00 PM'),
            ('May 8', 'Purdue', True, '5:00 PM'),
            ('May 9', 'Purdue', True, '1:00 PM'),
            ('May 10', 'Purdue', True, '12:00 PM'),
            ('May 15', 'Michigan State', False, '5:00 PM'),
            ('May 16', 'Michigan State', False, '1:00 PM'),
            ('May 17', 'Michigan State', False, '12:00 PM'),
        ],
        'michigan': [
            ('Feb 13', 'Pacific', False, '7:00 PM'),
            ('Feb 14', 'Pacific', False, '4:00 PM'),
            ('Feb 15', 'Pacific', False, '12:00 PM'),
            ('Feb 20', 'Santa Clara', False, '7:00 PM'),
            ('Feb 21', 'Santa Clara', False, '4:00 PM'),
            ('Feb 22', 'Santa Clara', False, '12:00 PM'),
            ('Feb 27', 'Ole Miss', True, '3:05 PM'),
            ('Feb 28', 'Ole Miss', True, '11:05 AM'),
            ('Mar 1', 'Ole Miss', True, '10:05 AM'),
            ('Mar 6', 'Cincinnati', False, '3:00 PM'),
            ('Mar 7', 'Cincinnati', False, '1:00 PM'),
            ('Mar 8', 'Cincinnati', False, '12:00 PM'),
            ('Mar 13', 'Dayton', True, '4:00 PM'),
            ('Mar 14', 'Dayton', True, '2:00 PM'),
            ('Mar 15', 'Dayton', True, '1:00 PM'),
            ('Mar 20', 'Rutgers', True, '4:00 PM'),
            ('Mar 21', 'Rutgers', True, '2:00 PM'),
            ('Mar 22', 'Rutgers', True, '1:00 PM'),
            ('Mar 27', 'Penn State', False, '5:00 PM'),
            ('Mar 28', 'Penn State', False, '1:00 PM'),
            ('Mar 29', 'Penn State', False, '12:00 PM'),
            ('Apr 3', 'Iowa', True, '5:00 PM'),
            ('Apr 4', 'Iowa', True, '1:00 PM'),
            ('Apr 5', 'Iowa', True, '12:00 PM'),
            ('Apr 10', 'Maryland', False, '5:00 PM'),
            ('Apr 11', 'Maryland', False, '1:00 PM'),
            ('Apr 12', 'Maryland', False, '12:00 PM'),
            ('Apr 17', 'Illinois', False, '6:00 PM'),
            ('Apr 18', 'Illinois', False, '2:00 PM'),
            ('Apr 19', 'Illinois', False, '12:00 PM'),
            ('Apr 24', 'Indiana', False, '5:00 PM'),
            ('Apr 25', 'Indiana', False, '1:00 PM'),
            ('Apr 26', 'Indiana', False, '12:00 PM'),
            ('May 1', 'Northwestern', True, '5:00 PM'),
            ('May 2', 'Northwestern', True, '1:00 PM'),
            ('May 3', 'Northwestern', True, '12:00 PM'),
            ('May 8', 'Michigan State', False, '5:00 PM'),
            ('May 9', 'Michigan State', False, '1:00 PM'),
            ('May 10', 'Michigan State', False, '12:00 PM'),
            ('May 15', 'Ohio State', True, '5:00 PM'),
            ('May 16', 'Ohio State', True, '1:00 PM'),
            ('May 17', 'Ohio State', True, '12:00 PM'),
        ],
        'michigan-state': [
            ('Feb 13', 'North Florida', False, '4:00 PM'),
            ('Feb 14', 'North Florida', False, '12:00 PM'),
            ('Feb 15', 'North Florida', False, '12:00 PM'),
            ('Feb 20', 'Texas', False, '6:30 PM'),
            ('Feb 21', 'Texas', False, '2:00 PM'),
            ('Feb 22', 'Texas', False, '12:00 PM'),
            ('Feb 27', 'Jacksonville', False, '6:00 PM'),
            ('Feb 28', 'Jacksonville', False, '3:00 PM'),
            ('Mar 1', 'Jacksonville', False, '12:00 PM'),
            ('Mar 6', 'Toledo', True, '4:00 PM'),
            ('Mar 7', 'Toledo', True, '2:00 PM'),
            ('Mar 8', 'Toledo', True, '1:00 PM'),
            ('Mar 13', 'Central Michigan', True, '4:00 PM'),
            ('Mar 14', 'Central Michigan', True, '2:00 PM'),
            ('Mar 15', 'Central Michigan', True, '1:00 PM'),
            ('Mar 20', 'Purdue', True, '4:00 PM'),
            ('Mar 21', 'Purdue', True, '2:00 PM'),
            ('Mar 22', 'Purdue', True, '1:00 PM'),
            ('Mar 27', 'Ohio State', True, '4:00 PM'),
            ('Mar 28', 'Ohio State', True, '2:00 PM'),
            ('Mar 29', 'Ohio State', True, '1:00 PM'),
            ('Apr 3', 'Nebraska', False, '7:00 PM'),
            ('Apr 4', 'Nebraska', False, '2:00 PM'),
            ('Apr 5', 'Nebraska', False, '1:00 PM'),
            ('Apr 10', 'Indiana', False, '5:00 PM'),
            ('Apr 11', 'Indiana', False, '1:00 PM'),
            ('Apr 12', 'Indiana', False, '12:00 PM'),
            ('Apr 17', 'Penn State', True, '5:00 PM'),
            ('Apr 18', 'Penn State', True, '1:00 PM'),
            ('Apr 19', 'Penn State', True, '12:00 PM'),
            ('Apr 24', 'Northwestern', True, '5:00 PM'),
            ('Apr 25', 'Northwestern', True, '1:00 PM'),
            ('Apr 26', 'Northwestern', True, '12:00 PM'),
            ('May 1', 'Iowa', False, '6:00 PM'),
            ('May 2', 'Iowa', False, '2:00 PM'),
            ('May 3', 'Iowa', False, '1:00 PM'),
            ('May 8', 'Michigan', True, '5:00 PM'),
            ('May 9', 'Michigan', True, '1:00 PM'),
            ('May 10', 'Michigan', True, '12:00 PM'),
            ('May 15', 'Maryland', True, '5:00 PM'),
            ('May 16', 'Maryland', True, '1:00 PM'),
            ('May 17', 'Maryland', True, '12:00 PM'),
        ],
        'minnesota': [
            ('Feb 14', 'Utah Tech', False, '2:00 PM'),
            ('Feb 15', 'Utah Tech', False, '2:00 PM'),
            ('Feb 16', 'Utah Tech', False, '12:00 PM'),
            ('Feb 20', 'New Mexico', False, '7:00 PM'),
            ('Feb 21', 'New Mexico', False, '5:00 PM'),
            ('Feb 22', 'New Mexico', False, '1:00 PM'),
            ('Feb 27', 'UCSB', False, '7:00 PM'),
            ('Feb 28', 'UCSB', False, '4:00 PM'),
            ('Mar 1', 'UCSB', False, '12:00 PM'),
            ('Mar 6', 'San Diego', False, '6:00 PM'),
            ('Mar 7', 'San Diego', False, '4:00 PM'),
            ('Mar 8', 'San Diego', False, '12:00 PM'),
            ('Mar 13', 'UC Riverside', False, '7:00 PM'),
            ('Mar 14', 'UC Riverside', False, '4:00 PM'),
            ('Mar 15', 'UC Riverside', False, '12:00 PM'),
            ('Mar 20', 'Northwestern', False, '3:00 PM'),
            ('Mar 21', 'Northwestern', False, '1:00 PM'),
            ('Mar 22', 'Northwestern', False, '12:00 PM'),
            ('Mar 27', 'Iowa', False, '6:00 PM'),
            ('Mar 28', 'Iowa', False, '2:00 PM'),
            ('Mar 29', 'Iowa', False, '1:00 PM'),
            ('Apr 3', 'Ohio State', True, '5:00 PM'),
            ('Apr 4', 'Ohio State', True, '1:00 PM'),
            ('Apr 5', 'Ohio State', True, '12:00 PM'),
            ('Apr 10', 'Purdue', False, '6:00 PM'),
            ('Apr 11', 'Purdue', False, '2:00 PM'),
            ('Apr 12', 'Purdue', False, '1:00 PM'),
            ('Apr 17', 'Nebraska', True, '6:00 PM'),
            ('Apr 18', 'Nebraska', True, '2:00 PM'),
            ('Apr 19', 'Nebraska', True, '1:00 PM'),
            ('Apr 24', 'Rutgers', True, '5:00 PM'),
            ('Apr 25', 'Rutgers', True, '1:00 PM'),
            ('Apr 26', 'Rutgers', True, '12:00 PM'),
            ('May 1', 'Penn State', False, '5:00 PM'),
            ('May 2', 'Penn State', False, '1:00 PM'),
            ('May 3', 'Penn State', False, '12:00 PM'),
            ('May 8', 'Illinois', True, '6:00 PM'),
            ('May 9', 'Illinois', True, '2:00 PM'),
            ('May 10', 'Illinois', True, '1:00 PM'),
            ('May 15', 'Indiana', True, '5:00 PM'),
            ('May 16', 'Indiana', True, '1:00 PM'),
            ('May 17', 'Indiana', True, '12:00 PM'),
        ],
        'nebraska': [
            ('Feb 13', 'Butler', False, '4:00 PM'),
            ('Feb 14', 'Butler', False, '1:00 PM'),
            ('Feb 15', 'Butler', False, '12:00 PM'),
            ('Feb 20', 'Fresno State', False, '7:00 PM'),
            ('Feb 21', 'Fresno State', False, '4:00 PM'),
            ('Feb 22', 'Fresno State', False, '12:00 PM'),
            ('Feb 27', 'Auburn', False, 'TBA'),
            ('Feb 28', 'Auburn', False, 'TBA'),
            ('Mar 1', 'Auburn', False, 'TBA'),
            ('Mar 6', 'North Dakota State', True, '6:30 PM'),
            ('Mar 7', 'North Dakota State', True, '2:00 PM'),
            ('Mar 8', 'North Dakota State', True, '1:00 PM'),
            ('Mar 13', 'South Dakota State', True, '6:30 PM'),
            ('Mar 14', 'South Dakota State', True, '2:00 PM'),
            ('Mar 15', 'South Dakota State', True, '1:00 PM'),
            ('Mar 20', 'Iowa', True, '7:00 PM'),
            ('Mar 21', 'Iowa', True, '2:00 PM'),
            ('Mar 22', 'Iowa', True, '1:00 PM'),
            ('Mar 27', 'Rutgers', True, '6:30 PM'),
            ('Mar 28', 'Rutgers', True, '2:00 PM'),
            ('Mar 29', 'Rutgers', True, '1:00 PM'),
            ('Apr 3', 'Michigan State', True, '7:00 PM'),
            ('Apr 4', 'Michigan State', True, '2:00 PM'),
            ('Apr 5', 'Michigan State', True, '1:00 PM'),
            ('Apr 10', 'Penn State', False, '5:00 PM'),
            ('Apr 11', 'Penn State', False, '1:00 PM'),
            ('Apr 12', 'Penn State', False, '12:00 PM'),
            ('Apr 17', 'Minnesota', False, '6:00 PM'),
            ('Apr 18', 'Minnesota', False, '2:00 PM'),
            ('Apr 19', 'Minnesota', False, '1:00 PM'),
            ('Apr 24', 'Illinois', True, '6:30 PM'),
            ('Apr 25', 'Illinois', True, '2:00 PM'),
            ('Apr 26', 'Illinois', True, '1:00 PM'),
            ('May 1', 'Northwestern', False, '5:00 PM'),
            ('May 2', 'Northwestern', False, '1:00 PM'),
            ('May 3', 'Northwestern', False, '12:00 PM'),
            ('May 8', 'Illinois', True, '7:00 PM'),
            ('May 9', 'Illinois', True, '2:00 PM'),
            ('May 10', 'Illinois', True, '1:00 PM'),
            ('May 15', 'Ohio State', False, '5:00 PM'),
            ('May 16', 'Ohio State', False, '1:00 PM'),
            ('May 17', 'Ohio State', False, '12:00 PM'),
        ],
        'northwestern': [
            ('Feb 14', 'Chicago State', True, '3:00 PM'),
            ('Feb 15', 'Chicago State', True, '1:00 PM'),
            ('Feb 16', 'Chicago State', True, '1:00 PM'),
            ('Feb 20', 'California Baptist', False, '7:00 PM'),
            ('Feb 21', 'California Baptist', False, '4:00 PM'),
            ('Feb 22', 'California Baptist', False, '12:00 PM'),
            ('Feb 27', 'Coastal Carolina', False, '3:00 PM'),
            ('Feb 28', 'Coastal Carolina', False, '1:00 PM'),
            ('Mar 1', 'Coastal Carolina', False, '11:00 AM'),
            ('Mar 6', 'Bradley', True, '3:00 PM'),
            ('Mar 7', 'Bradley', True, '1:00 PM'),
            ('Mar 8', 'Bradley', True, '1:00 PM'),
            ('Mar 13', 'Northern Illinois', True, '3:00 PM'),
            ('Mar 14', 'Northern Illinois', True, '1:00 PM'),
            ('Mar 15', 'Northern Illinois', True, '1:00 PM'),
            ('Mar 20', 'Illinois', False, '3:00 PM'),
            ('Mar 21', 'Illinois', False, '1:00 PM'),
            ('Mar 22', 'Illinois', False, '12:00 PM'),
            ('Mar 27', 'Minnesota', True, '3:00 PM'),
            ('Mar 28', 'Minnesota', True, '1:00 PM'),
            ('Mar 29', 'Minnesota', True, '12:00 PM'),
            ('Apr 3', 'Purdue', False, '6:00 PM'),
            ('Apr 4', 'Purdue', False, '2:00 PM'),
            ('Apr 5', 'Purdue', False, '1:00 PM'),
            ('Apr 10', 'Iowa', False, '6:00 PM'),
            ('Apr 11', 'Iowa', False, '2:00 PM'),
            ('Apr 12', 'Iowa', False, '1:00 PM'),
            ('Apr 17', 'Maryland', True, '5:00 PM'),
            ('Apr 18', 'Maryland', True, '1:00 PM'),
            ('Apr 19', 'Maryland', True, '12:00 PM'),
            ('Apr 24', 'Michigan State', False, '5:00 PM'),
            ('Apr 25', 'Michigan State', False, '1:00 PM'),
            ('Apr 26', 'Michigan State', False, '12:00 PM'),
            ('May 1', 'Michigan', False, '5:00 PM'),
            ('May 2', 'Michigan', False, '1:00 PM'),
            ('May 3', 'Michigan', False, '12:00 PM'),
            ('May 8', 'Nebraska', True, '5:00 PM'),
            ('May 9', 'Nebraska', True, '1:00 PM'),
            ('May 10', 'Nebraska', True, '12:00 PM'),
            ('May 15', 'Purdue', True, '5:00 PM'),
            ('May 16', 'Purdue', True, '1:00 PM'),
            ('May 17', 'Purdue', True, '12:00 PM'),
        ],
        'ohio-state': [
            ('Feb 14', 'UC San Diego', False, '7:00 PM'),
            ('Feb 15', 'UC San Diego', False, '4:00 PM'),
            ('Feb 16', 'UC San Diego', False, '12:00 PM'),
            ('Feb 20', 'Sacramento State', False, '8:00 PM'),
            ('Feb 21', 'Sacramento State', False, '5:00 PM'),
            ('Feb 22', 'Sacramento State', False, '1:00 PM'),
            ('Feb 27', 'San Jose State', False, '8:00 PM'),
            ('Feb 28', 'San Jose State', False, '5:00 PM'),
            ('Mar 1', 'San Jose State', False, '1:00 PM'),
            ('Mar 6', 'Hawaii', False, '10:35 PM'),
            ('Mar 7', 'Hawaii', False, '10:05 PM'),
            ('Mar 8', 'Hawaii', False, '7:05 PM'),
            ('Mar 13', 'UC Davis', True, '4:05 PM'),
            ('Mar 14', 'UC Davis', True, '2:05 PM'),
            ('Mar 15', 'UC Davis', True, '1:05 PM'),
            ('Mar 20', 'Penn State', True, '4:05 PM'),
            ('Mar 21', 'Penn State', True, '2:05 PM'),
            ('Mar 22', 'Penn State', True, '1:05 PM'),
            ('Mar 27', 'Michigan State', False, '4:00 PM'),
            ('Mar 28', 'Michigan State', False, '2:00 PM'),
            ('Mar 29', 'Michigan State', False, '1:00 PM'),
            ('Apr 3', 'Minnesota', False, '5:00 PM'),
            ('Apr 4', 'Minnesota', False, '1:00 PM'),
            ('Apr 5', 'Minnesota', False, '12:00 PM'),
            ('Apr 10', 'Illinois', True, '5:00 PM'),
            ('Apr 11', 'Illinois', True, '1:00 PM'),
            ('Apr 12', 'Illinois', True, '12:00 PM'),
            ('Apr 17', 'Rutgers', False, '3:00 PM'),
            ('Apr 18', 'Rutgers', False, '12:00 PM'),
            ('Apr 19', 'Rutgers', False, '11:00 AM'),
            ('Apr 24', 'Maryland', False, '5:00 PM'),
            ('Apr 25', 'Maryland', False, '1:00 PM'),
            ('Apr 26', 'Maryland', False, '12:00 PM'),
            ('May 1', 'Indiana', True, '5:00 PM'),
            ('May 2', 'Indiana', True, '1:00 PM'),
            ('May 3', 'Indiana', True, '12:00 PM'),
            ('May 8', 'Iowa', False, '6:00 PM'),
            ('May 9', 'Iowa', False, '2:00 PM'),
            ('May 10', 'Iowa', False, '1:00 PM'),
            ('May 15', 'Michigan', False, '5:00 PM'),
            ('May 16', 'Michigan', False, '1:00 PM'),
            ('May 17', 'Michigan', False, '12:00 PM'),
        ],
        'oregon': [
            ('Feb 14', 'San Diego', True, '6:00 PM'),
            ('Feb 15', 'San Diego', True, '4:00 PM'),
            ('Feb 16', 'San Diego', True, '1:00 PM'),
            ('Feb 20', 'Nevada', True, '6:00 PM'),
            ('Feb 21', 'Nevada', True, '4:00 PM'),
            ('Feb 22', 'Nevada', True, '1:00 PM'),
            ('Feb 27', 'Vanderbilt', True, '1:00 PM'),
            ('Feb 28', 'Arizona', True, '6:00 PM'),
            ('Mar 1', 'UC Irvine', True, '4:00 PM'),
            ('Mar 6', 'Gonzaga', True, '6:00 PM'),
            ('Mar 7', 'Gonzaga', True, '4:00 PM'),
            ('Mar 8', 'Gonzaga', True, '1:00 PM'),
            ('Mar 13', 'Utah', True, '6:00 PM'),
            ('Mar 14', 'Utah', True, '4:00 PM'),
            ('Mar 15', 'Utah', True, '1:00 PM'),
            ('Mar 20', 'Oregon State', False, '7:00 PM'),
            ('Mar 21', 'Oregon State', False, '5:00 PM'),
            ('Mar 22', 'Oregon State', False, '1:00 PM'),
            ('Mar 27', 'UCLA', True, '6:00 PM'),
            ('Mar 28', 'UCLA', True, '4:00 PM'),
            ('Mar 29', 'UCLA', True, '1:00 PM'),
            ('Apr 3', 'Washington', False, '6:00 PM'),
            ('Apr 4', 'Washington', False, '5:00 PM'),
            ('Apr 5', 'Washington', False, '2:00 PM'),
            ('Apr 10', 'USC', True, '6:00 PM'),
            ('Apr 11', 'USC', True, '4:00 PM'),
            ('Apr 12', 'USC', True, '1:00 PM'),
            ('Apr 17', 'UCLA', False, '6:00 PM'),
            ('Apr 18', 'UCLA', False, '5:00 PM'),
            ('Apr 19', 'UCLA', False, '1:00 PM'),
            ('Apr 24', 'Washington', True, '6:00 PM'),
            ('Apr 25', 'Washington', True, '4:00 PM'),
            ('Apr 26', 'Washington', True, '1:00 PM'),
            ('May 1', 'Oregon State', True, '6:00 PM'),
            ('May 2', 'Oregon State', True, '4:00 PM'),
            ('May 3', 'Oregon State', True, '1:00 PM'),
            ('May 8', 'USC', False, '6:00 PM'),
            ('May 9', 'USC', False, '4:00 PM'),
            ('May 10', 'USC', False, '1:00 PM'),
        ],
        'oregon-state': [
            ('Feb 13', 'Oklahoma', False, '6:00 PM'),
            ('Feb 14', 'Oklahoma', False, '2:00 PM'),
            ('Feb 15', 'Oklahoma', False, '1:00 PM'),
            ('Feb 20', 'Grand Canyon', False, '6:00 PM'),
            ('Feb 21', 'Grand Canyon', False, '6:00 PM'),
            ('Feb 22', 'Grand Canyon', False, '12:00 PM'),
            ('Feb 27', 'UC Irvine', True, '6:00 PM'),
            ('Feb 28', 'UC Irvine', True, '5:00 PM'),
            ('Mar 1', 'UC Irvine', True, '1:00 PM'),
            ('Mar 6', 'San Diego State', True, '6:00 PM'),
            ('Mar 7', 'San Diego State', True, '5:00 PM'),
            ('Mar 8', 'San Diego State', True, '1:00 PM'),
            ('Mar 13', 'San Francisco', True, '6:00 PM'),
            ('Mar 14', 'San Francisco', True, '5:00 PM'),
            ('Mar 15', 'San Francisco', True, '1:00 PM'),
            ('Mar 20', 'Oregon', True, '7:00 PM'),
            ('Mar 21', 'Oregon', True, '5:00 PM'),
            ('Mar 22', 'Oregon', True, '1:00 PM'),
            ('Mar 27', 'USC', False, '6:00 PM'),
            ('Mar 28', 'USC', False, '4:00 PM'),
            ('Mar 29', 'USC', False, '1:00 PM'),
            ('Apr 3', 'UCLA', True, '6:00 PM'),
            ('Apr 4', 'UCLA', True, '5:00 PM'),
            ('Apr 5', 'UCLA', True, '1:00 PM'),
            ('Apr 10', 'Washington', False, '6:00 PM'),
            ('Apr 11', 'Washington', False, '5:00 PM'),
            ('Apr 12', 'Washington', False, '2:00 PM'),
            ('Apr 17', 'USC', True, '6:00 PM'),
            ('Apr 18', 'USC', True, '5:00 PM'),
            ('Apr 19', 'USC', True, '1:00 PM'),
            ('Apr 24', 'UCLA', False, '6:00 PM'),
            ('Apr 25', 'UCLA', False, '4:00 PM'),
            ('Apr 26', 'UCLA', False, '1:00 PM'),
            ('May 1', 'Oregon', False, '6:00 PM'),
            ('May 2', 'Oregon', False, '4:00 PM'),
            ('May 3', 'Oregon', False, '1:00 PM'),
            ('May 8', 'Washington', True, '6:00 PM'),
            ('May 9', 'Washington', True, '5:00 PM'),
            ('May 10', 'Washington', True, '1:00 PM'),
        ],
        'penn-state': [
            ('Feb 14', 'High Point', False, '3:00 PM'),
            ('Feb 15', 'High Point', False, '12:00 PM'),
            ('Feb 16', 'High Point', False, '12:00 PM'),
            ('Feb 20', 'James Madison', False, '3:00 PM'),
            ('Feb 21', 'James Madison', False, '1:00 PM'),
            ('Feb 22', 'James Madison', False, '11:00 AM'),
            ('Feb 27', 'Stetson', False, '6:00 PM'),
            ('Feb 28', 'Stetson', False, '3:00 PM'),
            ('Mar 1', 'Stetson', False, '12:00 PM'),
            ('Mar 6', 'Indiana State', True, '3:00 PM'),
            ('Mar 7', 'Indiana State', True, '1:00 PM'),
            ('Mar 8', 'Indiana State', True, '12:00 PM'),
            ('Mar 13', 'Lehigh', True, '3:00 PM'),
            ('Mar 14', 'Lehigh', True, '1:00 PM'),
            ('Mar 15', 'Lehigh', True, '12:00 PM'),
            ('Mar 20', 'Ohio State', False, '4:05 PM'),
            ('Mar 21', 'Ohio State', False, '2:05 PM'),
            ('Mar 22', 'Ohio State', False, '1:05 PM'),
            ('Mar 27', 'Michigan', True, '5:00 PM'),
            ('Mar 28', 'Michigan', True, '1:00 PM'),
            ('Mar 29', 'Michigan', True, '12:00 PM'),
            ('Apr 3', 'Maryland', False, '5:00 PM'),
            ('Apr 4', 'Maryland', False, '1:00 PM'),
            ('Apr 5', 'Maryland', False, '12:00 PM'),
            ('Apr 10', 'Nebraska', True, '5:00 PM'),
            ('Apr 11', 'Nebraska', True, '1:00 PM'),
            ('Apr 12', 'Nebraska', True, '12:00 PM'),
            ('Apr 17', 'Michigan State', False, '5:00 PM'),
            ('Apr 18', 'Michigan State', False, '1:00 PM'),
            ('Apr 19', 'Michigan State', False, '12:00 PM'),
            ('Apr 24', 'Illinois', True, '5:00 PM'),
            ('Apr 25', 'Illinois', True, '1:00 PM'),
            ('Apr 26', 'Illinois', True, '12:00 PM'),
            ('May 1', 'Minnesota', True, '5:00 PM'),
            ('May 2', 'Minnesota', True, '1:00 PM'),
            ('May 3', 'Minnesota', True, '12:00 PM'),
            ('May 8', 'Indiana', False, '5:00 PM'),
            ('May 9', 'Indiana', False, '1:00 PM'),
            ('May 10', 'Indiana', False, '12:00 PM'),
            ('May 15', 'Iowa', True, '5:00 PM'),
            ('May 16', 'Iowa', True, '1:00 PM'),
            ('May 17', 'Iowa', True, '12:00 PM'),
        ],
        'purdue': [
            ('Feb 14', 'Northern Kentucky', False, '1:00 PM'),
            ('Feb 15', 'Northern Kentucky', False, '1:00 PM'),
            ('Feb 16', 'Northern Kentucky', False, '11:00 AM'),
            ('Feb 20', 'Ole Miss', False, '6:30 PM'),
            ('Feb 21', 'Ole Miss', False, '5:00 PM'),
            ('Feb 22', 'Ole Miss', False, '12:00 PM'),
            ('Feb 27', 'Florida Atlantic', False, '5:00 PM'),
            ('Feb 28', 'Florida Atlantic', False, '2:00 PM'),
            ('Mar 1', 'Florida Atlantic', False, '11:00 AM'),
            ('Mar 6', 'Bowling Green', True, '4:00 PM'),
            ('Mar 7', 'Bowling Green', True, '2:00 PM'),
            ('Mar 8', 'Bowling Green', True, '1:00 PM'),
            ('Mar 13', 'Western Michigan', True, '4:00 PM'),
            ('Mar 14', 'Western Michigan', True, '2:00 PM'),
            ('Mar 15', 'Western Michigan', True, '1:00 PM'),
            ('Mar 20', 'Michigan State', False, '4:00 PM'),
            ('Mar 21', 'Michigan State', False, '2:00 PM'),
            ('Mar 22', 'Michigan State', False, '1:00 PM'),
            ('Mar 27', 'Rutgers', False, '3:00 PM'),
            ('Mar 28', 'Rutgers', False, '12:00 PM'),
            ('Mar 29', 'Rutgers', False, '11:00 AM'),
            ('Apr 3', 'Illinois', False, '3:00 PM'),
            ('Apr 4', 'Illinois', False, '1:00 PM'),
            ('Apr 5', 'Illinois', False, '12:00 PM'),
            ('Apr 10', 'Minnesota', True, '6:00 PM'),
            ('Apr 11', 'Minnesota', True, '2:00 PM'),
            ('Apr 12', 'Minnesota', True, '1:00 PM'),
            ('Apr 17', 'Northwestern', True, '5:00 PM'),
            ('Apr 18', 'Northwestern', True, '1:00 PM'),
            ('Apr 19', 'Northwestern', True, '12:00 PM'),
            ('Apr 24', 'Iowa', True, '6:00 PM'),
            ('Apr 25', 'Iowa', True, '2:00 PM'),
            ('Apr 26', 'Iowa', True, '1:00 PM'),
            ('May 1', 'Rutgers', True, '5:00 PM'),
            ('May 2', 'Rutgers', True, '1:00 PM'),
            ('May 3', 'Rutgers', True, '12:00 PM'),
            ('May 8', 'Maryland', False, '5:00 PM'),
            ('May 9', 'Maryland', False, '1:00 PM'),
            ('May 10', 'Maryland', False, '12:00 PM'),
            ('May 15', 'Indiana', True, '6:00 PM'),
            ('May 16', 'Indiana', True, '2:00 PM'),
            ('May 17', 'Indiana', True, '1:00 PM'),
        ],
        'rutgers': [
            ('Feb 14', 'Bethune-Cookman', False, '4:00 PM'),
            ('Feb 15', 'Bethune-Cookman', False, '1:00 PM'),
            ('Feb 16', 'Bethune-Cookman', False, '12:00 PM'),
            ('Feb 20', 'North Florida', False, '6:30 PM'),
            ('Feb 21', 'North Florida', False, '1:00 PM'),
            ('Feb 22', 'North Florida', False, '12:00 PM'),
            ('Feb 27', 'Campbell', False, '4:00 PM'),
            ('Feb 28', 'Campbell', False, '1:00 PM'),
            ('Mar 1', 'Campbell', False, '12:00 PM'),
            ('Mar 6', 'Richmond', True, '3:00 PM'),
            ('Mar 7', 'Richmond', True, '12:00 PM'),
            ('Mar 8', 'Richmond', True, '11:00 AM'),
            ('Mar 13', 'Columbia', True, '3:00 PM'),
            ('Mar 14', 'Columbia', True, '12:00 PM'),
            ('Mar 15', 'Columbia', True, '11:00 AM'),
            ('Mar 20', 'Michigan', False, '4:00 PM'),
            ('Mar 21', 'Michigan', False, '2:00 PM'),
            ('Mar 22', 'Michigan', False, '1:00 PM'),
            ('Mar 27', 'Maryland', True, '3:00 PM'),
            ('Mar 28', 'Maryland', True, '12:00 PM'),
            ('Mar 29', 'Maryland', True, '11:00 AM'),
            ('Apr 3', 'Indiana', True, '3:00 PM'),
            ('Apr 4', 'Indiana', True, '12:00 PM'),
            ('Apr 5', 'Indiana', True, '11:00 AM'),
            ('Apr 10', 'Purdue', True, '3:00 PM'),
            ('Apr 11', 'Purdue', True, '12:00 PM'),
            ('Apr 12', 'Purdue', True, '11:00 AM'),
            ('Apr 17', 'Ohio State', True, '3:00 PM'),
            ('Apr 18', 'Ohio State', True, '12:00 PM'),
            ('Apr 19', 'Ohio State', True, '11:00 AM'),
            ('Apr 24', 'Minnesota', False, '5:00 PM'),
            ('Apr 25', 'Minnesota', False, '1:00 PM'),
            ('Apr 26', 'Minnesota', False, '12:00 PM'),
            ('May 1', 'Purdue', False, '5:00 PM'),
            ('May 2', 'Purdue', False, '1:00 PM'),
            ('May 3', 'Purdue', False, '12:00 PM'),
            ('May 8', 'Nebraska', False, '7:00 PM'),
            ('May 9', 'Nebraska', False, '2:00 PM'),
            ('May 10', 'Nebraska', False, '1:00 PM'),
            ('May 15', 'Illinois', False, '6:00 PM'),
            ('May 16', 'Illinois', False, '2:00 PM'),
            ('May 17', 'Illinois', False, '12:00 PM'),
        ],
        'ucla': [
            ('Feb 14', 'San Francisco', True, '6:00 PM'),
            ('Feb 15', 'San Francisco', True, '2:00 PM'),
            ('Feb 16', 'San Francisco', True, '1:00 PM'),
            ('Feb 20', 'Baylor', False, '4:00 PM'),
            ('Feb 21', 'Baylor', False, '1:00 PM'),
            ('Feb 22', 'Baylor', False, '11:00 AM'),
            ('Feb 27', 'Tennessee', True, '4:00 PM'),
            ('Feb 28', 'Tennessee', True, '12:00 PM'),
            ('Mar 1', 'Tennessee', True, '11:30 AM'),
            ('Mar 3', 'Texas A&M', True, '7:00 PM'),
            ('Mar 4', 'Texas A&M', True, '7:00 PM'),
            ('Mar 5', 'Texas A&M', True, '6:30 PM'),
            ('Mar 7', 'Cal State Fullerton', False, '6:00 PM'),
            ('Mar 8', 'Cal State Fullerton', False, '2:00 PM'),
            ('Mar 9', 'Cal State Fullerton', False, '1:00 PM'),
            ('Mar 14', 'Stanford', True, '6:00 PM'),
            ('Mar 15', 'Stanford', True, '2:00 PM'),
            ('Mar 16', 'Stanford', True, '1:00 PM'),
            ('Mar 20', 'USC', True, '6:00 PM'),
            ('Mar 21', 'USC', True, '5:00 PM'),
            ('Mar 22', 'USC', True, '1:00 PM'),
            ('Mar 27', 'Oregon', False, '6:00 PM'),
            ('Mar 28', 'Oregon', False, '4:00 PM'),
            ('Mar 29', 'Oregon', False, '1:00 PM'),
            ('Apr 3', 'Oregon State', False, '6:00 PM'),
            ('Apr 4', 'Oregon State', False, '5:00 PM'),
            ('Apr 5', 'Oregon State', False, '1:00 PM'),
            ('Apr 10', 'Washington', True, '6:00 PM'),
            ('Apr 11', 'Washington', True, '5:00 PM'),
            ('Apr 12', 'Washington', True, '1:00 PM'),
            ('Apr 17', 'Oregon', True, '6:00 PM'),
            ('Apr 18', 'Oregon', True, '5:00 PM'),
            ('Apr 19', 'Oregon', True, '1:00 PM'),
            ('Apr 24', 'Oregon State', True, '6:00 PM'),
            ('Apr 25', 'Oregon State', True, '4:00 PM'),
            ('Apr 26', 'Oregon State', True, '1:00 PM'),
            ('May 1', 'Washington', False, '6:00 PM'),
            ('May 2', 'Washington', False, '5:00 PM'),
            ('May 3', 'Washington', False, '2:00 PM'),
            ('May 8', 'USC', False, '6:00 PM'),
            ('May 9', 'USC', False, '5:00 PM'),
            ('May 10', 'USC', False, '1:00 PM'),
        ],
        'usc': [
            ('Feb 14', 'BYU', True, '6:00 PM'),
            ('Feb 15', 'BYU', True, '2:00 PM'),
            ('Feb 16', 'BYU', True, '1:00 PM'),
            ('Feb 20', 'Kansas', False, '4:00 PM'),
            ('Feb 21', 'Kansas', False, '4:00 PM'),
            ('Feb 22', 'Kansas', False, '2:00 PM'),
            ('Feb 27', 'Duke', True, '6:00 PM'),
            ('Feb 28', 'Duke', True, '4:00 PM'),
            ('Mar 1', 'Duke', True, '1:00 PM'),
            ('Mar 6', 'Saint Mary\'s', True, '6:00 PM'),
            ('Mar 7', 'Saint Mary\'s', True, '4:00 PM'),
            ('Mar 8', 'Saint Mary\'s', True, '1:00 PM'),
            ('Mar 13', 'Long Beach State', False, '6:00 PM'),
            ('Mar 14', 'Long Beach State', False, '4:00 PM'),
            ('Mar 15', 'Long Beach State', False, '1:00 PM'),
            ('Mar 20', 'UCLA', False, '6:00 PM'),
            ('Mar 21', 'UCLA', False, '5:00 PM'),
            ('Mar 22', 'UCLA', False, '1:00 PM'),
            ('Mar 27', 'Oregon State', True, '6:00 PM'),
            ('Mar 28', 'Oregon State', True, '4:00 PM'),
            ('Mar 29', 'Oregon State', True, '1:00 PM'),
            ('Apr 3', 'Washington', True, '6:00 PM'),
            ('Apr 4', 'Washington', True, '4:00 PM'),
            ('Apr 5', 'Washington', True, '1:00 PM'),
            ('Apr 10', 'Oregon', False, '6:00 PM'),
            ('Apr 11', 'Oregon', False, '4:00 PM'),
            ('Apr 12', 'Oregon', False, '1:00 PM'),
            ('Apr 17', 'Oregon State', False, '6:00 PM'),
            ('Apr 18', 'Oregon State', False, '5:00 PM'),
            ('Apr 19', 'Oregon State', False, '1:00 PM'),
            ('Apr 24', 'Washington', False, '6:00 PM'),
            ('Apr 25', 'Washington', False, '5:00 PM'),
            ('Apr 26', 'Washington', False, '2:00 PM'),
            ('May 1', 'UCLA', True, '6:00 PM'),
            ('May 2', 'UCLA', True, '4:00 PM'),
            ('May 3', 'UCLA', True, '1:00 PM'),
            ('May 8', 'Oregon', True, '6:00 PM'),
            ('May 9', 'Oregon', True, '4:00 PM'),
            ('May 10', 'Oregon', True, '1:00 PM'),
        ],
        'washington': [
            ('Feb 14', 'Santa Clara', True, '6:00 PM'),
            ('Feb 15', 'Santa Clara', True, '2:00 PM'),
            ('Feb 16', 'Santa Clara', True, '1:00 PM'),
            ('Feb 20', 'UC Riverside', True, '5:00 PM'),
            ('Feb 21', 'UC Riverside', True, '2:00 PM'),
            ('Feb 22', 'UC Riverside', True, '1:00 PM'),
            ('Feb 27', 'Texas State', False, '6:00 PM'),
            ('Feb 28', 'Texas State', False, '3:00 PM'),
            ('Mar 1', 'Texas State', False, '1:00 PM'),
            ('Mar 6', 'Cal Poly', True, '6:00 PM'),
            ('Mar 7', 'Cal Poly', True, '2:00 PM'),
            ('Mar 8', 'Cal Poly', True, '1:00 PM'),
            ('Mar 13', 'Sacramento State', True, '6:00 PM'),
            ('Mar 14', 'Sacramento State', True, '2:00 PM'),
            ('Mar 15', 'Sacramento State', True, '1:00 PM'),
            ('Mar 20', 'UCLA', False, '6:00 PM'),
            ('Mar 21', 'UCLA', False, '2:00 PM'),
            ('Mar 22', 'UCLA', False, '1:00 PM'),
            ('Mar 27', 'USC', False, '6:00 PM'),
            ('Mar 28', 'USC', False, '4:00 PM'),
            ('Mar 29', 'USC', False, '1:00 PM'),
            ('Apr 3', 'Oregon', True, '6:00 PM'),
            ('Apr 4', 'Oregon', True, '5:00 PM'),
            ('Apr 5', 'Oregon', True, '2:00 PM'),
            ('Apr 10', 'Oregon State', True, '6:00 PM'),
            ('Apr 11', 'Oregon State', True, '5:00 PM'),
            ('Apr 12', 'Oregon State', True, '2:00 PM'),
            ('Apr 17', 'UCLA', True, '6:00 PM'),
            ('Apr 18', 'UCLA', True, '2:00 PM'),
            ('Apr 19', 'UCLA', True, '1:00 PM'),
            ('Apr 24', 'Oregon', False, '6:00 PM'),
            ('Apr 25', 'Oregon', False, '4:00 PM'),
            ('Apr 26', 'Oregon', False, '1:00 PM'),
            ('May 1', 'UCLA', True, '6:00 PM'),
            ('May 2', 'UCLA', True, '5:00 PM'),
            ('May 3', 'UCLA', True, '2:00 PM'),
            ('May 8', 'Oregon State', False, '6:00 PM'),
            ('May 9', 'Oregon State', False, '5:00 PM'),
            ('May 10', 'Oregon State', False, '1:00 PM'),
        ],
    }
    
    return SCHEDULES


def parse_date(date_str, year=2026):
    """Parse date string like 'Feb 13' or 'Mar 1' into ISO format."""
    date_str = date_str.strip()
    
    date_str = re.sub(r'\([^)]+\)', '', date_str)
    date_str = re.sub(r'^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\.?\s*', '', date_str)
    date_str = date_str.strip()
    
    month_map = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    
    match = re.match(r'([A-Za-z]+)\.?\s+(\d+)', date_str)
    if match:
        month_str, day = match.groups()
        month = month_map.get(month_str.lower()[:3])
        if month:
            return f"{year}-{month:02d}-{int(day):02d}"
    
    return None


def load_schedules_to_db():
    """Load all Big Ten schedules to database"""
    print("\n" + "=" * 60)
    print("  Loading Big Ten Schedules")
    print("=" * 60)
    
    schedules = load_big_ten_schedules_hardcoded()
    
    conn = get_connection()
    cursor = conn.cursor()
    
    games_added = {}
    teams_added = set()
    
    for team_id, games in schedules.items():
        team_games = 0
        for game in games:
            date_str, opponent_name, is_home, time = game
            
            date = parse_date(date_str)
            if not date:
                print(f"  Warning: Could not parse date '{date_str}' for {team_id}")
                continue
            
            opponent_id = normalize_team_name(opponent_name)
            if not opponent_id:
                print(f"  Warning: Could not normalize opponent '{opponent_name}'")
                continue
            
            # Ensure opponent team exists
            cursor.execute("SELECT id FROM teams WHERE id = ?", (opponent_id,))
            if not cursor.fetchone():
                teams_added.add(opponent_id)
                cursor.execute("""
                    INSERT OR IGNORE INTO teams (id, name) VALUES (?, ?)
                """, (opponent_id, opponent_name))
            
            # Determine home/away
            if is_home:
                home_team_id = team_id
                away_team_id = opponent_id
            else:
                home_team_id = opponent_id
                away_team_id = team_id
            
            # Create game ID
            game_id = f"{date}_{away_team_id}_{home_team_id}"
            
            # Check if conference game (both teams in Big Ten)
            is_conference = 1 if opponent_id in BIG_TEN_TEAMS else 0
            
            # Insert game
            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO games 
                    (id, date, time, home_team_id, away_team_id, is_conference_game, status)
                    VALUES (?, ?, ?, ?, ?, ?, 'scheduled')
                """, (game_id, date, time, home_team_id, away_team_id, is_conference))
                
                if cursor.rowcount > 0:
                    team_games += 1
            except sqlite3.IntegrityError:
                pass
        
        games_added[team_id] = team_games
        print(f"  {BIG_TEN_TEAMS[team_id]['name']}: {team_games} games added")
    
    conn.commit()
    
    total_new = sum(games_added.values())
    print(f"\n=== SCHEDULE SUMMARY ===")
    print(f"Teams with new games: {len([k for k,v in games_added.items() if v > 0])}")
    print(f"Total new games added: {total_new}")
    print(f"New opponent teams added: {len(teams_added)}")
    
    conn.close()
    return games_added


def main():
    """Main function to load Big Ten teams, rosters, and schedules"""
    print("\n" + "=" * 70)
    print("  Big Ten Conference Loader")
    print("  18 Teams: rosters + schedules")
    print("=" * 70)
    
    # Step 1: Add all Big Ten teams to database
    print("\n📋 Step 1: Adding Big Ten teams to database...")
    for team_id, info in BIG_TEN_TEAMS.items():
        add_team_to_db(team_id, info['name'], 'Big Ten')
        print(f"  ✓ {info['name']}")
    
    # Step 2: Scrape rosters
    print("\n📋 Step 2: Scraping rosters...")
    total_players = 0
    roster_results = {}
    
    for team_id in BIG_TEN_TEAMS:
        players = scrape_team_roster(team_id)
        saved = save_roster_to_db(team_id, players)
        roster_results[team_id] = saved
        total_players += saved
        time.sleep(1)  # Be respectful to servers
    
    print(f"\n  Total players loaded: {total_players}")
    
    # Step 3: Load schedules
    print("\n📋 Step 3: Loading schedules...")
    load_schedules_to_db()
    
    # Summary
    print("\n" + "=" * 70)
    print("  COMPLETE")
    print("=" * 70)
    
    conn = get_connection()
    c = conn.cursor()
    
    c.execute("SELECT COUNT(*) FROM teams WHERE conference = 'Big Ten'")
    team_count = c.fetchone()[0]
    
    c.execute("""
        SELECT COUNT(*) FROM games 
        WHERE home_team_id IN (SELECT id FROM teams WHERE conference = 'Big Ten')
           OR away_team_id IN (SELECT id FROM teams WHERE conference = 'Big Ten')
    """)
    game_count = c.fetchone()[0]
    
    c.execute("""
        SELECT COUNT(*) FROM player_stats 
        WHERE team_id IN (SELECT id FROM teams WHERE conference = 'Big Ten')
    """)
    player_count = c.fetchone()[0]
    
    print(f"\n  Big Ten teams: {team_count}")
    print(f"  Big Ten players: {player_count}")
    print(f"  Games involving Big Ten: {game_count}")
    
    conn.close()


if __name__ == "__main__":
    main()
