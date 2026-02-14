#!/usr/bin/env python3
"""
Fill roster and schedule gaps for Power 4 teams from athletics websites

Supports multiple site formats:
- Clemson-style: #Number|Name|Height|Weight|Position:|POS|Year:|YR|
- Stanford-style: |Number|Name|Name|...|Year|Height|Weight|Position|
- Sidearm-style: Jersey Number|Number|Name|Position|Year|...
"""

import sys
import re
import time
from pathlib import Path
from datetime import datetime

import requests
from bs4 import BeautifulSoup

_scripts_dir = Path(__file__).parent
# sys.path.insert(0, str(_scripts_dir))  # Removed by cleanup

from scripts.database import get_connection
from scripts.player_stats import add_player, init_player_tables

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

# Athletics site roster URLs
ROSTER_URLS = {
    # ACC teams
    'clemson': 'https://clemsontigers.com/sports/baseball/roster/',
    'georgia-tech': 'https://ramblinwreck.com/sports/m-basebl/roster/',
    'notre-dame': 'https://fightingirish.com/sports/baseball/roster',
    'smu': 'https://smumustangs.com/sports/baseball/roster',
    'stanford': 'https://gostanford.com/sports/baseball/roster',
    'syracuse': 'https://cuse.com/sports/baseball/roster',
    'virginia': 'https://virginiasports.com/sports/baseball/roster',
    'miami-fl': 'https://miamihurricanes.com/sports/baseball/roster',
    # SEC teams
    'south-carolina': 'https://gamecocksonline.com/sports/baseball/roster',
    # Big 12 teams
    'arizona-state': 'https://thesundevils.com/sports/baseball/roster',
    'byu': 'https://byucougars.com/sports/baseball/roster/',
    'cincinnati': 'https://gobearcats.com/sports/baseball/roster',
    'colorado': 'https://cubuffs.com/sports/baseball/roster',
    'iowa-state': 'https://cyclones.com/sports/baseball/roster',
    'ucf': 'https://ucfknights.com/sports/baseball/roster',
    'tcu': 'https://gofrogs.com/sports/baseball/roster',
    # Big Ten teams
    'minnesota': 'https://gophersports.com/sports/baseball/roster',
}


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
    
    # Handle specific long forms first
    pos_map = {
        'right handed pitcher': 'RHP',
        'left handed pitcher': 'LHP',
        'right-handed pitcher': 'RHP',
        'left-handed pitcher': 'LHP',
        'infield': 'INF',
        'outfield': 'OF',
        'pitcher': 'P',
        'catcher': 'C',
        'first base': '1B',
        'second base': '2B',
        'third base': '3B',
        'shortstop': 'SS',
        'infield/outfield': 'INF/OF',
        'outfield/infield': 'OF/INF',
        'catcher/first base': 'C/1B',
        'infield/right handed pitcher': 'INF/RHP',
        'utility': 'UTL',
        'designated hitter': 'DH',
    }
    
    result = pos_str.lower()
    for long_form, abbrev in pos_map.items():
        if long_form in result:
            result = result.replace(long_form, abbrev)
    
    result = result.upper().strip()
    result = re.sub(r'\s*/\s*', '/', result)
    
    return result


def parse_bats_throws(bt_str):
    """Parse bats/throws string like 'R/R' or 'L/R'"""
    if not bt_str:
        return None, None
    
    match = re.search(r'([RLSB])\s*/\s*([RL])', bt_str.upper())
    if match:
        return match.group(1), match.group(2)
    
    return None, None


def parse_clemson_style(all_text, team_id):
    """
    Parse Clemson-style roster format.
    Pattern: #Number|Name|Height|Weight|Position:|POS|Year:|YR|
    """
    players = []
    
    # Pattern: #Number|Name|Height|Weight|Position:|POS|Year:|YR|
    pattern = r'\#(\d{1,2})\|([A-Z][a-zA-Z\'\-\. ]+)\|(\d+\-\d+)\|(\d+)\|Position:\|([A-Z/]+)\|Year:\|([A-Za-z\.\-]+)\|'
    
    for match in re.finditer(pattern, all_text):
        number = int(match.group(1))
        name = match.group(2).strip()
        height = match.group(3)
        weight = int(match.group(4))
        position = normalize_position(match.group(5))
        year = normalize_year(match.group(6))
        
        # Skip non-players
        if len(name) < 3 or any(x in name.lower() for x in ['coach', 'staff', 'iptay']):
            continue
        
        players.append({
            'number': number,
            'name': name,
            'position': position,
            'year': year,
            'height': height,
            'weight': weight
        })
    
    return players


def parse_stanford_style(all_text, team_id):
    """
    Parse Stanford-style roster format.
    Pattern: |Number|Name|Name|...|Year|Height|Weight|Position|
    Uses segment-based parsing.
    """
    players = []
    
    # Split by the number pattern at the start of each player
    segments = re.split(r'\|(\d+)\|([A-Z][a-zA-Z\s\'\-\.]+)\|', all_text)
    
    # Process in groups of 3: prefix, number, name, content
    for i in range(1, len(segments)-2, 3):
        try:
            number = int(segments[i])
            name = segments[i+1].strip()
            content = segments[i+2] if i+2 < len(segments) else ''
            
            # Skip non-player entries
            if 'Roster' in name or 'Open' in name or len(name) < 3:
                continue
            if any(x in name.lower() for x in ['coach', 'staff', 'iptay']):
                continue
            
            # Extract year, height, weight, position from content
            year_match = re.search(r'\|(Freshman|Sophomore|Junior|Senior|Graduate)\|', content)
            height_match = re.search(r'\|(\d+)[′\'](\d+)[″\"]\|', content)
            weight_match = re.search(r'\|(\d+) lbs\|', content)
            pos_match = re.search(r'lbs\|([A-Za-z/\s]+)\|', content)
            
            if not year_match or not pos_match:
                continue
            
            height = None
            if height_match:
                height = f"{height_match.group(1)}' {height_match.group(2)}''"
            
            player = {
                'number': number,
                'name': name,
                'year': normalize_year(year_match.group(1)),
                'height': height,
                'weight': int(weight_match.group(1)) if weight_match else None,
                'position': normalize_position(pos_match.group(1).strip())
            }
            
            players.append(player)
        except (ValueError, IndexError):
            continue
    
    return players


def parse_sidearm_style(all_text, team_id):
    """
    Parse Sidearm Sports roster format.
    Pattern: Jersey Number|Number|Name|Position|Year|...
    """
    players = []
    
    # Method 1: Modern Sidearm with B/T
    pattern_with_bt = r'Jersey Number\|(\d+)\|([^|]+)\|Position\|([^|]+)\|Academic Year\|([^|]+)\|Height\|([^|]+)\|Weight\|([^|]+)\|Custom Field 1\|([^|]+)'
    
    for match in re.finditer(pattern_with_bt, all_text):
        number = int(match.group(1))
        name = match.group(2).strip()
        position = normalize_position(match.group(3).strip())
        year = normalize_year(match.group(4).strip())
        height = match.group(5).strip()
        weight = int(match.group(6)) if match.group(6).strip().isdigit() else None
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
    
    for match in re.finditer(pattern_no_bt, all_text):
        number = int(match.group(1))
        name = match.group(2).strip()
        position = normalize_position(match.group(3).strip())
        year = normalize_year(match.group(4).strip())
        height = match.group(5).strip()
        weight = int(match.group(6)) if match.group(6).strip().isdigit() else None
        
        players.append({
            'name': name,
            'number': number,
            'position': position,
            'year': year,
            'height': height,
            'weight': weight,
        })
    
    return players


def parse_generic_roster(all_text, team_id):
    """
    Generic roster parser - tries to find basic player info.
    Looks for number|name|position patterns.
    """
    players = []
    
    # Generic pattern: Number|Name|Position 
    pattern = r'\|(\d{1,2})\|([A-Z][a-zA-Z\'\-\.\s]+)\|(RHP|LHP|C|1B|2B|3B|SS|OF|INF|UTL|DH|P|IF|[A-Z]+/[A-Z]+)\|'
    
    for match in re.finditer(pattern, all_text):
        number = int(match.group(1))
        name = match.group(2).strip()
        position = normalize_position(match.group(3))
        
        if len(name) < 3 or any(x in name.lower() for x in ['coach', 'staff', 'roster']):
            continue
        
        players.append({
            'number': number,
            'name': name,
            'position': position,
        })
    
    return players


def scrape_roster(team_id):
    """Scrape roster from athletics site using multiple parsing methods"""
    if team_id not in ROSTER_URLS:
        print(f"  ⚠️  No roster URL configured for {team_id}")
        return []
    
    url = ROSTER_URLS[team_id]
    print(f"  📥 Fetching: {url}")
    
    html = fetch_page(url)
    if not html:
        return []
    
    soup = BeautifulSoup(html, 'html.parser')
    all_text = soup.get_text('|', strip=True)
    
    # Try each parsing method in order of specificity
    players = parse_clemson_style(all_text, team_id)
    if len(players) >= 15:
        print(f"  ✓ Clemson-style parser found {len(players)} players")
        return players
    
    players = parse_stanford_style(all_text, team_id)
    if len(players) >= 15:
        print(f"  ✓ Stanford-style parser found {len(players)} players")
        return players
    
    players = parse_sidearm_style(all_text, team_id)
    if len(players) >= 15:
        print(f"  ✓ Sidearm-style parser found {len(players)} players")
        return players
    
    players = parse_generic_roster(all_text, team_id)
    if players:
        print(f"  ✓ Generic parser found {len(players)} players")
        return players
    
    print(f"  ⚠️  No players found with any parser")
    return []


def save_roster_to_db(team_id, players):
    """Save roster to database using INSERT OR IGNORE logic"""
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
    
    for player in players:
        if not player.get('name'):
            continue
        
        name = player['name']
        
        # Skip if already exists
        if name.lower() in existing:
            continue
        
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
            added += 1
        except Exception as e:
            print(f"  ✗ Error saving {name}: {e}")
    
    print(f"  📝 Added {added} new players to database")
    return added


def scrape_team(team_id):
    """Scrape roster for a team"""
    print(f"\n{'='*50}")
    print(f"⚾ {team_id.upper()}")
    print(f"{'='*50}")
    
    players = scrape_roster(team_id)
    
    if players:
        save_roster_to_db(team_id, players)
        
        # Show sample
        print(f"\n  Sample players:")
        for p in players[:5]:
            pos = p.get('position', '?')
            year = p.get('year', '?')
            print(f"    #{p.get('number', '?'):>2} {p.get('name', 'Unknown'):<25} {pos:<8} {year}")
    
    return len(players)


def normalize_team_name(name):
    """Convert team name to standard ID"""
    if not name:
        return None
    
    name_lower = name.lower().strip()
    name_lower = re.sub(r'^#?\d+\s+', '', name_lower)
    name_lower = re.sub(r'^no\.\s*\d+\s+', '', name_lower)
    name_lower = name_lower.strip()
    
    mappings = {
        'arizona': 'arizona',
        'arizona state': 'arizona-state',
        'arizona st.': 'arizona-state',
        'arkansas': 'arkansas',
        'auburn': 'auburn',
        'baylor': 'baylor',
        'brigham young': 'byu',
        'byu': 'byu',
        'cal': 'california',
        'california': 'california',
        'cincinnati': 'cincinnati',
        'clemson': 'clemson',
        'colorado': 'colorado',
        'florida': 'florida',
        'florida state': 'florida-state',
        'fsu': 'florida-state',
        'georgia': 'georgia',
        'georgia tech': 'georgia-tech',
        'houston': 'houston',
        'illinois': 'illinois',
        'indiana': 'indiana',
        'iowa': 'iowa',
        'iowa state': 'iowa-state',
        'iowa st.': 'iowa-state',
        'kansas': 'kansas',
        'kansas state': 'kansas-state',
        'kansas st.': 'kansas-state',
        'kentucky': 'kentucky',
        'lsu': 'lsu',
        'maryland': 'maryland',
        'miami': 'miami-fl',
        'miami (fl)': 'miami-fl',
        'michigan': 'michigan',
        'michigan state': 'michigan-state',
        'minnesota': 'minnesota',
        'mississippi state': 'mississippi-state',
        'miss state': 'mississippi-state',
        'missouri': 'missouri',
        'nebraska': 'nebraska',
        'north carolina': 'north-carolina',
        'unc': 'north-carolina',
        'northwestern': 'northwestern',
        'notre dame': 'notre-dame',
        'ohio state': 'ohio-state',
        'ohio st.': 'ohio-state',
        'oklahoma': 'oklahoma',
        'oklahoma state': 'oklahoma-state',
        'oklahoma st.': 'oklahoma-state',
        'ole miss': 'ole-miss',
        'mississippi': 'ole-miss',
        'oregon': 'oregon',
        'oregon state': 'oregon-state',
        'oregon st.': 'oregon-state',
        'penn state': 'penn-state',
        'penn st.': 'penn-state',
        'purdue': 'purdue',
        'rutgers': 'rutgers',
        'smu': 'smu',
        'south carolina': 'south-carolina',
        'stanford': 'stanford',
        'syracuse': 'syracuse',
        'tcu': 'tcu',
        'tennessee': 'tennessee',
        'texas': 'texas',
        'texas a&m': 'texas-am',
        'texas am': 'texas-am',
        'texas tech': 'texas-tech',
        'ucf': 'ucf',
        'central florida': 'ucf',
        'ucla': 'ucla',
        'usc': 'usc',
        'southern california': 'usc',
        'utah': 'utah',
        'vanderbilt': 'vanderbilt',
        'virginia': 'virginia',
        'uva': 'virginia',
        'virginia tech': 'virginia-tech',
        'wake forest': 'wake-forest',
        'washington': 'washington',
        'west virginia': 'west-virginia',
        'wvu': 'west-virginia',
    }
    
    if name_lower in mappings:
        return mappings[name_lower]
    
    # Try to create an ID from the name
    team_id = re.sub(r'[^a-z0-9]+', '-', name_lower).strip('-')
    return team_id


def main():
    # Teams that need rosters (0 players)
    teams_need_roster = [
        'clemson', 'georgia-tech', 'notre-dame', 'smu', 'stanford', 'syracuse', 'virginia',
        'arizona-state', 'byu', 'cincinnati', 'colorado', 'iowa-state', 'ucf'
    ]
    
    # Teams with partial rosters
    teams_fix_roster = [
        'miami-fl', 'south-carolina', 'tcu', 'minnesota'
    ]
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        
        if cmd == '--all':
            print("\n" + "="*60)
            print("  FILLING ALL ROSTER GAPS")
            print("="*60)
            
            total_added = 0
            for team_id in teams_need_roster + teams_fix_roster:
                count = scrape_team(team_id)
                total_added += count
                time.sleep(1)
            
            print("\n" + "="*60)
            print(f"  COMPLETE: Added players for {len(teams_need_roster + teams_fix_roster)} teams")
            print("="*60)
        
        elif cmd == '--status':
            # Show current roster status
            conn = get_connection()
            c = conn.cursor()
            c.execute("""
                SELECT team_id, COUNT(*) as cnt 
                FROM player_stats 
                WHERE team_id IN ({})
                GROUP BY team_id 
                ORDER BY cnt
            """.format(','.join(['?' for _ in teams_need_roster + teams_fix_roster])), 
                teams_need_roster + teams_fix_roster)
            
            print("\n📊 Roster status:")
            print("-" * 40)
            for row in c.fetchall():
                print(f"  {row['team_id']:<20} {row['cnt']:>3} players")
            conn.close()
        
        elif cmd in ROSTER_URLS:
            # Scrape a specific team
            scrape_team(cmd)
        
        else:
            # Try as team ID
            scrape_team(cmd)
    else:
        print("Usage:")
        print("  python fill_roster_gaps.py --all      - Fill all roster gaps")
        print("  python fill_roster_gaps.py --status   - Show current roster counts")
        print("  python fill_roster_gaps.py <team>     - Fill specific team roster")
        print()
        print("Teams needing rosters:")
        for team_id in teams_need_roster:
            print(f"  - {team_id}")
        print()
        print("Teams needing roster fixes (partial):")
        for team_id in teams_fix_roster:
            print(f"  - {team_id}")


if __name__ == "__main__":
    main()
