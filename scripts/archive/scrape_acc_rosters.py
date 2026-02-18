#!/usr/bin/env python3
"""
ACC Roster Scraper

Scrapes rosters for all 17 ACC baseball teams from their athletics websites.
Inserts/updates data in the player_stats table.

Usage:
    python scrape_acc_rosters.py                    # Scrape all teams
    python scrape_acc_rosters.py <team_id>          # Scrape one team
    python scrape_acc_rosters.py --list             # List teams and URLs
    python scrape_acc_rosters.py --status           # Show player counts per team
"""

import sys
import re
import time
import json
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

# ACC Team roster configurations
ACC_ROSTER_URLS = {
    "boston-college": {
        "name": "Boston College",
        "roster_url": "https://bceagles.com/sports/baseball/roster",
        "site_type": "sidearm"
    },
    "california": {
        "name": "California",
        "roster_url": "https://calbears.com/sports/baseball/roster",
        "site_type": "sidearm"
    },
    "clemson": {
        "name": "Clemson",
        "roster_url": "https://clemsontigers.com/sports/baseball/roster",
        "site_type": "sidearm"
    },
    "duke": {
        "name": "Duke",
        "roster_url": "https://goduke.com/sports/baseball/roster",
        "site_type": "sidearm"
    },
    "florida-state": {
        "name": "Florida State",
        "roster_url": "https://seminoles.com/sports/baseball/roster",
        "site_type": "sidearm"
    },
    "georgia-tech": {
        "name": "Georgia Tech",
        "roster_url": "https://ramblinwreck.com/sports/baseball/roster",
        "site_type": "sidearm"
    },
    "louisville": {
        "name": "Louisville",
        "roster_url": "https://gocards.com/sports/baseball/roster",
        "site_type": "sidearm"
    },
    "miami-fl": {
        "name": "Miami",
        "roster_url": "https://miamihurricanes.com/sports/baseball/roster",
        "site_type": "sidearm"
    },
    "nc-state": {
        "name": "NC State",
        "roster_url": "https://gopack.com/sports/baseball/roster",
        "site_type": "sidearm"
    },
    "north-carolina": {
        "name": "North Carolina",
        "roster_url": "https://goheels.com/sports/baseball/roster",
        "site_type": "sidearm"
    },
    "notre-dame": {
        "name": "Notre Dame",
        "roster_url": "https://und.com/sports/baseball/roster",
        "site_type": "sidearm"
    },
    "pittsburgh": {
        "name": "Pittsburgh",
        "roster_url": "https://pittsburghpanthers.com/sports/baseball/roster",
        "site_type": "sidearm"
    },
    "smu": {
        "name": "SMU",
        "roster_url": "https://smumustangs.com/sports/baseball/roster",
        "site_type": "sidearm"
    },
    "stanford": {
        "name": "Stanford",
        "roster_url": "https://gostanford.com/sports/baseball/roster",
        "site_type": "sidearm"
    },
    "syracuse": {
        "name": "Syracuse",
        "roster_url": "https://cuse.com/sports/baseball/roster",
        "site_type": "sidearm"
    },
    "virginia": {
        "name": "Virginia",
        "roster_url": "https://virginiasports.com/sports/baseball/roster",
        "site_type": "sidearm"
    },
    "wake-forest": {
        "name": "Wake Forest",
        "roster_url": "https://godeacs.com/sports/baseball/roster",
        "site_type": "sidearm"
    }
}


def normalize_year(year_str):
    """Normalize academic year to standard format"""
    if not year_str:
        return None
    year_str = year_str.strip()
    
    # Map various year formats
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
    
    # Map long form to abbreviations
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


def parse_height(height_str):
    """Parse height string to standard format like 6' 2''"""
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


def parse_bats_throws(bt_str):
    """Parse bats/throws string like 'R/R' or 'L/R'"""
    if not bt_str:
        return None, None
    
    match = re.search(r'([RLSB])\s*/\s*([RL])', bt_str.upper())
    if match:
        return match.group(1), match.group(2)
    
    return None, None


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
    Parse roster from Sidearm Sports (most ACC teams use this platform).
    Uses multiple parsing strategies to handle different Sidearm versions.
    """
    soup = BeautifulSoup(html, 'html.parser')
    players = []
    
    # Method 1: Modern Sidearm - pipe-delimited text pattern
    text = soup.get_text('|', strip=True)
    
    # Try pattern with Custom Field 1 (B/T)
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
    
    # Try pattern without B/T
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
    
    # Try alternate pattern: Number|Name|Position|Height|Weight|Class
    pos_pattern = r'(RHP|LHP|C|1B|2B|3B|SS|OF|INF|UTL|IF|P|Right-handed Pitcher|Left-handed Pitcher|Catcher|Infielder|Outfielder|Pitcher)'
    alt_pattern = r'\|(\d{1,2})\|([A-Z][a-zA-Z\'\-\s\.]+)\|' + pos_pattern + r'\|(\d+[\-\']\d+)\|(\d+)\s*(?:lbs?\.)?\|(Freshman|Sophomore|Junior|Senior|Graduate|Fr\.|So\.|Jr\.|Sr\.)\|'
    
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
    
    # Method 2: Look for roster cards
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
    
    if players:
        return players
    
    # Method 3: Try table-based roster
    tables = soup.find_all('table', class_=re.compile(r'roster'))
    for table in tables:
        rows = table.find_all('tr')
        for row in rows[1:]:  # Skip header
            cells = row.find_all(['td', 'th'])
            if len(cells) >= 3:
                player = {}
                
                for i, cell in enumerate(cells):
                    text = cell.get_text(strip=True)
                    
                    if i == 0 and re.match(r'^\d{1,2}$', text):
                        player['number'] = int(text)
                    
                    name_link = cell.find('a')
                    if name_link and not player.get('name'):
                        name_text = name_link.get_text(strip=True)
                        if len(name_text) > 3 and ' ' in name_text:
                            player['name'] = name_text
                    
                    if text.upper() in ['RHP', 'LHP', 'C', '1B', '2B', '3B', 'SS', 'OF', 'INF', 'UTL', 'DH']:
                        player['position'] = text.upper()
                
                if player.get('name'):
                    players.append(player)
    
    if players:
        return players
    
    # Method 4: Parse text blocks (last resort)
    pattern = r'#?(\d{1,2})\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\s+(RHP|LHP|C|1B|2B|3B|SS|OF|INF|UTL)'
    matches = re.findall(pattern, text)
    for match in matches:
        players.append({
            'number': int(match[0]),
            'name': match[1],
            'position': match[2]
        })
    
    return players


def scrape_team_roster(team_id):
    """
    Scrape roster for a specific ACC team
    """
    if team_id not in ACC_ROSTER_URLS:
        print(f"  ✗ Unknown team: {team_id}")
        return []
    
    config = ACC_ROSTER_URLS[team_id]
    url = config['roster_url']
    
    print(f"\n📥 Fetching {config['name']} roster from {url}")
    
    html = fetch_page(url)
    if not html:
        return []
    
    players = parse_sidearm_roster(html, team_id)
    
    print(f"  ✓ Found {len(players)} players")
    
    return players


def save_roster_to_db(team_id, players):
    """
    Save/update roster in database
    """
    if not players:
        print(f"  ⊘ No players to save for {team_id}")
        return 0
    
    init_player_tables()
    conn = get_connection()
    c = conn.cursor()
    
    # Get existing players
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


def scrape_all_teams(delay=2.0):
    """
    Scrape rosters for all ACC teams
    """
    print("\n" + "=" * 60)
    print("  ACC Roster Scraper - All Teams")
    print("=" * 60)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "teams": {}
    }
    
    total_players = 0
    
    for team_id in ACC_ROSTER_URLS:
        players = scrape_team_roster(team_id)
        saved = save_roster_to_db(team_id, players)
        
        results["teams"][team_id] = {
            "parsed": len(players),
            "saved": saved
        }
        
        total_players += saved
        time.sleep(delay)  # Be respectful to servers
    
    # Save results log
    log_dir = Path(__file__).parent.parent / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    with open(log_dir / f"acc_roster_scrape_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"  COMPLETE: {total_players} players across {len(ACC_ROSTER_URLS)} teams")
    print("=" * 60)
    
    return results


def show_status():
    """
    Show current player counts per team
    """
    conn = get_connection()
    c = conn.cursor()
    
    c.execute("""
        SELECT team_id, COUNT(*) as count 
        FROM player_stats 
        GROUP BY team_id 
        ORDER BY team_id
    """)
    rows = c.fetchall()
    conn.close()
    
    print("\n📊 Player counts in database:")
    print("-" * 40)
    
    acc_teams = set(ACC_ROSTER_URLS.keys())
    total = 0
    acc_total = 0
    
    for row in rows:
        team_id = row['team_id']
        count = row['count']
        total += count
        
        marker = "✓" if team_id in acc_teams else " "
        if team_id in acc_teams:
            acc_total += count
        print(f"  {marker} {team_id:<20} {count:>3} players")
    
    print("-" * 40)
    print(f"    Total: {total} players")
    print(f"    ACC Total: {acc_total} players")
    
    # Show ACC teams not yet in DB
    existing = {row['team_id'] for row in rows}
    missing = acc_teams - existing
    if missing:
        print(f"\n  ⚠️  Missing ACC teams: {', '.join(sorted(missing))}")


def main():
    if len(sys.argv) < 2:
        # Default: scrape all teams
        scrape_all_teams()
    elif sys.argv[1] == '--list':
        print("\nACC Team Roster URLs:")
        print("-" * 60)
        for team_id, config in sorted(ACC_ROSTER_URLS.items()):
            print(f"  {team_id:<20} {config['roster_url']}")
    elif sys.argv[1] == '--status':
        show_status()
    elif sys.argv[1] in ACC_ROSTER_URLS:
        # Scrape single team
        team_id = sys.argv[1]
        players = scrape_team_roster(team_id)
        save_roster_to_db(team_id, players)
        
        if players:
            print(f"\n  Sample players:")
            for p in players[:5]:
                print(f"    #{p.get('number', '?'):>2} {p.get('name', 'Unknown'):<25} {p.get('position', '?'):<8} {p.get('year', '')}")
    else:
        print(f"Unknown option: {sys.argv[1]}")
        print("\nUsage:")
        print("  python scrape_acc_rosters.py              # Scrape all teams")
        print("  python scrape_acc_rosters.py <team_id>    # Scrape one team")
        print("  python scrape_acc_rosters.py --list       # List team URLs")
        print("  python scrape_acc_rosters.py --status     # Show DB status")


if __name__ == "__main__":
    main()
