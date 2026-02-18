#!/usr/bin/env python3
"""
Scrape box scores from team athletic sites (SideArm Sports platform).

Handles multiple URL patterns:
- SideArm: /sports/baseball/stats/YEAR/OPPONENT/boxscore/ID
- LSU: /boxscore/OPPONENT/

Usage:
    python scrape_team_boxscore.py mississippi-state 2026-02-13
    python scrape_team_boxscore.py --all 2026-02-13
"""

import sys
import re
import sqlite3
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).parent.parent / 'data' / 'baseball.db'
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

# Team site configurations
TEAM_SITES = {
    # SEC
    "alabama": "https://rolltide.com/sports/baseball/schedule",
    "arkansas": "https://arkansasrazorbacks.com/sport/m-basebl/schedule/",
    "auburn": "https://auburntigers.com/sports/baseball/schedule",  # JS-rendered, may not work
    "florida": "https://floridagators.com/sports/baseball/schedule",
    "georgia": "https://georgiadogs.com/sports/baseball/schedule",
    "kentucky": "https://ukathletics.com/sports/baseball/schedule",
    "lsu": "https://lsusports.net/sports/baseball/schedule",  # JS-rendered, may not work
    "mississippi-state": "https://hailstate.com/sports/baseball/schedule",
    "missouri": "https://mutigers.com/sports/baseball/schedule",
    "oklahoma": "https://soonersports.com/sports/baseball/schedule",
    "ole-miss": "https://olemisssports.com/sports/baseball/schedule",
    "south-carolina": "https://gamecocksonline.com/sports/baseball/schedule",
    "tennessee": "https://utsports.com/sports/baseball/schedule",
    "texas": "https://texassports.com/sports/baseball/schedule",
    "texas-am": "https://12thman.com/sports/baseball/schedule",
    "vanderbilt": "https://vucommodores.com/sports/baseball/schedule",
    # Big Ten
    "michigan": "https://mgoblue.com/sports/baseball/schedule",
    "ohio-state": "https://ohiostatebuckeyes.com/sports/baseball/schedule",
    "indiana": "https://iuhoosiers.com/sports/baseball/schedule",
    "illinois": "https://fightingillini.com/sports/baseball/schedule",
    "penn-state": "https://gopsusports.com/sports/baseball/schedule",
    "maryland": "https://umterps.com/sports/baseball/schedule",
    "nebraska": "https://huskers.com/sports/baseball/schedule",
    "iowa": "https://hawkeyesports.com/sports/baseball/schedule",
    "minnesota": "https://gophersports.com/sports/baseball/schedule",
    "michigan-state": "https://msuspartans.com/sports/baseball/schedule",
    "northwestern": "https://nusports.com/sports/baseball/schedule",
    "purdue": "https://purduesports.com/sports/baseball/schedule",
    "rutgers": "https://scarletknights.com/sports/baseball/schedule",
    # ACC
    "clemson": "https://clemsontigers.com/sports/baseball/schedule",
    "duke": "https://goduke.com/sports/baseball/schedule",
    "florida-state": "https://seminoles.com/sports/baseball/schedule",
    "georgia-tech": "https://ramblinwreck.com/sports/baseball/schedule",
    "louisville": "https://gocards.com/sports/baseball/schedule",
    "miami-fl": "https://miamihurricanes.com/sports/baseball/schedule",
    "nc-state": "https://gopack.com/sports/baseball/schedule",
    "north-carolina": "https://goheels.com/sports/baseball/schedule",
    "notre-dame": "https://und.com/sports/baseball/schedule",
    "pittsburgh": "https://pittsburghpanthers.com/sports/baseball/schedule",
    "virginia": "https://virginiasports.com/sports/baseball/schedule",
    "virginia-tech": "https://hokiesports.com/sports/baseball/schedule",
    "wake-forest": "https://godeacs.com/sports/baseball/schedule",
    "boston-college": "https://bceagles.com/sports/baseball/schedule",
    "syracuse": "https://cuse.com/sports/baseball/schedule",
    # Big 12
    "baylor": "https://baylorbears.com/sports/baseball/schedule",
    "tcu": "https://gofrogs.com/sports/baseball/schedule",
    "kansas": "https://kuathletics.com/sports/baseball/schedule",
    "kansas-state": "https://kstatesports.com/sports/baseball/schedule",
    "oklahoma-state": "https://okstate.com/sports/baseball/schedule",
    "west-virginia": "https://wvusports.com/sports/baseball/schedule",
    "texas-tech": "https://texastech.com/sports/baseball/schedule",
    "cincinnati": "https://gobearcats.com/sports/baseball/schedule",
    "ucf": "https://ucfknights.com/sports/baseball/schedule",
    "byu": "https://byucougars.com/sports/baseball/schedule",
    # Pac-12 remnants
    "arizona": "https://arizonawildcats.com/sports/baseball/schedule",
    "arizona-state": "https://thesundevils.com/sports/baseball/schedule",
    "stanford": "https://gostanford.com/sports/baseball/schedule",
    "oregon-state": "https://osubeavers.com/sports/baseball/schedule",
    "ucla": "https://uclabruins.com/sports/baseball/schedule",
    "usc": "https://usctrojans.com/sports/baseball/schedule",
}


def get_boxscore_links(team_id):
    """Get all box score links from team schedule page"""
    if team_id not in TEAM_SITES:
        return []
    
    base_url = TEAM_SITES[team_id]
    domain = '/'.join(base_url.split('/')[:3])
    
    try:
        resp = requests.get(base_url, headers=HEADERS, timeout=30)
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            if 'boxscore' in href.lower():
                # Make absolute URL
                if href.startswith('/'):
                    href = domain + href
                elif not href.startswith('http'):
                    href = domain + '/' + href
                links.append(href)
        
        return list(set(links))  # Dedupe
    except Exception as e:
        print(f"  Error fetching schedule: {e}")
        return []


def parse_sidearm_boxscore(url):
    """Parse box score from SideArm sports site"""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        result = {
            'home_batting': [],
            'away_batting': [],
            'home_pitching': [],
            'away_pitching': [],
        }
        
        tables = soup.find_all('table')
        if len(tables) < 4:
            return None
        
        # Table order: 0=linescore, 1=scoring plays, 2=away batting, 3=home batting, 4=away pitching, 5=home pitching
        # But this can vary - let's detect by headers
        
        batting_tables = []
        pitching_tables = []
        
        for table in tables:
            headers = [th.get_text(strip=True).upper() for th in table.find_all('th')[:10]]
            header_str = ' '.join(headers)
            
            if 'AB' in headers and 'H' in headers and 'RBI' in headers:
                batting_tables.append(table)
            elif 'IP' in headers and ('ER' in headers or 'R' in headers):
                pitching_tables.append(table)
        
        # Parse batting tables
        for i, table in enumerate(batting_tables[:2]):
            key = 'away_batting' if i == 0 else 'home_batting'
            result[key] = parse_batting_table(table)
        
        # Parse pitching tables
        for i, table in enumerate(pitching_tables[:2]):
            key = 'away_pitching' if i == 0 else 'home_pitching'
            result[key] = parse_pitching_table(table)
        
        return result
        
    except Exception as e:
        print(f"  Error parsing boxscore: {e}")
        return None


def parse_batting_table(table):
    """Parse a batting stats table"""
    players = []
    rows = table.find_all('tr')[1:]  # Skip header
    
    for row in rows:
        cells = row.find_all(['td', 'th'])
        if len(cells) < 6:
            continue
        
        # Get player name from first cell (may have link)
        first_cell = cells[0]
        link = first_cell.find('a')
        name = link.get_text(strip=True) if link else first_cell.get_text(strip=True)
        
        # Skip totals row
        if 'total' in name.lower():
            continue
        
        # Get stats - order: Name, Pos, AB, R, H, RBI, BB, SO (varies)
        try:
            # Find AB column (usually 2nd or 3rd)
            stats = [c.get_text(strip=True) for c in cells]
            
            # Try to extract numeric stats
            ab = int(stats[2]) if len(stats) > 2 and stats[2].isdigit() else 0
            runs = int(stats[3]) if len(stats) > 3 and stats[3].isdigit() else 0
            hits = int(stats[4]) if len(stats) > 4 and stats[4].isdigit() else 0
            rbi = int(stats[5]) if len(stats) > 5 and stats[5].isdigit() else 0
            bb = int(stats[6]) if len(stats) > 6 and stats[6].isdigit() else 0
            so = int(stats[7]) if len(stats) > 7 and stats[7].isdigit() else 0
            
            if ab > 0 or bb > 0:  # Only include players who batted
                players.append({
                    'name': name,
                    'ab': ab,
                    'runs': runs,
                    'hits': hits,
                    'rbi': rbi,
                    'bb': bb,
                    'so': so
                })
        except (ValueError, IndexError):
            continue
    
    return players


def parse_pitching_table(table):
    """Parse a pitching stats table"""
    pitchers = []
    rows = table.find_all('tr')[1:]  # Skip header
    
    for row in rows:
        cells = row.find_all(['td', 'th'])
        if len(cells) < 6:
            continue
        
        first_cell = cells[0]
        link = first_cell.find('a')
        name = link.get_text(strip=True) if link else first_cell.get_text(strip=True)
        
        if 'total' in name.lower():
            continue
        
        try:
            stats = [c.get_text(strip=True) for c in cells]
            
            # IP is usually index 1, can be like "4.0" or "4.2" (4 2/3)
            ip_str = stats[1] if len(stats) > 1 else "0"
            ip = parse_innings(ip_str)
            
            hits = int(stats[2]) if len(stats) > 2 and stats[2].isdigit() else 0
            runs = int(stats[3]) if len(stats) > 3 and stats[3].isdigit() else 0
            er = int(stats[4]) if len(stats) > 4 and stats[4].isdigit() else 0
            bb = int(stats[5]) if len(stats) > 5 and stats[5].isdigit() else 0
            so = int(stats[6]) if len(stats) > 6 and stats[6].isdigit() else 0
            
            if ip > 0:
                pitchers.append({
                    'name': name,
                    'ip': ip,
                    'hits': hits,
                    'runs': runs,
                    'er': er,
                    'bb': bb,
                    'so': so
                })
        except (ValueError, IndexError):
            continue
    
    return pitchers


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


def find_boxscore_for_date(team_id, date_str, opponent_id=None):
    """Find the box score URL for a specific game"""
    links = get_boxscore_links(team_id)
    
    # Try to match by date or opponent in URL
    date_parts = date_str.split('-')
    year = date_parts[0]
    
    for link in links:
        # Check if opponent is in URL
        if opponent_id and opponent_id.replace('-', '') in link.lower().replace('-', ''):
            return link
        # Check for date pattern
        if year in link:
            # Could refine this with actual date matching
            pass
    
    # Return all links if no match (caller can try each)
    return links[0] if links else None


def update_player_stats(team_id, batting_stats, pitching_stats, date_str):
    """Update player stats in database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    updated = 0
    
    for batter in batting_stats:
        # Find player by name
        name = batter['name']
        c.execute("""
            SELECT id FROM player_stats 
            WHERE team_id = ? AND (name = ? OR name LIKE ?)
        """, (team_id, name, f"%{name.split()[-1]}%"))
        
        row = c.fetchone()
        if row:
            player_id = row[0]
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
                  datetime.now().isoformat(), player_id))
            updated += 1
    
    for pitcher in pitching_stats:
        name = pitcher['name']
        c.execute("""
            SELECT id FROM player_stats 
            WHERE team_id = ? AND (name = ? OR name LIKE ?)
        """, (team_id, name, f"%{name.split()[-1]}%"))
        
        row = c.fetchone()
        if row:
            player_id = row[0]
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
                  datetime.now().isoformat(), player_id))
            updated += 1
    
    conn.commit()
    conn.close()
    return updated


def scrape_game(team_id, date_str, opponent_id=None, dry_run=False):
    """Scrape box score for a specific game"""
    print(f"\n📊 Scraping {team_id} box score for {date_str}")
    
    # Get box score links
    links = get_boxscore_links(team_id)
    print(f"  Found {len(links)} box score links")
    
    if not links:
        print("  ⊘ No box score links found")
        return False
    
    # Try to find the right game
    target_link = None
    if opponent_id:
        opp_clean = opponent_id.replace('-', '').lower()
        for link in links:
            if opp_clean in link.lower().replace('-', ''):
                target_link = link
                break
    
    if not target_link:
        # Use first link (most recent game)
        target_link = links[0]
    
    print(f"  Fetching: {target_link}")
    
    # Parse box score
    box_score = parse_sidearm_boxscore(target_link)
    
    if not box_score:
        print("  ⊘ Failed to parse box score")
        return False
    
    # Show what we found
    home_bat = len(box_score.get('home_batting', []))
    away_bat = len(box_score.get('away_batting', []))
    home_pitch = len(box_score.get('home_pitching', []))
    away_pitch = len(box_score.get('away_pitching', []))
    
    print(f"  ✓ Parsed: {home_bat} home batters, {away_bat} away batters")
    print(f"           {home_pitch} home pitchers, {away_pitch} away pitchers")
    
    if dry_run:
        print("  [DRY RUN] Would update stats")
        # Show sample
        if box_score.get('home_batting'):
            print(f"  Sample: {box_score['home_batting'][0]}")
        return True
    
    # Update database
    updated = update_player_stats(
        team_id,
        box_score.get('home_batting', []),
        box_score.get('home_pitching', []),
        date_str
    )
    print(f"  ✓ Updated {updated} player records")
    
    return True


def scrape_all_sec(date_str, dry_run=False):
    """Scrape all SEC teams for a given date"""
    print(f"\n{'='*50}")
    print(f"Scraping all SEC box scores for {date_str}")
    print(f"{'='*50}")
    
    success = 0
    failed = 0
    
    for team_id in TEAM_SITES.keys():
        try:
            if scrape_game(team_id, date_str, dry_run=dry_run):
                success += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"Complete: {success} success, {failed} failed")
    return success, failed


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python scrape_team_boxscore.py TEAM_ID DATE")
        print("  python scrape_team_boxscore.py --all DATE")
        print("  python scrape_team_boxscore.py --test TEAM_ID")
        sys.exit(1)
    
    if sys.argv[1] == "--all":
        date_str = sys.argv[2] if len(sys.argv) > 2 else datetime.now().strftime('%Y-%m-%d')
        scrape_all_sec(date_str, dry_run='--dry' in sys.argv)
    elif sys.argv[1] == "--test":
        team_id = sys.argv[2] if len(sys.argv) > 2 else "mississippi-state"
        scrape_game(team_id, datetime.now().strftime('%Y-%m-%d'), dry_run=True)
    else:
        team_id = sys.argv[1]
        date_str = sys.argv[2] if len(sys.argv) > 2 else datetime.now().strftime('%Y-%m-%d')
        opponent = sys.argv[3] if len(sys.argv) > 3 else None
        scrape_game(team_id, date_str, opponent, dry_run='--dry' in sys.argv)
