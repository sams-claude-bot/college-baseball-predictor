#!/usr/bin/env python3
"""
Universal SIDEARM roster scraper for D1 baseball teams.
Most D1 athletic sites use SIDEARM Sports platform.

Usage:
    python3 scripts/sidearm_roster_scraper.py --team south-alabama
    python3 scripts/sidearm_roster_scraper.py --conference "Sun Belt"
    python3 scripts/sidearm_roster_scraper.py --all-missing
"""

import argparse
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Optional, List, Dict

import requests
from bs4 import BeautifulSoup

DB_PATH = Path(__file__).parent.parent / "data" / "baseball.db"
SITES_MAP = Path(__file__).parent.parent / "config" / "team_sites.json"

REQUEST_DELAY = 1.5  # Be polite

# Known athletic site domains
KNOWN_SITES = {
    # Sun Belt
    "south-alabama": "usajaguars.com",
    "troy": "troytrojans.com",
    "coastal-carolina": "goccusports.com",
    "georgia-southern": "gseagles.com",
    "georgia-state": "georgiastatesports.com",
    "appalachian-state": "appstatesports.com",
    "louisiana": "ragincajuns.com",
    "ul-monroe": "ulmwarhawks.com",
    "texas-state": "txstatebobcats.com",
    "arkansas-state": "astateredwolves.com",
    "southern-miss": "southernmiss.com",
    "marshall": "herdzone.com",
    "old-dominion": "odusports.com",
    "james-madison": "jmusports.com",
    
    # C-USA
    "liberty": "libertyflames.com",
    "jacksonville-state": "jsugamecocksports.com",
    "kennesaw-state": "ksuowls.com",
    "new-mexico-state": "nmstatesports.com",
    "sam-houston": "gobearkats.com",
    "middle-tennessee": "goblueraiders.com",
    "western-kentucky": "wkusports.com",
    "fiu": "fiusports.com",
    "louisiana-tech": "latechsports.com",
    
    # AAC
    "tulane": "tulanegreenwave.com",
    "memphis": "gotigersgo.com",
    "east-carolina": "ecupirates.com",
    "charlotte": "charlotte49ers.com",
    "south-florida": "gousfbulls.com",
    "wichita-state": "goshockers.com",
    "rice": "riceowls.com",
    "uab": "uabsports.com",
    "utsa": "gouters.com",
    "north-texas": "meangreensports.com",
    "tulsa": "tulsahurricane.com",
    
    # MVC
    "missouri-state": "missouristatebears.com",
    "dallas-baptist": "dbupatriots.com",
    "indiana-state": "gosycamores.com",
    "illinois-state": "goredbirds.com",
    "southern-illinois": "siusalukis.com",
    "evansville": "gopurpleaces.com",
    "valparaiso": "valpoathletics.com",
    
    # Big East
    "uconn": "uconnhuskies.com",
    "xavier": "goxavier.com",
    "creighton": "gocreighton.com",
    "butler": "butlersports.com",
    "seton-hall": "shupirates.com",
    "st-johns": "redstormsports.com",
    "villanova": "villanova.com",
    "georgetown": "guhoyas.com",
    
    # A-10
    "vcu": "vcuathletics.com",
    "george-mason": "gomason.com",
    "richmond": "richmondspiders.com",
    "george-washington": "gwsports.com",
    "dayton": "daytonflyers.com",
    "saint-louis": "slubillikens.com",
    "fordham": "fordhamsports.com",
    "massachusetts": "umassathletics.com",
    "rhode-island": "gorhody.com",
    "la-salle": "goexplorers.com",
    
    # CAA
    "northeastern": "gonu.com",
    "elon": "elonphoenix.com",
    "william-mary": "tribeathletics.com",
    "hofstra": "gohofstra.com",
    "charleston": "cofcsports.com",
    "towson": "towsontigers.com",
    "uncw": "uncwsports.com",
    "delaware": "bluehens.com",
    "stony-brook": "stonybrookathletics.com",
    
    # WCC
    "gonzaga": "gozags.com",
    "san-diego": "usdtoreros.com",
    "pepperdine": "pepperdinesports.com",
    "loyola-marymount": "lmulions.com",
    "santa-clara": "santaclarabroncos.com",
    "san-francisco": "usfdons.com",
    "pacific": "pacifictigers.com",
    "portland": "portlandpilots.com",
    
    # Mountain West
    "san-diego-state": "goaztecs.com",
    "fresno-state": "gobulldogs.com",
    "nevada": "nevadawolfpack.com",
    "unlv": "unlvrebels.com",
    "new-mexico": "golobos.com",
    "air-force": "goairforcefalcons.com",
    
    # ASUN
    "kennesaw-state": "ksuowls.com",
    "florida-gulf-coast": "fgcuathletics.com",
    "jacksonville": "judolphins.com",
    "stetson": "gohatters.com",
    "lipscomb": "lipscombsports.com",
    "bellarmine": "bellarmineathletics.com",
    "north-florida": "unfospreys.com",
    "central-arkansas": "ucasports.com",
    "eastern-kentucky": "ekusports.com",
    "north-alabama": "roarlions.com",
    "queens": "queensathletics.com",
    "austin-peay": "letsgopeay.com",
    
    # Southland
    "mcneese": "mcneesesports.com",
    "southeastern-louisiana": "lionsports.net",
    "new-orleans": "unoprivateers.com",
    "northwestern-state": "naborssports.com",
    "nicholls": "geauxcolonels.com",
    "houston-christian": "hcuhuskies.com",
    "incarnate-word": "uiwcardinals.com",
    "lamar": "lamarcardinals.com",
    "texas-am-corpus-christi": "goislanders.com",
    
    # Big West
    "uc-irvine": "ucirvinesports.com",
    "cal-state-fullerton": "fullertontitans.com",
    "long-beach-state": "longbeachstate.com",
    "uc-santa-barbara": "ucsbgauchos.com",
    "cal-poly": "gopoly.com",
    "uc-san-diego": "ucsdtritons.com",
    "uc-riverside": "gohighlanders.com",
    "uc-davis": "ucdavisaggies.com",
    "cal-state-bakersfield": "gorunners.com",
    "hawaii": "hawaiiathletics.com",
    
    # Ivy
    "columbia": "gocolumbialions.com",
    "cornell": "cornellbigred.com",
    "harvard": "gocrimson.com",
    "yale": "yalebulldogs.com",
    "penn": "pennathletics.com",
    "princeton": "goprincetontigers.com",
    "brown": "brownbears.com",
    "dartmouth": "dartmouthsports.com",
    
    # Patriot
    "army": "goarmywestpoint.com",
    "navy": "navysports.com",
    "lehigh": "lehighsports.com",
    "bucknell": "bucknellbison.com",
    "holy-cross": "goholycross.com",
    "lafayette": "goleopards.com",
    
    # SWAC
    "grambling-state": "gsutigers.com",
    "southern": "gojagsports.com",
    "jackson-state": "jsutigers.com",
    "alcorn-state": "alcornsports.com",
    "alabama-state": "bamastatesports.com",
    "alabama-am": "aamusports.com",
    "prairie-view": "pvpanthers.com",
    "arkansas-pine-bluff": "uapblionsroar.com",
    "texas-southern": "tsusports.com",
    "bethune-cookman": "bcuathletics.com",
    "florida-am": "famuathletics.com",
}

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def load_team_sites():
    """Load team site mappings"""
    if SITES_MAP.exists():
        with open(SITES_MAP) as f:
            return json.load(f)
    return KNOWN_SITES.copy()

def save_team_sites(sites: Dict):
    """Save team site mappings"""
    SITES_MAP.parent.mkdir(exist_ok=True)
    with open(SITES_MAP, 'w') as f:
        json.dump(sites, f, indent=2)

def scrape_sidearm_roster(team_id: str, domain: str) -> List[Dict]:
    """Scrape roster from SIDEARM-powered site"""
    
    url = f"https://{domain}/sports/baseball/roster"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print(f"  Error: {e}")
        return []
    
    soup = BeautifulSoup(resp.text, 'html.parser')
    players = []
    
    # SIDEARM roster format - find player cards
    roster_items = soup.find_all(['li', 'div', 'article'], class_=re.compile(r'roster|player', re.I))
    
    for item in roster_items:
        player = {"team_id": team_id}
        
        # Try to find name
        name_elem = item.find(['a', 'span', 'div'], class_=re.compile(r'name|title', re.I))
        if name_elem:
            # Get text, clean up
            name = name_elem.get_text(strip=True)
            # Remove jersey number prefix
            name = re.sub(r'^\d+\s*', '', name)
            player["name"] = name
        
        # Get all text content
        text = item.get_text(" ", strip=True)
        
        # Position patterns
        pos_match = re.search(r'\b(RHP|LHP|INF|OF|1B|2B|3B|SS|C|DH|P|UTL|UTIL|IF|RF|CF|LF)(/[A-Z]+)*\b', text)
        if pos_match:
            player["position"] = pos_match.group(0)
        
        # Height pattern (5'10", 6-2, etc.)
        height_match = re.search(r"\b(\d)['\-](\d{1,2})\"?\b", text)
        if height_match:
            player["height"] = f"{height_match.group(1)}'{height_match.group(2)}\""
        
        # Weight pattern
        weight_match = re.search(r'\b(\d{2,3})\s*(?:lbs?|pounds?)\b', text, re.I)
        if weight_match:
            player["weight"] = int(weight_match.group(1))
        
        # Bats/Throws (R/R, L/L, S/R, etc.)
        bt_match = re.search(r'\b([RLSB])/([RL])\b', text)
        if bt_match:
            player["bats"] = bt_match.group(1)
            player["throws"] = bt_match.group(2)
        
        # Year/Class
        year_match = re.search(r'\b(Fr\.?|So\.?|Jr\.?|Sr\.?|Gr\.?|R-Fr\.?|R-So\.?|R-Jr\.?|R-Sr\.?|5th|Freshman|Sophomore|Junior|Senior|Graduate)\b', text, re.I)
        if year_match:
            player["year"] = year_match.group(1)
        
        # Only add if we got a name and position
        if player.get("name") and player.get("position"):
            players.append(player)
    
    # Fallback: try parsing the raw text for structured data
    if not players:
        # Look for roster table
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2:
                    player = {"team_id": team_id}
                    for cell in cells:
                        text = cell.get_text(strip=True)
                        # Try to identify what each cell contains
                        if not player.get("name") and len(text) > 3 and not text.replace(" ", "").isdigit():
                            if not re.match(r'^(RHP|LHP|INF|OF|C|P|IF)$', text):
                                player["name"] = text
                        elif text.isdigit() and int(text) < 100 and "number" not in player:
                            player["number"] = int(text)
                        elif text in ["RHP", "LHP", "INF", "OF", "C", "1B", "2B", "3B", "SS", "P", "DH", "UTL", "IF", "RF", "CF", "LF"]:
                            player["position"] = text
                    
                    if player.get("name") and player.get("position"):
                        players.append(player)
    
    return players

def scrape_sidearm_stats(team_id: str, domain: str) -> Dict:
    """Scrape stats from SIDEARM-powered site"""
    
    url = f"https://{domain}/sports/baseball/stats"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code != 200:
            return {"batting": {}, "pitching": {}}
    except:
        return {"batting": {}, "pitching": {}}
    
    soup = BeautifulSoup(resp.text, 'html.parser')
    batting = {}
    pitching = {}
    
    # Find stats tables
    tables = soup.find_all('table')
    
    for table in tables:
        # Check table header to determine type
        header = table.find('thead')
        if not header:
            continue
        
        header_text = header.get_text().lower()
        is_batting = 'avg' in header_text or 'ab' in header_text or 'hitting' in header_text
        is_pitching = 'era' in header_text or 'ip' in header_text
        
        if not (is_batting or is_pitching):
            continue
        
        # Get column names
        header_cells = header.find_all(['th', 'td'])
        columns = [c.get_text(strip=True).lower() for c in header_cells]
        
        # Parse rows
        rows = table.find_all('tr')
        for row in rows:
            cells = row.find_all('td')
            if len(cells) < 3:
                continue
            
            values = [c.get_text(strip=True) for c in cells]
            if not values[0] or values[0].lower() in ['total', 'totals', 'team']:
                continue
            
            name = values[0]
            stats = {}
            
            for i, col in enumerate(columns[1:], 1):
                if i < len(values):
                    val = values[i]
                    try:
                        if '.' in val:
                            stats[col] = float(val)
                        elif val.isdigit() or (val.startswith('-') and val[1:].isdigit()):
                            stats[col] = int(val)
                    except:
                        pass
            
            if is_batting:
                batting[name] = stats
            elif is_pitching:
                pitching[name] = stats
    
    return {"batting": batting, "pitching": pitching}

def save_players(team_id: str, players: List[Dict], stats: Dict):
    """Save players to database"""
    conn = get_db()
    c = conn.cursor()
    
    batting = stats.get("batting", {})
    pitching = stats.get("pitching", {})
    
    saved = 0
    for player in players:
        name = player.get("name", "")
        if not name:
            continue
        
        # Merge with stats (fuzzy match by name)
        for stat_name in batting:
            if stat_name.lower() in name.lower() or name.lower() in stat_name.lower():
                for k, v in batting[stat_name].items():
                    col_map = {
                        "ab": "at_bats", "r": "runs", "h": "hits", "2b": "doubles",
                        "3b": "triples", "hr": "home_runs", "rbi": "rbi", "bb": "walks",
                        "k": "strikeouts", "so": "strikeouts", "sb": "stolen_bases",
                        "avg": "batting_avg", "obp": "obp", "slg": "slg", "ops": "ops",
                        "gp": "games", "g": "games"
                    }
                    if k in col_map:
                        player[col_map[k]] = v
                break
        
        for stat_name in pitching:
            if stat_name.lower() in name.lower() or name.lower() in stat_name.lower():
                for k, v in pitching[stat_name].items():
                    col_map = {
                        "era": "era", "w": "wins", "l": "losses", "sv": "saves",
                        "ip": "innings_pitched", "h": "hits_allowed", "r": "runs_allowed",
                        "er": "earned_runs", "bb": "walks_allowed", "k": "strikeouts_pitched",
                        "so": "strikeouts_pitched", "whip": "whip", "gp": "games_pitched",
                        "gs": "games_started", "app": "games_pitched"
                    }
                    if k in col_map:
                        player[col_map[k]] = v
                break
        
        try:
            # Check if exists
            c.execute("SELECT id FROM player_stats WHERE team_id = ? AND name = ?", (team_id, name))
            existing = c.fetchone()
            
            if existing:
                # Update
                updates = []
                values = []
                for col in ["number", "position", "year", "bats", "throws", "height", "weight",
                           "games", "at_bats", "runs", "hits", "doubles", "triples", "home_runs",
                           "rbi", "walks", "strikeouts", "stolen_bases", "batting_avg", "obp", 
                           "slg", "ops", "era", "wins", "losses", "saves", "innings_pitched",
                           "strikeouts_pitched", "walks_allowed", "whip", "games_pitched"]:
                    if col in player and player[col] is not None:
                        updates.append(f"{col} = ?")
                        values.append(player[col])
                
                if updates:
                    values.append(existing["id"])
                    c.execute(f"UPDATE player_stats SET {', '.join(updates)} WHERE id = ?", values)
            else:
                # Insert
                cols = ["team_id", "name"]
                vals = [team_id, name]
                for col in ["number", "position", "year", "bats", "throws", "height", "weight"]:
                    if col in player:
                        cols.append(col)
                        vals.append(player[col])
                
                c.execute(f"INSERT INTO player_stats ({', '.join(cols)}) VALUES ({', '.join(['?']*len(vals))})", vals)
            
            saved += 1
        except Exception as e:
            print(f"  Error saving {name}: {e}")
    
    conn.commit()
    conn.close()
    return saved

def get_teams_without_stats():
    """Get teams without player stats"""
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        SELECT t.id, t.name, t.conference
        FROM teams t
        LEFT JOIN player_stats ps ON t.id = ps.team_id
        WHERE ps.id IS NULL
        ORDER BY t.conference, t.name
    """)
    teams = [dict(r) for r in c.fetchall()]
    conn.close()
    return teams

def get_teams_by_conference(conference: str):
    """Get teams in a conference"""
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT id, name, conference FROM teams WHERE conference = ?", (conference,))
    teams = [dict(r) for r in c.fetchall()]
    conn.close()
    return teams

def main():
    parser = argparse.ArgumentParser(description="Scrape SIDEARM rosters")
    parser.add_argument("--team", help="Single team ID")
    parser.add_argument("--conference", help="Conference to scrape")
    parser.add_argument("--all-missing", action="store_true", help="Scrape all teams without stats")
    parser.add_argument("--dry-run", action="store_true", help="Don't save")
    args = parser.parse_args()
    
    sites = load_team_sites()
    
    if args.team:
        teams = [{"id": args.team, "name": args.team}]
    elif args.conference:
        teams = get_teams_by_conference(args.conference)
        print(f"Found {len(teams)} teams in {args.conference}")
    elif args.all_missing:
        teams = get_teams_without_stats()
        print(f"Found {len(teams)} teams without player stats")
    else:
        parser.print_help()
        return
    
    total_saved = 0
    success = 0
    
    for team in teams:
        team_id = team["id"]
        domain = sites.get(team_id)
        
        if not domain:
            # Try common patterns
            slug = team_id.replace("-", "")
            for suffix in ["athletics.com", "sports.com", f"{slug}.com"]:
                test_domain = f"go{slug}.com"
                try:
                    resp = requests.head(f"https://{test_domain}", timeout=5)
                    if resp.ok:
                        domain = test_domain
                        sites[team_id] = domain
                        break
                except:
                    pass
        
        if not domain:
            print(f"{team['name']}: No known site")
            continue
        
        print(f"\n{team['name']} ({domain})...")
        
        players = scrape_sidearm_roster(team_id, domain)
        print(f"  Found {len(players)} players")
        
        if not players:
            continue
        
        stats = scrape_sidearm_stats(team_id, domain)
        print(f"  Batting: {len(stats['batting'])}, Pitching: {len(stats['pitching'])}")
        
        if not args.dry_run:
            saved = save_players(team_id, players, stats)
            total_saved += saved
            print(f"  Saved {saved}")
            success += 1
        
        time.sleep(REQUEST_DELAY)
    
    # Save updated sites map
    save_team_sites(sites)
    
    print(f"\n=== Done: {success} teams, {total_saved} players ===")

if __name__ == "__main__":
    main()
