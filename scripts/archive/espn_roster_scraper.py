#!/usr/bin/env python3
"""
Scrape rosters + basic stats from ESPN for all D1 baseball teams.
ESPN has reliable roster pages for every team.

Usage:
    python3 scripts/espn_roster_scraper.py --team south-alabama
    python3 scripts/espn_roster_scraper.py --conference "Sun Belt"
    python3 scripts/espn_roster_scraper.py --all-missing  # Teams without player stats
"""

import argparse
import json
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict

import requests
from bs4 import BeautifulSoup

DB_PATH = Path(__file__).parent.parent / "data" / "baseball.db"
ESPN_SLUG_MAP = Path(__file__).parent.parent / "config" / "espn_team_ids.json"

# Rate limiting
REQUEST_DELAY = 1.0

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def load_espn_ids():
    """Load ESPN team ID mappings"""
    if ESPN_SLUG_MAP.exists():
        with open(ESPN_SLUG_MAP) as f:
            return json.load(f)
    return {}

def discover_espn_id(team_id: str, team_name: str) -> Optional[str]:
    """Try to discover ESPN team ID from search"""
    # Try common ESPN ID patterns
    search_url = f"https://www.espn.com/college-baseball/team/_/id/{team_id}"
    try:
        resp = requests.get(search_url, timeout=10)
        if resp.status_code == 200 and "roster" in resp.text.lower():
            return team_id
    except:
        pass
    return None

def scrape_espn_roster(team_id: str, espn_id: str = None) -> List[Dict]:
    """Scrape roster from ESPN team page"""
    
    # If no ESPN ID provided, try to use team_id directly
    if not espn_id:
        espn_id = team_id
    
    # ESPN roster URL pattern
    url = f"https://www.espn.com/college-baseball/team/roster/_/id/{espn_id}"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print(f"  Error fetching {url}: {e}")
        return []
    
    soup = BeautifulSoup(resp.text, "html.parser")
    players = []
    
    # Find roster tables
    tables = soup.find_all("table", class_=re.compile(r"Table"))
    
    for table in tables:
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all(["td", "th"])
            if len(cells) < 3:
                continue
            
            # Try to extract player data
            try:
                # ESPN format: Name, #, Pos, B/T, Ht, Wt, Class, Hometown
                name_cell = cells[0]
                name = name_cell.get_text(strip=True)
                
                # Skip header rows
                if name in ["Name", ""] or "Player" in name:
                    continue
                
                player = {"name": name, "team_id": team_id}
                
                # Parse remaining cells based on common patterns
                for i, cell in enumerate(cells[1:], 1):
                    text = cell.get_text(strip=True)
                    
                    # Number (digits only)
                    if text.isdigit() and i <= 2:
                        player["number"] = int(text)
                    # Position
                    elif text in ["P", "C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH", "OF", "INF", "UTL", 
                                  "RHP", "LHP", "RP", "SP", "UTIL", "IF", "PH"]:
                        player["position"] = text
                    # B/T (e.g., "R/R", "L/R", "S/R")
                    elif "/" in text and len(text) <= 4:
                        parts = text.split("/")
                        if len(parts) == 2:
                            player["bats"] = parts[0]
                            player["throws"] = parts[1]
                    # Height (e.g., "6'2"", "6-2")
                    elif "'" in text or (text.count("-") == 1 and len(text) <= 5):
                        player["height"] = text
                    # Weight (e.g., "185", "185 lbs")
                    elif text.replace(" lbs", "").replace(" lb", "").isdigit():
                        player["weight"] = int(text.replace(" lbs", "").replace(" lb", ""))
                    # Class/Year
                    elif text in ["Fr.", "So.", "Jr.", "Sr.", "Gr.", "R-Fr.", "R-So.", "R-Jr.", "R-Sr.",
                                  "FR", "SO", "JR", "SR", "GR", "Freshman", "Sophomore", "Junior", "Senior"]:
                        player["year"] = text
                
                if player.get("name") and player.get("position"):
                    players.append(player)
                    
            except Exception as e:
                continue
    
    return players

def scrape_espn_stats(team_id: str, espn_id: str = None) -> Dict:
    """Scrape batting/pitching stats from ESPN"""
    
    if not espn_id:
        espn_id = team_id
    
    stats_url = f"https://www.espn.com/college-baseball/team/stats/_/id/{espn_id}"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    try:
        resp = requests.get(stats_url, headers=headers, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print(f"  Error fetching stats: {e}")
        return {"batting": {}, "pitching": {}}
    
    soup = BeautifulSoup(resp.text, "html.parser")
    
    batting_stats = {}
    pitching_stats = {}
    
    # Find stats tables and parse them
    tables = soup.find_all("table", class_=re.compile(r"Table"))
    
    current_section = None
    for table in tables:
        # Check section header
        header = table.find_previous(["h2", "h3", "div"], class_=re.compile(r"header|title", re.I))
        if header:
            header_text = header.get_text(strip=True).lower()
            if "batting" in header_text or "hitting" in header_text:
                current_section = "batting"
            elif "pitching" in header_text:
                current_section = "pitching"
        
        rows = table.find_all("tr")
        headers_row = None
        
        for row in rows:
            cells = row.find_all(["td", "th"])
            cell_texts = [c.get_text(strip=True) for c in cells]
            
            # Detect header row
            if any(h in cell_texts for h in ["AB", "AVG", "HR", "ERA", "IP", "W", "Name"]):
                headers_row = cell_texts
                continue
            
            if not headers_row or len(cells) < 3:
                continue
            
            # Get player name
            name = cell_texts[0] if cell_texts else None
            if not name or name in ["Team", "Total", "Totals", ""]:
                continue
            
            # Build stats dict
            player_stats = {}
            for i, val in enumerate(cell_texts[1:], 1):
                if i < len(headers_row):
                    col_name = headers_row[i].lower().replace(".", "")
                    try:
                        # Convert to appropriate type
                        if "." in val:
                            player_stats[col_name] = float(val)
                        elif val.isdigit() or (val.startswith("-") and val[1:].isdigit()):
                            player_stats[col_name] = int(val)
                    except:
                        player_stats[col_name] = val
            
            if current_section == "batting" and player_stats:
                batting_stats[name] = player_stats
            elif current_section == "pitching" and player_stats:
                pitching_stats[name] = player_stats
    
    return {"batting": batting_stats, "pitching": pitching_stats}

def save_players(team_id: str, players: List[Dict], stats: Dict):
    """Save players to database"""
    conn = get_db()
    c = conn.cursor()
    
    batting_stats = stats.get("batting", {})
    pitching_stats = stats.get("pitching", {})
    
    saved = 0
    for player in players:
        name = player.get("name", "")
        
        # Merge with stats
        if name in batting_stats:
            for k, v in batting_stats[name].items():
                # Map ESPN column names to our schema
                col_map = {
                    "ab": "at_bats", "r": "runs", "h": "hits", "2b": "doubles",
                    "3b": "triples", "hr": "home_runs", "rbi": "rbi", "bb": "walks",
                    "so": "strikeouts", "sb": "stolen_bases", "cs": "caught_stealing",
                    "avg": "batting_avg", "obp": "obp", "slg": "slg", "ops": "ops",
                    "gp": "games", "g": "games"
                }
                if k in col_map:
                    player[col_map[k]] = v
        
        if name in pitching_stats:
            for k, v in pitching_stats[name].items():
                col_map = {
                    "era": "era", "w": "wins", "l": "losses", "sv": "saves",
                    "ip": "innings_pitched", "h": "hits_allowed", "r": "runs_allowed",
                    "er": "earned_runs", "bb": "walks_allowed", "so": "strikeouts_pitched",
                    "hr": "home_runs_allowed", "whip": "whip", "gp": "games_pitched",
                    "gs": "games_started", "g": "games_pitched"
                }
                if k in col_map:
                    player[col_map[k]] = v
        
        # Insert/update player
        try:
            # Check if player exists
            c.execute("""
                SELECT id FROM player_stats WHERE team_id = ? AND name = ?
            """, (team_id, name))
            existing = c.fetchone()
            
            if existing:
                # Update existing
                updates = []
                values = []
                for col in ["number", "position", "year", "bats", "throws", "height", "weight",
                           "games", "at_bats", "runs", "hits", "doubles", "triples", "home_runs",
                           "rbi", "walks", "strikeouts", "stolen_bases", "caught_stealing",
                           "batting_avg", "obp", "slg", "ops", "era", "wins", "losses", "saves",
                           "innings_pitched", "hits_allowed", "runs_allowed", "earned_runs",
                           "walks_allowed", "strikeouts_pitched", "whip", "games_pitched", "games_started"]:
                    if col in player and player[col] is not None:
                        updates.append(f"{col} = ?")
                        values.append(player[col])
                
                if updates:
                    values.append(existing["id"])
                    c.execute(f"UPDATE player_stats SET {', '.join(updates)} WHERE id = ?", values)
            else:
                # Insert new
                cols = ["team_id", "name"]
                vals = [team_id, name]
                for col in ["number", "position", "year", "bats", "throws", "height", "weight"]:
                    if col in player:
                        cols.append(col)
                        vals.append(player[col])
                
                c.execute(f"""
                    INSERT INTO player_stats ({', '.join(cols)})
                    VALUES ({', '.join(['?'] * len(vals))})
                """, vals)
            
            saved += 1
        except Exception as e:
            print(f"  Error saving {name}: {e}")
    
    conn.commit()
    conn.close()
    return saved

def get_teams_without_stats():
    """Get list of teams that don't have player stats"""
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
    c.execute("""
        SELECT id, name, conference FROM teams WHERE conference = ?
    """, (conference,))
    teams = [dict(r) for r in c.fetchall()]
    conn.close()
    return teams

def main():
    parser = argparse.ArgumentParser(description="Scrape ESPN rosters")
    parser.add_argument("--team", help="Single team ID to scrape")
    parser.add_argument("--conference", help="Conference to scrape")
    parser.add_argument("--all-missing", action="store_true", help="Scrape all teams without stats")
    parser.add_argument("--dry-run", action="store_true", help="Don't save to database")
    args = parser.parse_args()
    
    espn_ids = load_espn_ids()
    
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
    for team in teams:
        team_id = team["id"]
        espn_id = espn_ids.get(team_id, team_id)
        
        print(f"\n{team['name']} ({team_id})...")
        
        # Scrape roster
        players = scrape_espn_roster(team_id, espn_id)
        print(f"  Found {len(players)} players")
        
        if not players:
            # Try alternative ESPN ID formats
            alt_ids = [
                team_id.replace("-", ""),
                team_id.replace("-", " ").title().replace(" ", ""),
            ]
            for alt in alt_ids:
                players = scrape_espn_roster(team_id, alt)
                if players:
                    print(f"  (using ESPN ID: {alt})")
                    espn_id = alt
                    break
        
        if not players:
            print(f"  Could not find roster")
            continue
        
        # Scrape stats
        stats = scrape_espn_stats(team_id, espn_id)
        print(f"  Batting stats for {len(stats['batting'])} players")
        print(f"  Pitching stats for {len(stats['pitching'])} players")
        
        # Save
        if not args.dry_run:
            saved = save_players(team_id, players, stats)
            total_saved += saved
            print(f"  Saved {saved} players")
        
        time.sleep(REQUEST_DELAY)
    
    print(f"\n=== Done: {total_saved} players saved ===")

if __name__ == "__main__":
    main()
