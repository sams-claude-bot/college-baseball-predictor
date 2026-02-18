#!/usr/bin/env python3
"""
Track all SEC teams - schedules, results, out-of-conference games
"""

import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime
from pathlib import Path
import re
import time

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
SEC_DIR = DATA_DIR / "teams" / "sec"
SEC_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

# SEC team schedule URLs
SEC_TEAMS = {
    "alabama": {
        "name": "Alabama",
        "schedule_url": "https://rolltide.com/sports/baseball/schedule",
        "division": "west"
    },
    "arkansas": {
        "name": "Arkansas", 
        "schedule_url": "https://arkansasrazorbacks.com/sport/baseball/schedule/",
        "division": "west"
    },
    "auburn": {
        "name": "Auburn",
        "schedule_url": "https://auburntigers.com/sports/baseball/schedule",
        "division": "west"
    },
    "florida": {
        "name": "Florida",
        "schedule_url": "https://floridagators.com/sports/baseball/schedule",
        "division": "east"
    },
    "georgia": {
        "name": "Georgia",
        "schedule_url": "https://georgiadogs.com/sports/baseball/schedule",
        "division": "east"
    },
    "kentucky": {
        "name": "Kentucky",
        "schedule_url": "https://ukathletics.com/sports/baseball/schedule",
        "division": "east"
    },
    "lsu": {
        "name": "LSU",
        "schedule_url": "https://lsusports.net/sports/baseball/schedule",
        "division": "west"
    },
    "mississippi-state": {
        "name": "Mississippi State",
        "schedule_url": "https://hailstate.com/sports/baseball/schedule",
        "division": "west"
    },
    "missouri": {
        "name": "Missouri",
        "schedule_url": "https://mutigers.com/sports/baseball/schedule",
        "division": "east"
    },
    "oklahoma": {
        "name": "Oklahoma",
        "schedule_url": "https://soonersports.com/sports/baseball/schedule",
        "division": "west"
    },
    "ole-miss": {
        "name": "Ole Miss",
        "schedule_url": "https://olemisssports.com/sports/baseball/schedule",
        "division": "west"
    },
    "south-carolina": {
        "name": "South Carolina",
        "schedule_url": "https://gamecocksonline.com/sports/baseball/schedule",
        "division": "east"
    },
    "tennessee": {
        "name": "Tennessee",
        "schedule_url": "https://utsports.com/sports/baseball/schedule",
        "division": "east"
    },
    "texas": {
        "name": "Texas",
        "schedule_url": "https://texassports.com/sports/baseball/schedule",
        "division": "west"
    },
    "texas-am": {
        "name": "Texas A&M",
        "schedule_url": "https://12thman.com/sports/baseball/schedule",
        "division": "east"
    },
    "vanderbilt": {
        "name": "Vanderbilt",
        "schedule_url": "https://vucommodores.com/sport/baseball/schedule/",
        "division": "east"
    }
}

# Known SEC matchups (for identifying conference games)
SEC_TEAM_NAMES = [t["name"].lower() for t in SEC_TEAMS.values()]
SEC_TEAM_NAMES.extend(["ole miss", "miss state", "mississippi state", "texas a&m", "south carolina"])

def is_sec_opponent(opponent):
    """Check if opponent is an SEC team"""
    opp_lower = opponent.lower()
    for name in SEC_TEAM_NAMES:
        if name in opp_lower or opp_lower in name:
            return True
    return False

def fetch_team_schedule(team_id):
    """Fetch schedule for an SEC team"""
    team = SEC_TEAMS.get(team_id)
    if not team:
        print(f"Unknown team: {team_id}")
        return None
    
    try:
        resp = requests.get(team["schedule_url"], headers=HEADERS, timeout=30)
        resp.raise_for_status()
        
        # Basic schedule structure - actual parsing varies by site
        schedule = {
            "team_id": team_id,
            "team_name": team["name"],
            "division": team["division"],
            "schedule_url": team["schedule_url"],
            "fetched_at": datetime.now().isoformat(),
            "games": [],
            "parse_status": "basic"  # Will improve with site-specific parsing
        }
        
        # Try to extract any schedule info from the page
        soup = BeautifulSoup(resp.text, 'html.parser')
        text = soup.get_text()
        
        # Look for date patterns
        date_pattern = r'(Jan|Feb|Mar|Apr|May|Jun)\s+\d{1,2}'
        dates_found = re.findall(date_pattern, text)
        schedule["dates_detected"] = len(dates_found)
        
        return schedule
        
    except Exception as e:
        print(f"Error fetching {team_id}: {e}")
        return None

def fetch_all_sec_schedules():
    """Fetch schedules for all SEC teams"""
    results = {
        "fetched_at": datetime.now().isoformat(),
        "teams": {}
    }
    
    for team_id in SEC_TEAMS:
        print(f"Fetching {team_id}...")
        schedule = fetch_team_schedule(team_id)
        if schedule:
            results["teams"][team_id] = schedule
            
            # Save individual team schedule
            team_file = SEC_DIR / f"{team_id}_schedule.json"
            with open(team_file, 'w') as f:
                json.dump(schedule, f, indent=2)
        
        time.sleep(1)  # Be nice to servers
    
    # Save combined results
    with open(SEC_DIR / "all_schedules.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nFetched {len(results['teams'])} team schedules")
    return results

def get_ooc_games(team_id=None):
    """Get out-of-conference games for SEC teams"""
    games_file = DATA_DIR / "games" / "all_games.json"
    if not games_file.exists():
        return []
    
    with open(games_file) as f:
        data = json.load(f)
    
    ooc_games = []
    for game in data.get("games", []):
        home = game["home_team"]
        away = game["away_team"]
        
        home_sec = is_sec_opponent(home)
        away_sec = is_sec_opponent(away)
        
        # OOC = one SEC team vs one non-SEC team
        if home_sec != away_sec:
            sec_team = home if home_sec else away
            ooc_team = away if home_sec else home
            
            if team_id is None or team_id.lower() in sec_team.lower():
                ooc_games.append({
                    "date": game["date"],
                    "sec_team": sec_team,
                    "opponent": ooc_team,
                    "sec_home": home_sec,
                    "result": game.get("winner")
                })
    
    return ooc_games

def summarize_sec_ooc():
    """Summarize SEC out-of-conference performance"""
    games_file = DATA_DIR / "games" / "all_games.json"
    if not games_file.exists():
        print("No games recorded yet")
        return
    
    with open(games_file) as f:
        data = json.load(f)
    
    sec_ooc_record = {}
    
    for game in data.get("games", []):
        home = game["home_team"]
        away = game["away_team"]
        winner = game.get("winner")
        
        if not winner:
            continue
        
        home_sec = is_sec_opponent(home)
        away_sec = is_sec_opponent(away)
        
        if home_sec != away_sec:
            sec_team = home if home_sec else away
            
            # Normalize team name
            sec_key = sec_team.lower().replace(" ", "-")
            if sec_key not in sec_ooc_record:
                sec_ooc_record[sec_key] = {"wins": 0, "losses": 0, "team": sec_team}
            
            sec_won = winner.lower() in sec_team.lower() or sec_team.lower() in winner.lower()
            if sec_won:
                sec_ooc_record[sec_key]["wins"] += 1
            else:
                sec_ooc_record[sec_key]["losses"] += 1
    
    print("\nSEC Out-of-Conference Records:")
    print("-" * 40)
    
    total_wins = 0
    total_losses = 0
    
    for team_id, record in sorted(sec_ooc_record.items()):
        w, l = record["wins"], record["losses"]
        total_wins += w
        total_losses += l
        pct = w / (w + l) if (w + l) > 0 else 0
        print(f"  {record['team']}: {w}-{l} ({pct:.3f})")
    
    if total_wins + total_losses > 0:
        print("-" * 40)
        print(f"  SEC Total: {total_wins}-{total_losses} ({total_wins/(total_wins+total_losses):.3f})")

def main():
    import sys
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        
        if cmd == "fetch":
            if len(sys.argv) > 2:
                team_id = sys.argv[2]
                schedule = fetch_team_schedule(team_id)
                if schedule:
                    print(f"Fetched {schedule['team_name']}: {schedule['dates_detected']} dates detected")
            else:
                fetch_all_sec_schedules()
        
        elif cmd == "ooc":
            team_id = sys.argv[2] if len(sys.argv) > 2 else None
            games = get_ooc_games(team_id)
            print(f"\nOut-of-Conference Games{' for ' + team_id if team_id else ''}:")
            for g in games:
                loc = "vs" if g["sec_home"] else "@"
                result = f" - {g['result']}" if g.get("result") else ""
                print(f"  {g['date']}: {g['sec_team']} {loc} {g['opponent']}{result}")
        
        elif cmd == "summary":
            summarize_sec_ooc()
        
        else:
            print(f"Unknown command: {cmd}")
    else:
        print("Usage:")
        print("  python track_sec_teams.py fetch [team_id]  - Fetch schedules")
        print("  python track_sec_teams.py ooc [team_id]    - Show OOC games")
        print("  python track_sec_teams.py summary          - SEC OOC record summary")

if __name__ == "__main__":
    main()
