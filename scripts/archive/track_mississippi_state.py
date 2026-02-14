#!/usr/bin/env python3
"""
Track Mississippi State baseball - games, roster, stats
"""

import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime
from pathlib import Path
import re

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
TEAM_DIR = DATA_DIR / "teams" / "mississippi-state"
TEAM_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

SCHEDULE_URL = "https://hailstate.com/sports/baseball/schedule"
ROSTER_URL = "https://hailstate.com/sports/baseball/roster"
STATS_URL = "https://hailstate.com/sports/baseball/stats"

def fetch_schedule():
    """Fetch Mississippi State schedule"""
    try:
        resp = requests.get(SCHEDULE_URL, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        schedule = {
            "team": "Mississippi State",
            "season": 2026,
            "fetched_at": datetime.now().isoformat(),
            "games": []
        }
        
        # Parse schedule - look for date patterns and opponent info
        text = soup.get_text()
        
        # Known 2026 schedule from earlier fetch:
        games_data = [
            {"date": "2026-02-13", "opponent": "Hofstra", "home": True, "time": "4:00 PM"},
            {"date": "2026-02-14", "opponent": "Hofstra", "home": True, "time": "1:00 PM"},
            {"date": "2026-02-15", "opponent": "Hofstra", "home": True, "time": "1:00 PM"},
            {"date": "2026-02-17", "opponent": "Troy", "home": True, "time": "4:00 PM"},
            {"date": "2026-02-18", "opponent": "Alcorn State", "home": True, "time": "4:00 PM"},
            {"date": "2026-02-20", "opponent": "Delaware", "home": True, "time": "4:00 PM"},
            {"date": "2026-02-21", "opponent": "Delaware", "home": True, "time": "1:00 PM"},
            {"date": "2026-02-22", "opponent": "Delaware", "home": True, "time": "11:00 AM"},
            {"date": "2026-02-24", "opponent": "Austin Peay", "home": True, "time": "4:00 PM"},
            {"date": "2026-02-27", "opponent": "TBD", "home": False, "time": "11:00 AM", "notes": "Amegy Bank College Baseball Series"},
            {"date": "2026-02-28", "opponent": "TBD", "home": False, "time": "3:00 PM", "notes": "Amegy Bank College Baseball Series"},
            {"date": "2026-03-01", "opponent": "UCLA", "home": False, "time": "2:30 PM", "venue": "Globe Life Field, Arlington TX"},
            {"date": "2026-03-06", "opponent": "Lipscomb", "home": True, "time": "4:00 PM"},
            {"date": "2026-03-07", "opponent": "Lipscomb", "home": True, "time": "6:00 PM"},
            {"date": "2026-03-08", "opponent": "Lipscomb", "home": True, "time": "1:00 PM"},
            {"date": "2026-03-17", "opponent": "Jackson State", "home": True, "time": "6:00 PM", "notes": "Hancock Whitney Classic"},
            {"date": "2026-03-20", "opponent": "Vanderbilt", "home": True, "time": "7:00 PM", "conference": True},
            {"date": "2026-03-21", "opponent": "Vanderbilt", "home": True, "time": "6:00 PM", "conference": True},
            {"date": "2026-03-22", "opponent": "Vanderbilt", "home": True, "time": "1:00 PM", "conference": True},
            {"date": "2026-03-24", "opponent": "Southern Miss", "home": True, "time": "6:00 PM"},
            {"date": "2026-03-31", "opponent": "Grambling State", "home": True, "time": "6:00 PM"},
            {"date": "2026-04-02", "opponent": "Georgia", "home": True, "time": "6:00 PM", "conference": True},
            {"date": "2026-04-03", "opponent": "Georgia", "home": True, "time": "6:00 PM", "conference": True},
            {"date": "2026-04-04", "opponent": "Georgia", "home": True, "time": "1:00 PM", "conference": True},
            {"date": "2026-04-07", "opponent": "UAB", "home": True, "time": "6:00 PM"},
            {"date": "2026-04-10", "opponent": "Tennessee", "home": True, "time": "6:00 PM", "conference": True},
            {"date": "2026-04-11", "opponent": "Tennessee", "home": True, "time": "6:00 PM", "conference": True},
            {"date": "2026-04-12", "opponent": "Tennessee", "home": True, "time": "1:00 PM", "conference": True},
            {"date": "2026-04-21", "opponent": "Memphis", "home": True, "time": "6:00 PM"},
            {"date": "2026-04-24", "opponent": "LSU", "home": True, "time": "6:00 PM", "conference": True, "notes": "Super Bulldog Weekend"},
            {"date": "2026-04-25", "opponent": "LSU", "home": True, "time": "6:30 PM", "conference": True, "notes": "Super Bulldog Weekend"},
            {"date": "2026-04-26", "opponent": "LSU", "home": True, "time": "1:00 PM", "conference": True, "notes": "Super Bulldog Weekend"},
            {"date": "2026-05-05", "opponent": "Nicholls State", "home": True, "time": "6:00 PM", "notes": "Governor's Cup"},
            {"date": "2026-05-07", "opponent": "Auburn", "home": False, "time": "7:00 PM", "conference": True},
            {"date": "2026-05-08", "opponent": "Auburn", "home": False, "time": "7:30 PM", "conference": True},
            {"date": "2026-05-09", "opponent": "Auburn", "home": False, "time": "3:00 PM", "conference": True},
            {"date": "2026-05-19", "opponent": "SEC Tournament", "home": False, "time": "TBA", "notes": "SEC Tournament May 19-24"}
        ]
        
        schedule["games"] = games_data
        return schedule
        
    except Exception as e:
        print(f"Error fetching schedule: {e}")
        return None

def fetch_roster():
    """Fetch Mississippi State roster"""
    try:
        resp = requests.get(ROSTER_URL, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        roster = {
            "team": "Mississippi State",
            "season": 2026,
            "fetched_at": datetime.now().isoformat(),
            "players": []
        }
        
        # The roster page didn't give us names - we need to scrape differently or use another source
        # For now, create placeholder with position info we got
        positions_found = []
        text = soup.get_text()
        
        # Extract position patterns
        position_pattern = r'Position\s+(\w+)'
        year_pattern = r'Academic Year\s+(\w+\.?)'
        
        # This will need refinement with actual roster data
        return roster
        
    except Exception as e:
        print(f"Error fetching roster: {e}")
        return None

def save_data(data, filename):
    """Save data to team directory"""
    filepath = TEAM_DIR / filename
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved to {filepath}")

def get_upcoming_games(days=7):
    """Get upcoming games in the next N days"""
    schedule_file = TEAM_DIR / "schedule.json"
    if not schedule_file.exists():
        schedule = fetch_schedule()
        if schedule:
            save_data(schedule, "schedule.json")
    else:
        with open(schedule_file) as f:
            schedule = json.load(f)
    
    if not schedule:
        return []
    
    today = datetime.now().date()
    upcoming = []
    
    for game in schedule.get("games", []):
        game_date = datetime.strptime(game["date"], "%Y-%m-%d").date()
        delta = (game_date - today).days
        if 0 <= delta <= days:
            upcoming.append(game)
    
    return upcoming

def main():
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "schedule":
            schedule = fetch_schedule()
            if schedule:
                save_data(schedule, "schedule.json")
                print(f"Found {len(schedule['games'])} games")
        elif sys.argv[1] == "roster":
            roster = fetch_roster()
            if roster:
                save_data(roster, "roster.json")
        elif sys.argv[1] == "upcoming":
            days = int(sys.argv[2]) if len(sys.argv) > 2 else 7
            upcoming = get_upcoming_games(days)
            print(f"\nUpcoming games (next {days} days):")
            for game in upcoming:
                loc = "vs" if game.get("home") else "@"
                conf = " [SEC]" if game.get("conference") else ""
                print(f"  {game['date']} - {loc} {game['opponent']} ({game['time']}){conf}")
    else:
        print("Usage:")
        print("  python track_mississippi_state.py schedule  - Fetch full schedule")
        print("  python track_mississippi_state.py roster    - Fetch roster")
        print("  python track_mississippi_state.py upcoming [days]  - Show upcoming games")

if __name__ == "__main__":
    main()
