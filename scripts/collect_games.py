#!/usr/bin/env python3
"""
Collect game results and scores from various sources
"""

import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime, timedelta
from pathlib import Path
import re

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
GAMES_DIR = DATA_DIR / "games"
GAMES_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

def load_existing_games():
    """Load all existing game records"""
    games_file = GAMES_DIR / "all_games.json"
    if games_file.exists():
        with open(games_file) as f:
            return json.load(f)
    return {"games": [], "last_updated": None}

def save_games(data):
    """Save games to JSON"""
    data["last_updated"] = datetime.now().isoformat()
    with open(GAMES_DIR / "all_games.json", 'w') as f:
        json.dump(data, f, indent=2)

def add_game(home_team, away_team, home_score, away_score, date, 
             venue=None, conference_game=False, notes=None):
    """Add a game result to the database"""
    
    data = load_existing_games()
    
    game = {
        "id": f"{date}_{away_team}_{home_team}".lower().replace(" ", "-"),
        "date": date,
        "home_team": home_team,
        "away_team": away_team,
        "home_score": int(home_score) if home_score else None,
        "away_score": int(away_score) if away_score else None,
        "winner": home_team if (home_score and away_score and int(home_score) > int(away_score)) 
                  else (away_team if (home_score and away_score and int(away_score) > int(home_score)) else None),
        "venue": venue,
        "conference_game": conference_game,
        "notes": notes,
        "added_at": datetime.now().isoformat()
    }
    
    # Check for duplicates
    existing_ids = [g["id"] for g in data["games"]]
    if game["id"] not in existing_ids:
        data["games"].append(game)
        save_games(data)
        print(f"Added game: {away_team} @ {home_team} ({away_score}-{home_score})")
        return True
    else:
        print(f"Game already exists: {game['id']}")
        return False

def fetch_ncaa_scoreboard(date=None):
    """Fetch scores from NCAA.com scoreboard"""
    if date is None:
        date = datetime.now()
    
    date_str = date.strftime("%Y/%m/%d")
    url = f"https://www.ncaa.com/scoreboard/baseball/d1/{date_str}"
    
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, 'html.parser')
            # Parse scoreboard - structure varies by site
            # This will need refinement based on actual HTML structure
            games = []
            
            # Look for game containers
            game_elements = soup.find_all(class_=re.compile('game|score', re.I))
            
            return {"date": date_str, "source": url, "raw_elements": len(game_elements)}
    except Exception as e:
        print(f"Error fetching scoreboard: {e}")
    
    return None

def get_team_record(team_id):
    """Get win-loss record for a team"""
    data = load_existing_games()
    
    wins = 0
    losses = 0
    
    for game in data["games"]:
        if game.get("winner"):
            if team_id.lower() in game["home_team"].lower() or team_id.lower() in game["away_team"].lower():
                if team_id.lower() in game["winner"].lower():
                    wins += 1
                else:
                    losses += 1
    
    return {"team": team_id, "wins": wins, "losses": losses, "pct": wins/(wins+losses) if (wins+losses) > 0 else 0}

def main():
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "add" and len(sys.argv) >= 7:
            # add home_team away_team home_score away_score date
            add_game(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
        elif sys.argv[1] == "record" and len(sys.argv) >= 3:
            record = get_team_record(sys.argv[2])
            print(f"{record['team']}: {record['wins']}-{record['losses']} ({record['pct']:.3f})")
        elif sys.argv[1] == "fetch":
            result = fetch_ncaa_scoreboard()
            print(json.dumps(result, indent=2) if result else "Failed to fetch")
    else:
        print("Usage:")
        print("  python collect_games.py add <home_team> <away_team> <home_score> <away_score> <date>")
        print("  python collect_games.py record <team_id>")
        print("  python collect_games.py fetch")

if __name__ == "__main__":
    main()
