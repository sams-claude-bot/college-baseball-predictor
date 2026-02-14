#!/usr/bin/env python3
"""
Migrate existing JSON data to SQLite database
Also seeds initial data (SEC teams, known tournaments)
"""

import json
from pathlib import Path
from scripts.database import (
    init_database, add_team, add_tournament, add_game, 
    get_connection, DB_PATH
)

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

# SEC Teams
SEC_TEAMS = [
    {"id": "alabama", "name": "Alabama", "nickname": "Crimson Tide", "division": "west"},
    {"id": "arkansas", "name": "Arkansas", "nickname": "Razorbacks", "division": "west"},
    {"id": "auburn", "name": "Auburn", "nickname": "Tigers", "division": "west"},
    {"id": "florida", "name": "Florida", "nickname": "Gators", "division": "east"},
    {"id": "georgia", "name": "Georgia", "nickname": "Bulldogs", "division": "east"},
    {"id": "kentucky", "name": "Kentucky", "nickname": "Wildcats", "division": "east"},
    {"id": "lsu", "name": "LSU", "nickname": "Tigers", "division": "west"},
    {"id": "mississippi-state", "name": "Mississippi State", "nickname": "Bulldogs", "division": "west"},
    {"id": "missouri", "name": "Missouri", "nickname": "Tigers", "division": "east"},
    {"id": "oklahoma", "name": "Oklahoma", "nickname": "Sooners", "division": "west"},
    {"id": "ole-miss", "name": "Ole Miss", "nickname": "Rebels", "division": "west"},
    {"id": "south-carolina", "name": "South Carolina", "nickname": "Gamecocks", "division": "east"},
    {"id": "tennessee", "name": "Tennessee", "nickname": "Volunteers", "division": "east"},
    {"id": "texas", "name": "Texas", "nickname": "Longhorns", "division": "west"},
    {"id": "texas-am", "name": "Texas A&M", "nickname": "Aggies", "division": "east"},
    {"id": "vanderbilt", "name": "Vanderbilt", "nickname": "Commodores", "division": "east"},
]

# Known 2026 Tournaments
TOURNAMENTS_2026 = [
    {
        "id": "amegy-bank-2026",
        "name": "Amegy Bank College Baseball Series",
        "location": "Arlington, Texas",
        "venue": "Globe Life Field",
        "start_date": "2026-02-27",
        "end_date": "2026-03-01",
        "teams": ["mississippi-state", "ucla", "tbd1", "tbd2"],
        "notes": "Round-robin tournament at Globe Life Field"
    },
    {
        "id": "hancock-whitney-2026",
        "name": "Hancock Whitney Classic",
        "location": "Starkville, MS",
        "venue": "Dudy Noble Field",
        "start_date": "2026-03-17",
        "end_date": "2026-03-17",
        "teams": ["mississippi-state", "jackson-state"],
        "notes": "Single game event"
    },
    {
        "id": "sec-tournament-2026",
        "name": "SEC Baseball Tournament",
        "location": "Hoover, AL",
        "venue": "Hoover Metropolitan Stadium",
        "start_date": "2026-05-19",
        "end_date": "2026-05-24",
        "teams": SEC_TEAMS,
        "notes": "Conference tournament"
    },
    {
        "id": "super-bulldog-weekend-2026",
        "name": "Super Bulldog Weekend",
        "location": "Starkville, MS",
        "venue": "Dudy Noble Field",
        "start_date": "2026-04-24",
        "end_date": "2026-04-26",
        "teams": ["mississippi-state", "lsu"],
        "notes": "Annual rivalry series with special events"
    },
]

def seed_sec_teams():
    """Add all SEC teams to database"""
    print("\n📍 Seeding SEC teams...")
    for team in SEC_TEAMS:
        add_team(
            team_id=team["id"],
            name=team["name"],
            nickname=team["nickname"],
            conference="SEC",
            division=team["division"]
        )
        print(f"  ✓ {team['name']}")
    print(f"  Added {len(SEC_TEAMS)} SEC teams")

def seed_tournaments():
    """Add known 2026 tournaments"""
    print("\n🏆 Seeding tournaments...")
    for t in TOURNAMENTS_2026:
        team_ids = [team["id"] if isinstance(team, dict) else team for team in t.get("teams", [])]
        add_tournament(
            tournament_id=t["id"],
            name=t["name"],
            location=t["location"],
            start_date=t["start_date"],
            end_date=t["end_date"],
            venue=t.get("venue"),
            teams=team_ids,
            notes=t.get("notes")
        )
        print(f"  ✓ {t['name']} ({t['start_date']} - {t['end_date']})")
    print(f"  Added {len(TOURNAMENTS_2026)} tournaments")

def migrate_json_games():
    """Migrate existing JSON game data to database"""
    games_file = DATA_DIR / "games" / "all_games.json"
    
    if not games_file.exists():
        print("\n📄 No existing games JSON to migrate")
        return
    
    print("\n📄 Migrating JSON games...")
    with open(games_file) as f:
        data = json.load(f)
    
    count = 0
    for game in data.get("games", []):
        # Normalize team IDs
        home_id = game["home_team"].lower().replace(" ", "-")
        away_id = game["away_team"].lower().replace(" ", "-")
        
        # Make sure teams exist
        add_team(home_id, game["home_team"])
        add_team(away_id, game["away_team"])
        
        add_game(
            date=game["date"],
            home_team_id=home_id,
            away_team_id=away_id,
            home_score=game.get("home_score"),
            away_score=game.get("away_score"),
            is_conference_game=game.get("conference_game", False)
        )
        count += 1
    
    print(f"  ✓ Migrated {count} games")

def seed_mississippi_state_schedule():
    """Add Mississippi State 2026 schedule to database"""
    print("\n📅 Seeding Mississippi State schedule...")
    
    # Make sure we have MS State opponents as teams
    opponents = [
        {"id": "hofstra", "name": "Hofstra", "nickname": "Pride"},
        {"id": "troy", "name": "Troy", "nickname": "Trojans"},
        {"id": "alcorn-state", "name": "Alcorn State", "nickname": "Braves"},
        {"id": "delaware", "name": "Delaware", "nickname": "Blue Hens"},
        {"id": "austin-peay", "name": "Austin Peay", "nickname": "Governors"},
        {"id": "ucla", "name": "UCLA", "nickname": "Bruins"},
        {"id": "lipscomb", "name": "Lipscomb", "nickname": "Bisons"},
        {"id": "jackson-state", "name": "Jackson State", "nickname": "Tigers"},
        {"id": "southern-miss", "name": "Southern Miss", "nickname": "Golden Eagles"},
        {"id": "grambling-state", "name": "Grambling State", "nickname": "Tigers"},
        {"id": "uab", "name": "UAB", "nickname": "Blazers"},
        {"id": "memphis", "name": "Memphis", "nickname": "Tigers"},
        {"id": "nicholls-state", "name": "Nicholls State", "nickname": "Colonels"},
    ]
    
    for opp in opponents:
        add_team(opp["id"], opp["name"], opp.get("nickname"))
    
    # Mississippi State 2026 schedule
    ms_state_games = [
        # February - Non-conference
        {"date": "2026-02-13", "opponent": "hofstra", "home": True, "time": "4:00 PM"},
        {"date": "2026-02-14", "opponent": "hofstra", "home": True, "time": "1:00 PM"},
        {"date": "2026-02-15", "opponent": "hofstra", "home": True, "time": "1:00 PM"},
        {"date": "2026-02-17", "opponent": "troy", "home": True, "time": "4:00 PM"},
        {"date": "2026-02-18", "opponent": "alcorn-state", "home": True, "time": "4:00 PM"},
        {"date": "2026-02-20", "opponent": "delaware", "home": True, "time": "4:00 PM"},
        {"date": "2026-02-21", "opponent": "delaware", "home": True, "time": "1:00 PM"},
        {"date": "2026-02-22", "opponent": "delaware", "home": True, "time": "11:00 AM"},
        {"date": "2026-02-24", "opponent": "austin-peay", "home": True, "time": "4:00 PM"},
        # Amegy Bank Tournament (neutral site)
        {"date": "2026-02-27", "opponent": "tbd", "home": False, "time": "11:00 AM", "tournament": "amegy-bank-2026", "neutral": True},
        {"date": "2026-02-28", "opponent": "tbd", "home": False, "time": "3:00 PM", "tournament": "amegy-bank-2026", "neutral": True},
        {"date": "2026-03-01", "opponent": "ucla", "home": False, "time": "2:30 PM", "tournament": "amegy-bank-2026", "neutral": True},
        # March
        {"date": "2026-03-06", "opponent": "lipscomb", "home": True, "time": "4:00 PM"},
        {"date": "2026-03-07", "opponent": "lipscomb", "home": True, "time": "6:00 PM"},
        {"date": "2026-03-08", "opponent": "lipscomb", "home": True, "time": "1:00 PM"},
        {"date": "2026-03-17", "opponent": "jackson-state", "home": True, "time": "6:00 PM", "tournament": "hancock-whitney-2026"},
        # SEC Play begins
        {"date": "2026-03-20", "opponent": "vanderbilt", "home": True, "time": "7:00 PM", "conference": True},
        {"date": "2026-03-21", "opponent": "vanderbilt", "home": True, "time": "6:00 PM", "conference": True},
        {"date": "2026-03-22", "opponent": "vanderbilt", "home": True, "time": "1:00 PM", "conference": True},
        {"date": "2026-03-24", "opponent": "southern-miss", "home": True, "time": "6:00 PM"},
        {"date": "2026-03-31", "opponent": "grambling-state", "home": True, "time": "6:00 PM"},
        # April - SEC
        {"date": "2026-04-02", "opponent": "georgia", "home": True, "time": "6:00 PM", "conference": True},
        {"date": "2026-04-03", "opponent": "georgia", "home": True, "time": "6:00 PM", "conference": True},
        {"date": "2026-04-04", "opponent": "georgia", "home": True, "time": "1:00 PM", "conference": True},
        {"date": "2026-04-07", "opponent": "uab", "home": True, "time": "6:00 PM"},
        {"date": "2026-04-10", "opponent": "tennessee", "home": True, "time": "6:00 PM", "conference": True},
        {"date": "2026-04-11", "opponent": "tennessee", "home": True, "time": "6:00 PM", "conference": True},
        {"date": "2026-04-12", "opponent": "tennessee", "home": True, "time": "1:00 PM", "conference": True},
        {"date": "2026-04-21", "opponent": "memphis", "home": True, "time": "6:00 PM"},
        # Super Bulldog Weekend
        {"date": "2026-04-24", "opponent": "lsu", "home": True, "time": "6:00 PM", "conference": True, "tournament": "super-bulldog-weekend-2026"},
        {"date": "2026-04-25", "opponent": "lsu", "home": True, "time": "6:30 PM", "conference": True, "tournament": "super-bulldog-weekend-2026"},
        {"date": "2026-04-26", "opponent": "lsu", "home": True, "time": "1:00 PM", "conference": True, "tournament": "super-bulldog-weekend-2026"},
        # May
        {"date": "2026-05-05", "opponent": "nicholls-state", "home": True, "time": "6:00 PM"},
        {"date": "2026-05-07", "opponent": "auburn", "home": False, "time": "7:00 PM", "conference": True},
        {"date": "2026-05-08", "opponent": "auburn", "home": False, "time": "7:30 PM", "conference": True},
        {"date": "2026-05-09", "opponent": "auburn", "home": False, "time": "3:00 PM", "conference": True},
    ]
    
    count = 0
    for game in ms_state_games:
        opp_id = game["opponent"]
        if opp_id == "tbd":
            continue  # Skip TBD games for now
        
        if game["home"]:
            home_id = "mississippi-state"
            away_id = opp_id
        else:
            home_id = opp_id
            away_id = "mississippi-state"
        
        add_game(
            date=game["date"],
            home_team_id=home_id,
            away_team_id=away_id,
            time=game.get("time"),
            tournament_id=game.get("tournament"),
            is_neutral_site=game.get("neutral", False),
            is_conference_game=game.get("conference", False)
        )
        count += 1
    
    print(f"  ✓ Added {count} Mississippi State games")

def run_migration():
    """Run full migration"""
    print("="*50)
    print("College Baseball Database Migration")
    print("="*50)
    
    # Initialize if needed
    if not DB_PATH.exists():
        init_database()
    
    seed_sec_teams()
    seed_tournaments()
    migrate_json_games()
    seed_mississippi_state_schedule()
    
    print("\n" + "="*50)
    print("Migration complete!")
    print("="*50)
    
    # Show stats
    conn = get_connection()
    c = conn.cursor()
    
    c.execute("SELECT COUNT(*) FROM teams")
    print(f"Teams: {c.fetchone()[0]}")
    
    c.execute("SELECT COUNT(*) FROM games")
    print(f"Games: {c.fetchone()[0]}")
    
    c.execute("SELECT COUNT(*) FROM tournaments")
    print(f"Tournaments: {c.fetchone()[0]}")
    
    conn.close()

if __name__ == "__main__":
    run_migration()
