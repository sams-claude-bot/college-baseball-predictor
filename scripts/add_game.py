#!/usr/bin/env python3
"""
Quick game entry script - easier than the full collect_games.py interface

Features:
- Add game results quickly
- Automatically updates model accuracy tracking
- Records predictions vs outcomes for ensemble weight adjustment

Usage:
    python add_game.py "Winner" "Loser" winner_score loser_score [date] [--away]

Examples:
    # Mississippi State beats Hofstra 8-3 at home (today)
    python add_game.py "Mississippi State" "Hofstra" 8 3
    
    # Auburn beats Alabama 5-2 on the road
    python add_game.py "Auburn" "Alabama" 5 2 2026-02-15 --away
    
    # Add yesterday's game
    python add_game.py "LSU" "McNeese" 12 4 2026-02-12
"""

import sys
import json
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
GAMES_FILE = BASE_DIR / "data" / "games" / "all_games.json"
GAMES_FILE.parent.mkdir(parents=True, exist_ok=True)

# Add models to path
# sys.path.insert(0, str(BASE_DIR / "models"))  # Removed by cleanup
# sys.path.insert(0, str(BASE_DIR / "scripts"))  # Removed by cleanup

# SEC teams for auto-tagging conference games
SEC_TEAMS = [
    "alabama", "arkansas", "auburn", "florida", "georgia", "kentucky",
    "lsu", "mississippi state", "miss state", "missouri", "oklahoma",
    "ole miss", "south carolina", "tennessee", "texas", "texas a&m", "vanderbilt"
]

def is_sec_team(name):
    return any(sec in name.lower() for sec in SEC_TEAMS)

def load_games():
    if GAMES_FILE.exists():
        with open(GAMES_FILE) as f:
            return json.load(f)
    return {"games": [], "last_updated": None}

def save_games(data):
    data["last_updated"] = datetime.now().isoformat()
    with open(GAMES_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def update_model_accuracy(home_team, away_team, home_won, neutral_site=False):
    """
    Record the game result for model accuracy tracking.
    This updates the ensemble model's weight adjustments.
    """
    try:
        from models.ensemble_model import EnsembleModel
        
        ensemble = EnsembleModel()
        home_id = home_team.lower().replace(" ", "-")
        away_id = away_team.lower().replace(" ", "-")
        
        # Get predictions from each component model
        predictions = {}
        for name, model in ensemble.models.items():
            try:
                pred = model.predict_game(home_id, away_id, neutral_site)
                predictions[name] = pred['home_win_probability']
            except Exception:
                pass  # Skip models that fail
        
        if predictions:
            # Record the outcome
            game_id = f"{datetime.now().strftime('%Y-%m-%d')}_{away_id}_{home_id}"
            actual_winner = 'home' if home_won else 'away'
            
            ensemble.record_prediction(game_id, predictions, actual_winner)
            print(f"  ✓ Updated model accuracy tracking")
    except Exception as e:
        print(f"  ⚠️  Could not update model accuracy: {e}")

def add_game_to_database(home_team, away_team, home_score, away_score, date, 
                         neutral_site=False, conference_game=False):
    """Add game to SQLite database"""
    try:
        from scripts.database import add_game as db_add_game
        
        winner_id = home_team.lower().replace(" ", "-") if home_score > away_score else away_team.lower().replace(" ", "-")
        
        game_id = db_add_game(
            date=date,
            home_team_id=home_team.lower().replace(" ", "-"),
            away_team_id=away_team.lower().replace(" ", "-"),
            home_score=home_score,
            away_score=away_score,
            is_neutral_site=neutral_site,
            is_conference_game=conference_game,
            status='final'
        )
        
        # Update Elo ratings
        try:
            from models.elo_model import EloModel
            elo = EloModel()
            margin = home_score - away_score
            elo.update_ratings(
                home_team.lower().replace(" ", "-"),
                away_team.lower().replace(" ", "-"),
                home_won=(home_score > away_score),
                margin=margin
            )
            print(f"  ✓ Updated Elo ratings")
        except Exception as e:
            print(f"  ⚠️  Could not update Elo: {e}")
        
        return game_id
    except Exception as e:
        print(f"  ⚠️  Could not add to database: {e}")
        return None

def add_game(winner, loser, winner_score, loser_score, date=None, winner_away=False):
    """Add a game result"""
    
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    # Determine home/away
    if winner_away:
        home_team = loser
        away_team = winner
        home_score = loser_score
        away_score = winner_score
        home_won = False
    else:
        home_team = winner
        away_team = loser
        home_score = winner_score
        away_score = loser_score
        home_won = True
    
    # Check if SEC conference game
    conference_game = is_sec_team(home_team) and is_sec_team(away_team)
    
    game = {
        "id": f"{date}_{away_team}_{home_team}".lower().replace(" ", "-"),
        "date": date,
        "home_team": home_team,
        "away_team": away_team,
        "home_score": int(home_score),
        "away_score": int(away_score),
        "winner": winner,
        "loser": loser,
        "conference_game": conference_game,
        "added_at": datetime.now().isoformat()
    }
    
    data = load_games()
    
    # Check for duplicate
    existing_ids = [g["id"] for g in data["games"]]
    if game["id"] in existing_ids:
        print(f"⚠️  Game already exists: {game['id']}")
        return False
    
    data["games"].append(game)
    save_games(data)
    
    # Pretty output
    loc = "@" if winner_away else "vs"
    conf = " [SEC]" if conference_game else ""
    print(f"✓ Added: {winner} {winner_score}, {loser} {loser_score} ({date}){conf}")
    print(f"  {away_team} @ {home_team}")
    
    # Add to SQLite database
    add_game_to_database(home_team, away_team, int(home_score), int(away_score), 
                        date, conference_game=conference_game)
    
    # Update model accuracy tracking
    update_model_accuracy(home_team, away_team, home_won)
    
    return True

def show_recent(n=10):
    """Show recent games"""
    data = load_games()
    games = sorted(data["games"], key=lambda g: g["date"], reverse=True)[:n]
    
    print(f"\nRecent {len(games)} games:")
    print("-" * 50)
    for g in games:
        conf = " [SEC]" if g.get("conference_game") else ""
        print(f"  {g['date']}: {g['away_team']} {g['away_score']} @ {g['home_team']} {g['home_score']}{conf}")
        print(f"           Winner: {g['winner']}")

def show_model_accuracy():
    """Show current model accuracy stats"""
    try:
        from models.ensemble_model import EnsembleModel
        ensemble = EnsembleModel()
        print(ensemble.get_weights_report())
    except Exception as e:
        print(f"Could not load model accuracy: {e}")

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--accuracy":
        show_model_accuracy()
        return
    
    if len(sys.argv) < 5:
        print(__doc__)
        print("\nOptions:")
        print("  --away      Winner was the away team")
        print("  --accuracy  Show model accuracy stats")
        print("\nRecent games:")
        show_recent(5)
        return
    
    winner = sys.argv[1]
    loser = sys.argv[2]
    winner_score = int(sys.argv[3])
    loser_score = int(sys.argv[4])
    
    date = None
    winner_away = False
    
    for arg in sys.argv[5:]:
        if arg == "--away":
            winner_away = True
        elif arg.startswith("20"):  # Date like 2026-02-13
            date = arg
    
    add_game(winner, loser, winner_score, loser_score, date, winner_away)

if __name__ == "__main__":
    main()
