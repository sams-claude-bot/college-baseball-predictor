#!/usr/bin/env python3
"""
Daily data collection orchestrator

Run this daily to:
1. Collect NCAA stats
2. Update game results
3. Generate predictions for upcoming games
4. Save daily snapshot
"""

import json
import sys
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))
sys.path.insert(0, str(BASE_DIR / "models"))

from collect_ncaa_stats import fetch_ncaa_stats, save_stats
from track_mississippi_state import fetch_schedule, get_upcoming_games
from predictor import Predictor

def run_daily_collection():
    """Main daily collection routine"""
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"\n{'='*50}")
    print(f"Daily Collection - {today}")
    print('='*50)
    
    results = {
        "date": today,
        "timestamp": datetime.now().isoformat(),
        "ncaa_stats": False,
        "upcoming_games": [],
        "predictions": [],
        "errors": []
    }
    
    # 1. Collect NCAA stats
    print("\n[1/4] Collecting NCAA stats...")
    try:
        stats = fetch_ncaa_stats()
        if stats:
            save_stats(stats, f"ncaa_stats_{today}.json")
            results["ncaa_stats"] = True
            print("  ✓ NCAA stats collected")
        else:
            print("  ✗ Failed to collect NCAA stats")
            results["errors"].append("NCAA stats collection failed")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        results["errors"].append(str(e))
    
    # 2. Get upcoming Mississippi State games
    print("\n[2/4] Getting upcoming MS State games...")
    try:
        upcoming = get_upcoming_games(7)
        results["upcoming_games"] = upcoming
        print(f"  ✓ Found {len(upcoming)} upcoming games")
        for game in upcoming:
            loc = "vs" if game.get("home") else "@"
            print(f"    - {game['date']}: {loc} {game['opponent']}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        results["errors"].append(str(e))
    
    # 3. Generate predictions
    print("\n[3/4] Generating predictions...")
    try:
        predictor = Predictor()
        for game in upcoming[:3]:  # Next 3 games
            if game.get("home"):
                pred = predictor.predict_game("Mississippi State", game["opponent"])
            else:
                pred = predictor.predict_game(game["opponent"], "Mississippi State")
            
            results["predictions"].append({
                "date": game["date"],
                "matchup": f"{'Mississippi State' if game.get('home') else game['opponent']} vs {game['opponent'] if game.get('home') else 'Mississippi State'}",
                "prediction": pred
            })
            print(f"    - {game['date']} vs {game['opponent']}: {pred['predicted_winner']} ({pred['confidence']*100:.0f}% conf)")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        results["errors"].append(str(e))
    
    # 4. Save daily snapshot
    print("\n[4/4] Saving daily snapshot...")
    snapshot_dir = BASE_DIR / "data" / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_file = snapshot_dir / f"daily_{today}.json"
    
    with open(snapshot_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ Saved to {snapshot_file}")
    
    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print('='*50)
    print(f"Stats collected: {'Yes' if results['ncaa_stats'] else 'No'}")
    print(f"Upcoming games: {len(results['upcoming_games'])}")
    print(f"Predictions made: {len(results['predictions'])}")
    print(f"Errors: {len(results['errors'])}")
    
    if results["errors"]:
        print("\nErrors encountered:")
        for err in results["errors"]:
            print(f"  - {err}")
    
    return results

if __name__ == "__main__":
    run_daily_collection()
